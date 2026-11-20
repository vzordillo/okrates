//! Multi-ion site-exclusion diffusion on periodic SC, FCC, or 2D square crystals.

use std::env;
use std::process::exit;

use okrates_core::{
    AbsorptionPolicy, IncrementalSimulation, SampleGrid, StreamRole, run_ensemble,
    sample_trajectory,
};
use okrates_models::lattice_hopping::{LatticeHopping, PeriodicLattice};

const USAGE: &str = "usage: ionic_diffusion --lattice (sc|fcc|square2d) --cells nx ny nz --ions N --rate F --lattice-a M --t-end F --runs R --seed S";

fn usage() -> ! {
    eprintln!("{USAGE}");
    exit(2);
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("error: {message}");
    exit(1);
}

#[derive(Clone, Copy)]
enum LatticeKind {
    Sc,
    Fcc,
    Square2d,
}

impl LatticeKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Sc => "sc",
            Self::Fcc => "fcc",
            Self::Square2d => "square2d",
        }
    }
}

struct Args {
    lattice: LatticeKind,
    cells: [usize; 3],
    ions: usize,
    rate: f64,
    lattice_a: f64,
    t_end: f64,
    runs: usize,
    seed: u64,
}

fn parse() -> Args {
    let argv: Vec<String> = env::args().skip(1).collect();
    let (mut lattice, mut cells, mut ions, mut rate, mut lattice_a, mut t_end, mut runs, mut seed) =
        (None, None, None, None, None, None, None, None);

    let mut i = 0;
    let next = |i: &mut usize| -> &str {
        *i += 1;
        argv.get(*i).map(String::as_str).unwrap_or_else(|| usage())
    };

    while i < argv.len() {
        match argv[i].as_str() {
            "--lattice" => {
                lattice = Some(match next(&mut i) {
                    "sc" => LatticeKind::Sc,
                    "fcc" => LatticeKind::Fcc,
                    "square2d" => LatticeKind::Square2d,
                    _ => usage(),
                });
            }
            "--cells" => {
                let nx = next(&mut i).parse::<usize>().unwrap_or_else(|_| usage());
                let ny = next(&mut i).parse::<usize>().unwrap_or_else(|_| usage());
                let nz = next(&mut i).parse::<usize>().unwrap_or_else(|_| usage());
                cells = Some([nx, ny, nz]);
            }
            "--ions" => ions = Some(next(&mut i).parse::<usize>().unwrap_or_else(|_| usage())),
            "--rate" => rate = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage())),
            "--lattice-a" => {
                lattice_a = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage()));
            }
            "--t-end" => t_end = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage())),
            "--runs" => runs = Some(next(&mut i).parse::<usize>().unwrap_or_else(|_| usage())),
            "--seed" => seed = Some(next(&mut i).parse::<u64>().unwrap_or_else(|_| usage())),
            _ => usage(),
        }
        i += 1;
    }

    let args = Args {
        lattice: lattice.unwrap_or_else(|| usage()),
        cells: cells.unwrap_or_else(|| usage()),
        ions: ions.unwrap_or_else(|| usage()),
        rate: rate.unwrap_or_else(|| usage()),
        lattice_a: lattice_a.unwrap_or_else(|| usage()),
        t_end: t_end.unwrap_or_else(|| usage()),
        runs: runs.unwrap_or_else(|| usage()),
        seed: seed.unwrap_or_else(|| usage()),
    };
    if args.runs == 0 || !args.t_end.is_finite() || args.t_end <= 0.0 {
        usage();
    }
    if matches!(args.lattice, LatticeKind::Square2d) && args.cells[2] != 1 {
        usage();
    }
    args
}

struct ReplicaResult {
    tracer_msd: f64,
    collective_msd: f64,
    steps: u64,
}

fn main() {
    let args = parse();
    let lattice = match args.lattice {
        LatticeKind::Sc => PeriodicLattice::simple_cubic(args.cells, args.lattice_a),
        LatticeKind::Fcc => PeriodicLattice::fcc(args.cells, args.lattice_a),
        LatticeKind::Square2d => {
            PeriodicLattice::square_2d([args.cells[0], args.cells[1]], args.lattice_a)
        }
    }
    .unwrap_or_else(|err| fail(err));

    let dim = lattice.spatial_dimension();
    let sites = lattice.site_count();
    let model = LatticeHopping::homogeneous(lattice, args.rate).unwrap_or_else(|err| fail(err));
    let grid = SampleGrid::from_origin(args.t_end, 1).unwrap_or_else(|err| fail(err));

    let outcomes = run_ensemble(args.seed, args.runs as u64, |ctx| {
        let mut init_rng = ctx.rng(StreamRole::InitialState);
        let init = model
            .random_state_with_rng(args.ions, &mut init_rng)
            .map_err(|err| err.to_string())?;
        let mut sim =
            IncrementalSimulation::with_rng(model.clone(), init, ctx.rng(StreamRole::Dynamics))
                .map_err(|err| err.to_string())?;
        let traj = sample_trajectory(
            &mut sim,
            &grid,
            None,
            AbsorptionPolicy::HoldFinal,
            |m, st| {
                m.check_invariants(st).map(|()| {
                    (
                        st.mean_tracer_msd_m2(m.lattice()),
                        st.collective_msd_m2(m.lattice()),
                    )
                })
            },
        )
        .map_err(|err| err.to_string())?;
        let (tracer_msd, collective_msd) = traj
            .values
            .into_iter()
            .next()
            .ok_or_else(|| "missing sampled endpoint".to_string())?
            .map_err(|err| err.to_string())?;
        Ok::<ReplicaResult, String>(ReplicaResult {
            tracer_msd,
            collective_msd,
            steps: traj.fired_steps,
        })
    });

    let mut sum_tr = 0.0_f64;
    let mut sum_coll = 0.0_f64;
    let mut sum_steps = 0_u128;
    for outcome in outcomes {
        let outcome = outcome.unwrap_or_else(|err| fail(err));
        sum_tr += outcome.tracer_msd;
        sum_coll += outcome.collective_msd;
        sum_steps += u128::from(outcome.steps);
    }

    let n = args.runs as f64;
    let mean_tr = sum_tr / n;
    let mean_coll = sum_coll / n;
    let denom = 2.0 * (dim as f64) * args.t_end;
    let d_tracer = mean_tr / denom;
    let d_coll = mean_coll / denom;
    let haven_ratio = if d_coll > 0.0 {
        d_tracer / d_coll
    } else {
        f64::NAN
    };
    let occupancy = (args.ions as f64) / (sites as f64);

    println!(
        "lattice = {} dim = {} cells = {:?} sites = {} ions = {} theta = {:.6}",
        args.lattice.as_str(),
        dim,
        args.cells,
        sites,
        args.ions,
        occupancy
    );
    println!(
        "rate = {} a_m = {:e} t_end = {} runs = {} seed = {}",
        args.rate, args.lattice_a, args.t_end, args.runs, args.seed
    );
    println!(
        "mean_tracer_msd_m2 = {:.12e} mean_collective_msd_m2 = {:.12e}",
        mean_tr, mean_coll
    );
    println!(
        "D_tracer_m2_s = {:.12e} D_collective_m2_s = {:.12e} H_R = {:.6} mean_steps = {:.6}",
        d_tracer,
        d_coll,
        haven_ratio,
        (sum_steps as f64) / n
    );
    println!("invariants = verified");
}
