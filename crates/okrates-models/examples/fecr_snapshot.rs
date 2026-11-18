//! Writes extended-XYZ frames of one Fe–Cr (Martínez 2012) cell (`a = 2.855` Å)
//! across Kawasaki pre-annealing and subsequent single-vacancy diffusion.
//!
//! Usage: `fecr_snapshot <cr_count> <seed> <anneal_sweeps> <sweeps_per_frame> <jumps> <jumps_per_frame> <temperature_k>`.

use std::env;
use std::io::{self, BufWriter, Write};

use okrates_core::{KmcRng, Simulation, Step, StreamRole};
use okrates_models::bcc::BccLattice;
use okrates_models::fecr::{FeCrModel, Species};
use okrates_models::fecr_anneal::KawasakiAnnealer;
use okrates_models::fecr_martinez::{Martinez2012, params};

const A: f64 = 2.855;
const CELLS: usize = 6;

fn usage() -> ! {
    eprintln!(
        "usage: fecr_snapshot <cr_count> <seed> <anneal_sweeps> <sweeps_per_frame> <jumps> <jumps_per_frame> <temperature_k>"
    );
    std::process::exit(2);
}

fn fail(error: impl std::fmt::Display) -> ! {
    eprintln!("error: {error}");
    std::process::exit(1);
}

fn arg<T: std::str::FromStr>(s: &str) -> T {
    s.parse().unwrap_or_else(|_| usage())
}

fn frame(
    out: &mut impl Write,
    lattice: &BccLattice,
    occupancy: &[Species],
    stage: &str,
    step: u64,
    time: f64,
    disp: [i64; 3],
) -> io::Result<()> {
    let l = A * CELLS as f64;
    writeln!(out, "{}", occupancy.len())?;
    writeln!(
        out,
        "Lattice=\"{l} 0 0 0 {l} 0 0 0 {l}\" Properties=species:S:1:pos:R:3 pbc=\"T T T\" Stage={stage} Step={step} Time={time:e} VacancyDisp=\"{} {} {}\"",
        disp[0] as f64 * A / 2.0,
        disp[1] as f64 * A / 2.0,
        disp[2] as f64 * A / 2.0
    )?;
    for (site, species) in occupancy.iter().enumerate() {
        let [x, y, z] = lattice.position(site);
        let name = match species {
            Species::Fe => "Fe",
            Species::Cr => "Cr",
            Species::Vacancy => "V",
        };
        writeln!(
            out,
            "{name} {:.4} {:.4} {:.4}",
            x as f64 * A / 2.0,
            y as f64 * A / 2.0,
            z as f64 * A / 2.0
        )?;
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 7 {
        usage();
    }
    let cr: usize = arg(&args[0]);
    let seed: u64 = arg(&args[1]);
    let sweeps: u64 = arg(&args[2]);
    let per_frame: u64 = arg(&args[3]);
    let jumps: u64 = arg(&args[4]);
    let jumps_per_frame: u64 = arg(&args[5]);
    let temperature: f64 = arg(&args[6]);
    if per_frame == 0 || jumps_per_frame == 0 {
        usage();
    }
    let lattice = BccLattice::new([CELLS; 3]).unwrap_or_else(|e| fail(e));
    let energetics = Martinez2012::new(temperature).unwrap_or_else(|e| fail(e));
    let model = FeCrModel::new(lattice.clone(), energetics, params::NU_FE, temperature)
        .unwrap_or_else(|e| fail(e));
    let mut out = BufWriter::new(io::stdout().lock());

    let mut init_rng = KmcRng::for_replica(seed, 0, StreamRole::InitialState);
    let initial = model
        .random_state_with_rng(cr, &mut init_rng)
        .unwrap_or_else(|e| fail(e))
        .occupancy()
        .to_vec();
    frame(&mut out, &lattice, &initial, "anneal", 0, 0.0, [0; 3])?;

    let mut annealer = KawasakiAnnealer::new(
        &model,
        initial,
        KmcRng::for_replica(seed, 0, StreamRole::Equilibration),
    )
    .unwrap_or_else(|e| fail(e));

    let mut done = 0;
    while done < sweeps {
        let n = per_frame.min(sweeps - done);
        annealer.run_sweeps(n).unwrap_or_else(|e| fail(e));
        done += n;
        frame(
            &mut out,
            &lattice,
            annealer.occupancy(),
            "anneal",
            done,
            0.0,
            [0; 3],
        )?;
    }

    let state = model
        .state(annealer.finish().occupancy)
        .unwrap_or_else(|e| fail(e));
    let mut sim = Simulation::with_rng(
        model,
        state,
        KmcRng::for_replica(seed, 0, StreamRole::Dynamics),
    );
    for jump in 1..=jumps {
        match sim.step().unwrap_or_else(|e| fail(e)) {
            Step::Fired { .. } => {}
            Step::Absorbed => fail("Fe-Cr model unexpectedly absorbed"),
        }
        if jump % jumps_per_frame == 0 {
            let s = sim.state();
            frame(
                &mut out,
                &lattice,
                s.occupancy(),
                "diffusion",
                jump,
                sim.time(),
                s.vacancy_displacement(),
            )?;
        }
    }
    out.flush()
}
