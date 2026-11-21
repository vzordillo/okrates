//! Langmuir-Hinshelwood CO oxidation catalysis on a 2D square catalyst surface.

use std::env;
use std::process::exit;

use okrates_core::{
    AbsorptionPolicy, IncrementalSimulation, SampleGrid, StreamRole, run_ensemble,
    sample_trajectory,
};
use okrates_models::lattice_gas::{LatticeGas, LatticeGasEnergetics, PairRule, SingleSiteRule};
use okrates_models::lattice_hopping::PeriodicLattice;

const USAGE: &str =
    "usage: surface_catalysis [seed] [runs] [t_end] [nx] [ny] [p_co_bar] [p_o2_bar]";

fn usage() -> ! {
    eprintln!("{USAGE}");
    exit(2);
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("error: {message}");
    exit(1);
}

struct Args {
    seed: u64,
    runs: usize,
    t_end: f64,
    nx: usize,
    ny: usize,
    p_co_bar: f64,
    p_o2_bar: f64,
}

fn parse() -> Args {
    let argv: Vec<String> = env::args().skip(1).collect();
    if argv.len() > 7 {
        usage();
    }
    let parse_arg = |idx: usize| -> Option<&str> { argv.get(idx).map(String::as_str) };

    let seed = parse_arg(0)
        .map(|s| s.parse::<u64>().unwrap_or_else(|_| usage()))
        .unwrap_or(19);
    let runs = parse_arg(1)
        .map(|s| s.parse::<usize>().unwrap_or_else(|_| usage()))
        .unwrap_or(4);
    let t_end = parse_arg(2)
        .map(|s| s.parse::<f64>().unwrap_or_else(|_| usage()))
        .unwrap_or(0.05);
    let nx = parse_arg(3)
        .map(|s| s.parse::<usize>().unwrap_or_else(|_| usage()))
        .unwrap_or(4);
    let ny = parse_arg(4)
        .map(|s| s.parse::<usize>().unwrap_or_else(|_| usage()))
        .unwrap_or(4);
    let p_co_bar = parse_arg(5)
        .map(|s| s.parse::<f64>().unwrap_or_else(|_| usage()))
        .unwrap_or(1.0);
    let p_o2_bar = parse_arg(6)
        .map(|s| s.parse::<f64>().unwrap_or_else(|_| usage()))
        .unwrap_or(1.0);

    if runs == 0 || !t_end.is_finite() || t_end <= 0.0 {
        usage();
    }
    Args {
        seed,
        runs,
        t_end,
        nx,
        ny,
        p_co_bar,
        p_o2_bar,
    }
}

struct ReplicaOutcome {
    theta_co: f64,
    theta_o: f64,
    theta_empty: f64,
    tof_hz: f64,
    co2_events: u64,
    steps: u64,
}

fn main() {
    let args = parse();
    let lattice =
        PeriodicLattice::square_2d([args.nx, args.ny], 2.8e-10).unwrap_or_else(|err| fail(err));
    let sites = lattice.site_count();

    let single_rules = vec![
        SingleSiteRule::adsorption(0, args.p_co_bar, 12.0).unwrap_or_else(|err| fail(err)),
        SingleSiteRule::desorption(0, 3.5).unwrap_or_else(|err| fail(err)),
    ];
    let pair_rules = vec![
        PairRule::dissociative_adsorption(1, 1, args.p_o2_bar, 18.0)
            .unwrap_or_else(|err| fail(err)),
        PairRule::associative_desorption(1, 1, 1.5).unwrap_or_else(|err| fail(err)),
        PairRule::diffusion(0, 25.0).unwrap_or_else(|err| fail(err)),
        PairRule::diffusion(1, 8.0).unwrap_or_else(|err| fail(err)),
        PairRule::reaction((Some(0), Some(1)), (None, None), 40.0).unwrap_or_else(|err| fail(err)),
    ];
    let ener = LatticeGasEnergetics {
        temperature_k: 550.0,
        site_energies_ev: vec![-0.05, -0.02],
        pair_interactions_ev: vec![0.04, -0.02, -0.02, 0.06],
    };

    let model = LatticeGas::new(lattice, 2, single_rules, pair_rules, Some(ener))
        .unwrap_or_else(|err| fail(err));
    let init = model.empty_state();
    let grid = SampleGrid::from_origin(args.t_end, 1).unwrap_or_else(|err| fail(err));

    let outcomes = run_ensemble(args.seed, args.runs as u64, |ctx| {
        let mut sim = IncrementalSimulation::with_rng(
            model.clone(),
            init.clone(),
            ctx.rng(StreamRole::Dynamics),
        )
        .map_err(|err| err.to_string())?;
        let traj = sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |m, st| {
            m.check_invariants(st).map_err(|err| err.to_string())?;
            let n_co = st
                .species_count_on_surface(0)
                .ok_or_else(|| "missing CO count".to_string())?;
            let n_o = st
                .species_count_on_surface(1)
                .ok_or_else(|| "missing O count".to_string())?;
            let co_ads = st.single_rule_count(0).unwrap_or(0);
            let co_des = st.single_rule_count(1).unwrap_or(0);
            let o2_ads = st.pair_rule_count(0).unwrap_or(0);
            let o2_des = st.pair_rule_count(1).unwrap_or(0);
            let co2_lh = st.pair_rule_count(4).unwrap_or(0);
            if co_ads != (n_co as u64) + co_des + co2_lh
                || 2 * o2_ads != (n_o as u64) + 2 * o2_des + co2_lh
            {
                return Err("stoichiometric surface balance violated".to_string());
            }
            let tof_hz = st
                .turnover_frequency_per_site_hz(args.t_end, &[], &[4])
                .ok_or_else(|| "invalid TOF denominator".to_string())?;
            Ok::<_, String>((
                st.coverage(0).unwrap_or(0.0),
                st.coverage(1).unwrap_or(0.0),
                st.empty_fraction(),
                tof_hz,
                co2_lh,
            ))
        })
        .map_err(|err| err.to_string())?;
        let (theta_co, theta_o, theta_empty, tof_hz, co2_events) = traj.values[0].clone()?;
        Ok::<_, String>(ReplicaOutcome {
            theta_co,
            theta_o,
            theta_empty,
            tof_hz,
            co2_events,
            steps: traj.steps_in_window,
        })
    });

    let mut mean_co = 0.0;
    let mut mean_o = 0.0;
    let mut mean_empty = 0.0;
    let mut mean_tof = 0.0;
    let mut mean_co2 = 0.0;
    let mut mean_steps = 0.0;
    for res in &outcomes {
        let r = match res {
            Ok(val) => val,
            Err(err) => fail(err),
        };
        mean_co += r.theta_co;
        mean_o += r.theta_o;
        mean_empty += r.theta_empty;
        mean_tof += r.tof_hz;
        mean_co2 += r.co2_events as f64;
        mean_steps += r.steps as f64;
    }
    let n = outcomes.len() as f64;
    mean_co /= n;
    mean_o /= n;
    mean_empty /= n;
    mean_tof /= n;
    mean_co2 /= n;
    mean_steps /= n;

    println!(
        "surface = square2d cells = [{}, {}] sites = {} T = 550 p_CO = {} p_O2 = {}",
        args.nx, args.ny, sites, args.p_co_bar, args.p_o2_bar
    );
    println!(
        "t_end = {} runs = {} seed = {}",
        args.t_end, args.runs, args.seed
    );
    println!(
        "mean_theta_CO = {mean_co:.6} mean_theta_O = {mean_o:.6} mean_theta_empty = {mean_empty:.6}"
    );
    println!(
        "mean_tof_per_site_hz = {mean_tof:.6} mean_co2_events = {mean_co2:.6} mean_steps = {mean_steps:.6}"
    );
    println!("conservation = verified");
}
