//! Round-trip output for the Python parity test: the observables of one Fe–Cr
//! composition with the Martínez 2012 energetics, optionally pre-annealed.
//!
//! Usage: `fecr_observables_martinez <cr_count> <seed_base> <runs>=2 <anneal_sweeps>`.

use std::env;

use okrates_models::bcc::BccLattice;
use okrates_models::fecr::FeCrModel;
use okrates_models::fecr_diffusion::{run_composition, summarize};
use okrates_models::fecr_martinez::{Martinez2012, params};
use okrates_models::vacancy::BOLTZMANN_EV_PER_K;

fn usage() -> ! {
    eprintln!("usage: fecr_observables_martinez <cr_count> <seed_base> <runs>=2 <anneal_sweeps>");
    std::process::exit(2);
}

fn parse_args() -> (usize, u64, u64, u64) {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 4 {
        usage();
    }
    let cr_count = args[0].parse::<usize>().unwrap_or_else(|_| usage());
    let seed_base = args[1].parse::<u64>().unwrap_or_else(|_| usage());
    let runs = args[2].parse::<u64>().unwrap_or_else(|_| usage());
    let anneal_sweeps = args[3].parse::<u64>().unwrap_or_else(|_| usage());
    if runs < 2 {
        usage();
    }
    (cr_count, seed_base, runs, anneal_sweeps)
}

fn main() {
    let (cr_count, seed_base, runs, anneal_sweeps) = parse_args();
    let (temperature, k_count) = (800.0, 10);
    let kt = BOLTZMANN_EV_PER_K * temperature;
    let r0 = params::NU_FE * params::DS_MIG.exp() * (-params::DH0_FE / kt).exp();
    let t_end = 5000.0 / (8.0 * r0);
    let model = FeCrModel::new(
        BccLattice::new([6, 6, 6]).expect("valid cells"),
        Martinez2012::new(temperature).expect("valid energetics"),
        params::NU_FE,
        temperature,
    )
    .expect("valid model");
    let replicas = match run_composition(
        &model,
        cr_count,
        seed_base,
        runs,
        t_end,
        k_count,
        1.0,
        anneal_sweeps,
    ) {
        Ok(replicas) => replicas,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    };
    let summary = summarize(&replicas, cr_count, 431);
    println!("t_end {t_end:?}");
    let rows = [
        ("vacancy", summary.vacancy),
        ("tracer_fe", summary.tracer[0]),
        ("tracer_cr", summary.tracer[1]),
        ("tracer_all", summary.tracer_all),
        ("lambda_fefe", summary.collective[0]),
        ("lambda_fecr", summary.collective[1]),
        ("lambda_crcr", summary.collective[2]),
        ("f", summary.f),
    ];
    for (key, (mean, se)) in rows {
        println!("{key} {mean:?} {se:?}");
    }
    println!("mean_jumps {:?}", summary.mean_jumps);
}
