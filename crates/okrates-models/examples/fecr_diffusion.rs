use std::env;

use okrates_core::ensemble::{run_replicas, thread_count};
use okrates_models::bcc::BccLattice;
use okrates_models::fecr::{FeCrModel, PairEnergetics};
use okrates_models::fecr_diffusion::{
    ANNEAL_SEED_OFFSET, ReplicaEstimate, STATE_SEED_OFFSET, estimate, mean_se, ols_slope,
    origin_slope, run_composition, sample_run,
};
use okrates_models::vacancy::BOLTZMANN_EV_PER_K;

fn env_usize(name: &str, default: usize) -> usize {
    match env::var(name) {
        Ok(value) => match value.parse::<usize>() {
            Ok(v) if v >= 2 => v,
            _ => {
                eprintln!("{name} must be an integer >= 2");
                std::process::exit(2);
            }
        },
        Err(_) => default,
    }
}

/// Reads `ANNEAL_SWEEPS`: unset means no anneal, otherwise an integer >= 1.
fn env_anneal_sweeps() -> u64 {
    match env::var("ANNEAL_SWEEPS") {
        Ok(value) => match value.parse::<u64>() {
            Ok(v) if v >= 1 => v,
            _ => {
                eprintln!("ANNEAL_SWEEPS must be an integer >= 1");
                std::process::exit(2);
            }
        },
        Err(_) => 0,
    }
}

fn report(name: &str, values: &[f64]) {
    let (mean, se) = mean_se(values);
    println!("{name} = {mean:.12e}");
    println!("{name}_se = {se:.12e}");
}

fn correlation_factor() {
    let runs = env_usize("RUNS_F", 500);
    let seed_base = 20261120_u64;
    let (n_sites, n_atoms, n_cr, k_count, t_end) = (432.0, 431.0, 215, 20, 2500.0);
    let energetics = PairEnergetics::new([[0.0; 3]; 3], 0.0, 0.0).expect("valid energetics");
    let model = FeCrModel::new(
        BccLattice::new([6, 6, 6]).expect("valid cells"),
        energetics,
        1.0,
        1000.0,
    )
    .expect("valid model");
    let replicas = run_replicas(seed_base, runs as u64, |_, seed| {
        let state = model
            .random_state(n_cr, seed + STATE_SEED_OFFSET)
            .expect("valid state");
        let (times, samples, fired) = sample_run(model.clone(), state, seed, t_end, k_count, 1.0);
        let d_v = estimate(&times, &samples, origin_slope).vacancy;
        let late = estimate(&times[10..], &samples[10..], ols_slope);
        (d_v, late.tracer_all, fired)
    })
    .expect("seeds do not overflow");
    let mut d_v = Vec::with_capacity(runs);
    let mut d_all = Vec::with_capacity(runs);
    let mut f_n = Vec::with_capacity(runs);
    let mut total_fired = 0_u64;
    for replica in replicas {
        let (v, t, fired) = replica.value;
        d_v.push(v);
        d_all.push(t);
        f_n.push(n_atoms * t);
        total_fired += fired;
    }
    println!(
        "# correlation factor, equal-energy limit (all rates 1), N = {n_sites}, N_Cr = {n_cr}"
    );
    report("D_v", &d_v);
    report("D_tracer_all", &d_all);
    report("f_N", &f_n);
    println!("runs = {runs}");
    println!("seed_base = {seed_base}");
    println!("total_fired_steps = {total_fired}");
}

fn illustrative(anneal_sweeps: u64) {
    let runs = env_usize("RUNS", 500);
    let seed_base = 20261121_u64;
    let (nu, temperature, k_count) = (1e13, 800.0, 10);
    let n_sites = 432.0;
    let r0 = nu * (-0.60 / (BOLTZMANN_EV_PER_K * temperature)).exp();
    let t_end = 5000.0 / (8.0 * r0);
    let model = FeCrModel::new(
        BccLattice::new([6, 6, 6]).expect("valid cells"),
        PairEnergetics::illustrative(),
        nu,
        temperature,
    )
    .expect("valid model");
    println!("# ILLUSTRATIVE parameters: no physical claims");
    println!("nu = {nu:.12e}");
    println!("T = {temperature:.12e}");
    println!("r0 = {r0:.12e}");
    println!("t_end = {t_end:.12e}");
    println!("units = a^2/s with a = 1");
    println!("runs = {runs}");
    println!("seed_base = {seed_base}");
    if anneal_sweeps > 0 {
        println!("anneal_sweeps = {anneal_sweeps}");
        println!("anneal_seed_offset = {ANNEAL_SEED_OFFSET}");
    }
    for n_cr in [0_usize, 22, 43, 86] {
        let replicas = run_composition(
            &model,
            n_cr,
            seed_base,
            runs as u64,
            t_end,
            k_count,
            1.0,
            anneal_sweeps,
        )
        .expect("valid state");
        let estimates: Vec<ReplicaEstimate> = replicas.iter().map(|r| r.estimate).collect();
        let total_fired: u64 = replicas.iter().map(|r| r.fired).sum();
        let column =
            |f: &dyn Fn(&ReplicaEstimate) -> f64| -> Vec<f64> { estimates.iter().map(f).collect() };
        println!("n_cr = {n_cr}");
        println!("x_cr = {:.12e}", n_cr as f64 / (n_sites - 1.0));
        println!(
            "mean_jumps_per_run = {:.12e}",
            total_fired as f64 / runs as f64
        );
        if anneal_sweeps > 0 {
            let per_site = |index: u64| -> Vec<f64> {
                replicas
                    .iter()
                    .map(|r| r.anneal_energy[index as usize] / n_sites)
                    .collect()
            };
            report("anneal_E_site_q1", &per_site(anneal_sweeps / 4));
            report("anneal_E_site_half", &per_site(anneal_sweeps / 2));
            report("anneal_E_site_end", &per_site(anneal_sweeps));
        }
        report("D_v", &column(&|e| e.vacancy));
        report("D_Fe", &column(&|e| e.tracer[0]));
        if n_cr == 0 {
            println!("D_Cr = NaN");
            println!("D_Cr_se = NaN");
        } else {
            report("D_Cr", &column(&|e| e.tracer[1]));
        }
        report("Lambda_FeFe", &column(&|e| e.collective[0]));
        report("Lambda_FeCr", &column(&|e| e.collective[1]));
        report("Lambda_CrCr", &column(&|e| e.collective[2]));
        report("N_D_Fe", &column(&|e| n_sites * e.tracer[0]));
        if n_cr == 0 {
            println!("N_D_Cr = NaN");
            println!("N_D_Cr_se = NaN");
        } else {
            report("N_D_Cr", &column(&|e| n_sites * e.tracer[1]));
        }
    }
}

fn main() {
    let anneal_sweeps = env_anneal_sweeps();
    correlation_factor();
    illustrative(anneal_sweeps);
    eprintln!("threads = {}", thread_count());
}
