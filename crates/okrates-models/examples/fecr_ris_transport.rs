//! Ensemble bulk single-defect transport driver for the Fe–Cr RIS model.
//!
//! Usage:
//! `fecr_ris_transport --defect <vacancy|dumbbell> --temperature K --cr-count N --runs R \
//! --t-end s --samples K --seed S [--t-equil s] [--cells nx ny nz] [--lattice-a m] \
//! [--max-steps N] [--fit <ols|origin>]`.

use std::env;
use std::process::exit;
use std::time::Instant;

use okrates_models::fecr_diffusion::FitMethod;
use okrates_models::fecr_ris::{DumbbellParams, FeCrRis, FeCrRisParams};
use okrates_models::fecr_ris_transport::{
    DefectKind, RisTransportConfig, run_ris_transport, summarize_ris_transport,
};

const DEFAULT_LATTICE_A_M: f64 = 2.8665e-10;
const USAGE: &str = "usage: fecr_ris_transport --defect <vacancy|dumbbell> --temperature K --cr-count N --runs R --t-end s --samples K --seed S [--t-equil s] [--cells nx ny nz] [--lattice-a m] [--max-steps N] [--fit <ols|origin>]";

fn usage() -> ! {
    eprintln!("{USAGE}");
    exit(2);
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("error: {message}");
    exit(1);
}

struct Args {
    defect: DefectKind,
    temperature_k: f64,
    cr_count: usize,
    runs: u64,
    t_equil: f64,
    t_end: f64,
    samples: usize,
    seed: u64,
    cells: [usize; 3],
    lattice_a: f64,
    max_steps: Option<u64>,
    fit_method: FitMethod,
}

fn parse() -> Args {
    let argv: Vec<String> = env::args().skip(1).collect();
    let (mut defect, mut temperature_k, mut cr_count, mut runs, mut t_end, mut samples, mut seed) =
        (None, None, None, None, None, None, None);
    let mut t_equil = 0.0_f64;
    let mut cells = [16_usize, 16, 8];
    let mut lattice_a = DEFAULT_LATTICE_A_M;
    let mut max_steps = None;
    let mut fit_method = FitMethod::WindowedOls;
    let mut i = 0;
    let next = |i: &mut usize| -> &str {
        *i += 1;
        argv.get(*i).map(String::as_str).unwrap_or_else(|| usage())
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--defect" => {
                defect = match next(&mut i) {
                    "vacancy" => Some(DefectKind::Vacancy),
                    "dumbbell" => Some(DefectKind::Dumbbell),
                    _ => usage(),
                };
            }
            "--temperature" => {
                temperature_k = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage()));
            }
            "--cr-count" => {
                cr_count = Some(next(&mut i).parse::<usize>().unwrap_or_else(|_| usage()));
            }
            "--runs" => {
                runs = Some(next(&mut i).parse::<u64>().unwrap_or_else(|_| usage()));
            }
            "--t-equil" => {
                t_equil = next(&mut i).parse::<f64>().unwrap_or_else(|_| usage());
            }
            "--t-end" => {
                t_end = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage()));
            }
            "--samples" => {
                samples = Some(next(&mut i).parse::<usize>().unwrap_or_else(|_| usage()));
            }
            "--seed" => {
                seed = Some(next(&mut i).parse::<u64>().unwrap_or_else(|_| usage()));
            }
            "--cells" => {
                for c in &mut cells {
                    *c = next(&mut i).parse::<usize>().unwrap_or_else(|_| usage());
                }
            }
            "--lattice-a" => {
                lattice_a = next(&mut i).parse::<f64>().unwrap_or_else(|_| usage());
            }
            "--max-steps" => {
                let parsed = next(&mut i).parse::<u64>().unwrap_or_else(|_| usage());
                if parsed == 0 {
                    usage();
                }
                max_steps = Some(parsed);
            }
            "--fit" => {
                fit_method = match next(&mut i) {
                    "ols" => FitMethod::WindowedOls,
                    "origin" => FitMethod::OriginSlope,
                    _ => usage(),
                };
            }
            _ => usage(),
        }
        i += 1;
    }

    match (defect, temperature_k, cr_count, runs, t_end, samples, seed) {
        (
            Some(defect),
            Some(temperature_k),
            Some(cr_count),
            Some(runs),
            Some(t_end),
            Some(samples),
            Some(seed),
        ) if temperature_k.is_finite()
            && temperature_k > 0.0
            && t_equil.is_finite()
            && t_equil >= 0.0
            && t_end.is_finite()
            && t_end > t_equil
            && lattice_a.is_finite()
            && lattice_a > 0.0
            && samples > 0 =>
        {
            Args {
                defect,
                temperature_k,
                cr_count,
                runs,
                t_equil,
                t_end,
                samples,
                seed,
                cells,
                lattice_a,
                max_steps,
                fit_method,
            }
        }
        _ => usage(),
    }
}

fn main() {
    let a = parse();
    let model = FeCrRis::new(FeCrRisParams {
        cells: a.cells,
        sink_planes: Vec::new(),
        temperature_k: a.temperature_k,
        k0_dpa_per_s: 0.0,
        capacity: 1,
        dumbbell: DumbbellParams::senninger2016_r1(),
    })
    .unwrap_or_else(|e| fail(e));

    let config = RisTransportConfig::windowed(a.t_equil, a.t_end, a.samples, a.lattice_a)
        .with_fit_method(a.fit_method)
        .with_max_steps(a.max_steps);

    let start = Instant::now();
    let runs = run_ris_transport(&model, a.defect, a.cr_count, a.seed, a.runs, &config)
        .unwrap_or_else(|e| fail(e));
    let site_count = model.lattice().site_count();
    let summary = summarize_ris_transport(&runs, a.defect, site_count, a.cr_count)
        .unwrap_or_else(|e| fail(e));
    let wall_s = start.elapsed().as_secs_f64();

    println!("defect={}", a.defect.name());
    println!("temperature_k={}", a.temperature_k);
    println!("cells={},{},{}", a.cells[0], a.cells[1], a.cells[2]);
    println!("site_count={site_count}");
    println!("cr_count={}", a.cr_count);
    println!("runs={}", a.runs);
    println!("fit={}", a.fit_method.name());
    println!("d_defect={:.12e}", summary.defect.0);
    println!("d_defect_se={:.12e}", summary.defect.1);
    println!("lambda_fe_fe={:.12e}", summary.fe_fe.0);
    println!("lambda_fe_cr={:.12e}", summary.fe_cr.0);
    println!("lambda_cr_cr={:.12e}", summary.cr_cr.0);
    println!("lambda_fe_d={:.12e}", summary.fe_defect.0);
    println!("lambda_cr_d={:.12e}", summary.cr_defect.0);
    println!("l_cr_d_over_l_fe_d={:.12e}", summary.l_cr_d_over_l_fe_d.0);
    println!(
        "l_cr_d_over_l_fe_d_se={:.12e}",
        summary.l_cr_d_over_l_fe_d.1
    );
    println!("l_fe_d_over_l_fe_fe={:.12e}", summary.l_fe_d_over_l_fe_fe.0);
    println!("l_cr_d_over_l_cr_cr={:.12e}", summary.l_cr_d_over_l_cr_cr.0);
    println!("d_cr_over_d_fe={:.12e}", summary.d_cr_over_d_fe.0);
    println!("d_cr_over_d_fe_se={:.12e}", summary.d_cr_over_d_fe.1);
    println!("mean_jumps={:.6}", summary.mean_jumps);
    println!("mean_floored_jumps={:.6}", summary.mean_floored_jumps);
    println!(
        "mean_initial_sro_alpha1={:.12e}",
        summary.mean_initial_sro_alpha1
    );
    println!(
        "mean_final_sro_alpha1={:.12e}",
        summary.mean_final_sro_alpha1
    );
    println!(
        "mean_delta_sro_alpha1={:.12e}",
        summary.mean_delta_sro_alpha1
    );
    println!("wall_s={wall_s:.4}");
}
