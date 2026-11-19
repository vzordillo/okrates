//! Ensemble grain-boundary radiation-induced segregation (RIS) driver for [`FeCrRis`].
//!
//! Computes `C_Cr^GB(dpa)`, `C_Cr^bulk(dpa)`, Gibbsian Cr excess `S_Cr^tot(dpa)` (nm),
//! signed-distance spatial Cr and steady-state V/I concentration profiles, and optional CSV
//! outputs for grain-boundary RIS studies.

use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

use okrates_models::fecr_ris::{DumbbellParams, FeCrRis, FeCrRisParams};
use okrates_models::fecr_ris_segregation::{
    DEFAULT_LATTICE_A_M, RisSegregationConfig, RisSegregationSummary,
    default_bulk_min_distance_planes, fit_exponential_relaxation, run_ris_segregation,
    summarize_ris_segregation,
};

const USAGE: &str = "usage: fecr_ris_segregation --temperature K --k0 dpa_per_s (--cr-frac f | --cr-count N) --dpa-end x --capacity P --runs R --samples K --seed S [--cells nx ny nz] [--sink-plane x] [--bulk-min-planes n] [--steady-start-dpa x] [--lattice-a m] [--max-steps N] [--timeseries-out path] [--profile-out path]";

fn usage() -> ! {
    eprintln!("{USAGE}");
    exit(2);
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("error: {message}");
    exit(1);
}

enum CrSpec {
    Frac(f64),
    Count(usize),
}

struct Args {
    temperature_k: f64,
    k0: f64,
    cr_spec: CrSpec,
    dpa_end: f64,
    capacity: usize,
    runs: usize,
    samples: usize,
    seed: u64,
    cells: [usize; 3],
    sink_plane: i64,
    bulk_min_planes: Option<usize>,
    steady_start_dpa: f64,
    lattice_a: f64,
    max_steps: Option<u64>,
    timeseries_out: Option<PathBuf>,
    profile_out: Option<PathBuf>,
}

fn parse() -> Args {
    let argv: Vec<String> = env::args().skip(1).collect();
    let (mut temperature_k, mut k0, mut dpa_end, mut capacity, mut runs, mut samples, mut seed) =
        (None, None, None, None, None, None, None);
    let mut cr_frac: Option<f64> = None;
    let mut cr_count: Option<usize> = None;
    let mut cells = [128_usize, 32, 32];
    let mut sink_plane = 0_i64;
    let mut bulk_min_planes = None;
    let mut steady_start_dpa = 0.0_f64;
    let mut lattice_a = DEFAULT_LATTICE_A_M;
    let mut max_steps = None;
    let mut timeseries_out = None;
    let mut profile_out = None;
    let mut i = 0;

    let next = |i: &mut usize| -> &str {
        *i += 1;
        argv.get(*i).map(String::as_str).unwrap_or_else(|| usage())
    };

    while i < argv.len() {
        match argv[i].as_str() {
            "--temperature" => {
                temperature_k = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage()));
            }
            "--k0" => k0 = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage())),
            "--cr-frac" => cr_frac = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage())),
            "--cr-count" => {
                cr_count = Some(next(&mut i).parse::<usize>().unwrap_or_else(|_| usage()));
            }
            "--dpa-end" => dpa_end = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage())),
            "--capacity" => {
                capacity = Some(next(&mut i).parse::<usize>().unwrap_or_else(|_| usage()));
            }
            "--runs" => runs = Some(next(&mut i).parse::<usize>().unwrap_or_else(|_| usage())),
            "--samples" => {
                samples = Some(next(&mut i).parse::<usize>().unwrap_or_else(|_| usage()));
            }
            "--seed" => seed = Some(next(&mut i).parse::<u64>().unwrap_or_else(|_| usage())),
            "--cells" => {
                for c in &mut cells {
                    *c = next(&mut i).parse::<usize>().unwrap_or_else(|_| usage());
                }
            }
            "--sink-plane" => {
                sink_plane = next(&mut i).parse::<i64>().unwrap_or_else(|_| usage());
            }
            "--bulk-min-planes" => {
                bulk_min_planes = Some(next(&mut i).parse::<usize>().unwrap_or_else(|_| usage()));
            }
            "--steady-start-dpa" => {
                steady_start_dpa = next(&mut i).parse::<f64>().unwrap_or_else(|_| usage());
            }
            "--lattice-a" => lattice_a = next(&mut i).parse::<f64>().unwrap_or_else(|_| usage()),
            "--max-steps" => {
                let parsed = next(&mut i).parse::<u64>().unwrap_or_else(|_| usage());
                if parsed == 0 {
                    usage();
                }
                max_steps = Some(parsed);
            }
            "--timeseries-out" => timeseries_out = Some(PathBuf::from(next(&mut i))),
            "--profile-out" => profile_out = Some(PathBuf::from(next(&mut i))),
            _ => usage(),
        }
        i += 1;
    }

    let cr_spec = match (cr_frac, cr_count) {
        (Some(f), None) if f.is_finite() && (0.0..=1.0).contains(&f) => CrSpec::Frac(f),
        (None, Some(n)) => CrSpec::Count(n),
        _ => usage(),
    };

    match (temperature_k, k0, dpa_end, capacity, runs, samples, seed) {
        (
            Some(temperature_k),
            Some(k0),
            Some(dpa_end),
            Some(capacity),
            Some(runs),
            Some(samples),
            Some(seed),
        ) if temperature_k.is_finite()
            && temperature_k > 0.0
            && k0.is_finite()
            && k0 > 0.0
            && dpa_end.is_finite()
            && dpa_end > 0.0
            && capacity > 0
            && runs > 0
            && samples > 0
            && steady_start_dpa.is_finite()
            && steady_start_dpa >= 0.0
            && steady_start_dpa < dpa_end
            && lattice_a.is_finite()
            && lattice_a > 0.0 =>
        {
            Args {
                temperature_k,
                k0,
                cr_spec,
                dpa_end,
                capacity,
                runs,
                samples,
                seed,
                cells,
                sink_plane,
                bulk_min_planes,
                steady_start_dpa,
                lattice_a,
                max_steps,
                timeseries_out,
                profile_out,
            }
        }
        _ => usage(),
    }
}

fn open_output_file(path: &Path) -> Result<fs::File, std::io::Error> {
    fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

fn write_timeseries_csv(
    file: fs::File,
    summary: &RisSegregationSummary,
) -> Result<(), std::io::Error> {
    file.set_len(0)?;
    let mut f = BufWriter::new(file);
    writeln!(
        f,
        "dose_dpa,c_cr_gb,c_cr_gb_se,c_cr_bulk,c_cr_bulk_se,s_cr_tot_nm,s_cr_tot_nm_se,s_cr_linear_nm,s_cr_linear_nm_se"
    )?;
    for k in 0..summary.doses.len() {
        writeln!(
            f,
            "{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            summary.doses[k],
            summary.c_cr_gb[k].0,
            summary.c_cr_gb[k].1,
            summary.c_cr_bulk[k].0,
            summary.c_cr_bulk[k].1,
            summary.s_cr_tot_nm[k].0,
            summary.s_cr_tot_nm[k].1,
            summary.s_cr_linear_nm[k].0,
            summary.s_cr_linear_nm[k].1
        )?;
    }
    f.flush()
}

fn write_profile_csv(
    file: fs::File,
    summary: &RisSegregationSummary,
) -> Result<(), std::io::Error> {
    file.set_len(0)?;
    let mut f = BufWriter::new(file);
    writeln!(
        f,
        "d_planes,distance_nm,c_cr_mean,c_cr_se,c_v_mean,c_v_se,c_i_mean,c_i_se"
    )?;
    for p in 0..summary.d_planes.len() {
        writeln!(
            f,
            "{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            summary.d_planes[p],
            summary.distance_nm[p],
            summary.profile_c_cr[p].0,
            summary.profile_c_cr[p].1,
            summary.profile_c_v[p].0,
            summary.profile_c_v[p].1,
            summary.profile_c_i[p].0,
            summary.profile_c_i[p].1
        )?;
    }
    f.flush()
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.12e}"),
        None => "na".to_string(),
    }
}

#[cfg(unix)]
fn same_file_metadata(a: &fs::Metadata, b: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.dev() == b.dev() && a.ino() == b.ino()
}

fn same_output_destination(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    #[cfg(unix)]
    if let (Ok(ma), Ok(mb)) = (fs::metadata(a), fs::metadata(b))
        && same_file_metadata(&ma, &mb)
    {
        return true;
    }
    if let (Ok(ca), Ok(cb)) = (fs::canonicalize(a), fs::canonicalize(b)) {
        return ca == cb;
    }
    let resolved_parent_and_name = |p: &Path| -> Option<(PathBuf, std::ffi::OsString)> {
        let name = p.file_name()?.to_os_string();
        let raw_parent = p
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let parent = fs::canonicalize(raw_parent).unwrap_or_else(|_| raw_parent.to_path_buf());
        Some((parent, name))
    };
    matches!(
        (resolved_parent_and_name(a), resolved_parent_and_name(b)),
        (Some(pa), Some(pb)) if pa == pb
    )
}

fn main() {
    let a = parse();
    if let (Some(ts), Some(prof)) = (&a.timeseries_out, &a.profile_out)
        && same_output_destination(ts, prof)
    {
        fail("--timeseries-out and --profile-out must refer to different paths");
    }
    let model = FeCrRis::new(FeCrRisParams {
        cells: a.cells,
        sink_planes: vec![a.sink_plane],
        temperature_k: a.temperature_k,
        k0_dpa_per_s: a.k0,
        capacity: a.capacity,
        dumbbell: DumbbellParams::senninger2016_r1(),
    })
    .unwrap_or_else(|e| fail(e));

    let site_count = model.lattice().site_count();
    let planes = 2 * model.lattice().cells()[0];
    let cr_count = match a.cr_spec {
        CrSpec::Frac(f) => (f * site_count as f64).round() as usize,
        CrSpec::Count(n) => n,
    };
    let bulk_min_planes = a
        .bulk_min_planes
        .unwrap_or_else(|| default_bulk_min_distance_planes(a.cells));

    let config = RisSegregationConfig::uniform(
        cr_count,
        a.dpa_end,
        a.samples,
        a.steady_start_dpa,
        bulk_min_planes,
        a.runs,
        a.seed,
    )
    .unwrap_or_else(|e| fail(e))
    .with_lattice_a_m(a.lattice_a)
    .with_max_steps(a.max_steps);

    let start = Instant::now();
    let replicas = run_ris_segregation(&model, &config).unwrap_or_else(|e| fail(e));
    let summary = summarize_ris_segregation(&replicas).unwrap_or_else(|e| fail(e));
    let wall_s = start.elapsed().as_secs_f64();
    let total_steps: f64 = replicas.iter().map(|r| r.steps as f64).sum();
    let us_per_step = if total_steps > 0.0 {
        wall_s * 1.0e6 / total_steps
    } else {
        0.0
    };

    let ts_file = a.timeseries_out.as_deref().map(|path| {
        let file = open_output_file(path)
            .unwrap_or_else(|err| fail(format!("opening {}: {err}", path.display())));
        (path, file)
    });
    let prof_file = a.profile_out.as_deref().map(|path| {
        let file = open_output_file(path)
            .unwrap_or_else(|err| fail(format!("opening {}: {err}", path.display())));
        (path, file)
    });
    #[cfg(unix)]
    if let (Some((ts_path, ts_f)), Some((prof_path, prof_f))) = (&ts_file, &prof_file) {
        let ts_meta = ts_f
            .metadata()
            .unwrap_or_else(|err| fail(format!("inspecting {}: {err}", ts_path.display())));
        let prof_meta = prof_f
            .metadata()
            .unwrap_or_else(|err| fail(format!("inspecting {}: {err}", prof_path.display())));
        if same_file_metadata(&ts_meta, &prof_meta) {
            fail("--timeseries-out and --profile-out must refer to different paths");
        }
    }
    if let Some((path, file)) = ts_file
        && let Err(err) = write_timeseries_csv(file, &summary)
    {
        fail(format!("writing {}: {err}", path.display()));
    }
    if let Some((path, file)) = prof_file
        && let Err(err) = write_profile_csv(file, &summary)
    {
        fail(format!("writing {}: {err}", path.display()));
    }

    let nominal_cr_frac = cr_count as f64 / site_count as f64;
    let gb_vals: Vec<f64> = summary.c_cr_gb.iter().map(|&(m, _)| m).collect();
    let s_vals: Vec<f64> = summary.s_cr_tot_nm.iter().map(|&(m, _)| m).collect();
    let fit_gb = fit_exponential_relaxation(&summary.doses, &gb_vals, nominal_cr_frac);
    let fit_s = fit_exponential_relaxation(&summary.doses, &s_vals, 0.0);
    let Some(last) = summary.doses.len().checked_sub(1) else {
        fail("empty summary");
    };

    println!("temperature_k={}", a.temperature_k);
    println!("k0_dpa_per_s={:.12e}", a.k0);
    println!("cells={},{},{}", a.cells[0], a.cells[1], a.cells[2]);
    println!("site_count={site_count}");
    println!("planes={planes}");
    println!("sink_plane={}", a.sink_plane);
    println!("bulk_min_planes={bulk_min_planes}");
    println!("cr_count={cr_count}");
    println!("nominal_cr_frac={nominal_cr_frac:.12e}");
    println!("capacity={}", a.capacity);
    println!("runs={}", a.runs);
    println!("samples={}", a.samples);
    println!("dpa_end={:.12e}", a.dpa_end);
    println!("steady_start_dpa={:.12e}", a.steady_start_dpa);
    println!("seed={}", a.seed);
    println!("c_cr_gb_final={:.12e}", summary.c_cr_gb[last].0);
    println!("c_cr_gb_final_se={:.12e}", summary.c_cr_gb[last].1);
    println!("c_cr_bulk_final={:.12e}", summary.c_cr_bulk[last].0);
    println!("c_cr_bulk_final_se={:.12e}", summary.c_cr_bulk[last].1);
    println!("s_cr_tot_nm_final={:.12e}", summary.s_cr_tot_nm[last].0);
    println!("s_cr_tot_nm_final_se={:.12e}", summary.s_cr_tot_nm[last].1);
    println!(
        "s_cr_linear_nm_final={:.12e}",
        summary.s_cr_linear_nm[last].0
    );
    println!(
        "s_cr_linear_nm_final_se={:.12e}",
        summary.s_cr_linear_nm[last].1
    );
    println!("v_mean_total={:.12e}", summary.v_mean_total.0);
    println!("v_mean_total_se={:.12e}", summary.v_mean_total.1);
    println!("i_mean_total={:.12e}", summary.i_mean_total.0);
    println!("i_mean_total_se={:.12e}", summary.i_mean_total.1);
    println!("v_bulk_mean={:.12e}", summary.v_bulk_mean.0);
    println!("v_bulk_mean_se={:.12e}", summary.v_bulk_mean.1);
    println!("i_bulk_mean={:.12e}", summary.i_bulk_mean.0);
    println!("i_bulk_mean_se={:.12e}", summary.i_bulk_mean.1);
    println!("phi_rel_c_cr_gb={}", fmt_opt(fit_gb.map(|f| f.phi_rel)));
    println!("c_cr_gb_inf={}", fmt_opt(fit_gb.map(|f| f.y_inf)));
    println!("phi_rel_s_cr_tot={}", fmt_opt(fit_s.map(|f| f.phi_rel)));
    println!("s_cr_tot_inf_nm={}", fmt_opt(fit_s.map(|f| f.y_inf)));
    println!("mean_steps={:.6}", summary.mean_steps);
    println!("mean_created={:.6}", summary.mean_created);
    println!("mean_recombined={:.6}", summary.mean_recombined);
    println!("mean_absorbed_v={:.6}", summary.mean_absorbed_v);
    println!("mean_absorbed_i={:.6}", summary.mean_absorbed_i);
    println!("mean_borrowed_fe={:.6}", summary.mean_borrowed_fe);
    println!(
        "mean_floored_dumbbell_jumps={:.6}",
        summary.mean_floored_dumbbell_jumps
    );
    println!(
        "mean_floored_vacancy_jumps={:.6}",
        summary.mean_floored_vacancy_jumps
    );
    println!("wall_s={wall_s:.4}");
    println!("us_per_step={us_per_step:.2}");
}
