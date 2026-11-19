//! End-to-end tests: run the okrates-models example binaries and check exit
//! codes, output format and seeded values.
//!
//! Expected outputs were produced by running the examples; never edit them by hand.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

const PACKAGE: &str = "okrates-models";

/// Builds this package's examples once per test binary, so a filtered
/// `cargo test --test e2e` never runs a stale example binary, and returns a
/// command for the named example. Paths come from cargo's JSON messages.
fn example(name: &str) -> Command {
    static PATHS: OnceLock<HashMap<String, PathBuf>> = OnceLock::new();
    let paths = PATHS.get_or_init(|| {
        let output = Command::new(env!("CARGO"))
            .args(["build", "--locked", "-q", "--examples", "-p", PACKAGE])
            .arg("--message-format=json")
            .output()
            .expect("cargo runs");
        assert!(output.status.success(), "building the examples failed");
        let field = |line: &str, key: &str| -> Option<String> {
            let start = line.find(key)? + key.len();
            let end = line[start..].find('"')?;
            Some(line[start..start + end].to_string())
        };
        String::from_utf8(output.stdout)
            .expect("utf-8 cargo output")
            .lines()
            .filter(|line| line.contains(r#""kind":["example"]"#))
            .filter_map(|line| {
                let name = field(line, r#""name":""#)?;
                let exe = field(line, r#""executable":""#)?;
                Some((name, PathBuf::from(exe)))
            })
            .collect()
    });
    let mut command = Command::new(&paths[name]);
    command.env_remove("RAYON_NUM_THREADS");
    command
}

/// Runs a command and returns (exit code, stdout, stderr).
fn run(command: &mut Command) -> (i32, String, String) {
    let output = command.output().expect("example runs");
    (
        output.status.code().expect("exit code"),
        String::from_utf8(output.stdout).expect("utf-8 stdout"),
        String::from_utf8(output.stderr).expect("utf-8 stderr"),
    )
}

/// Asserts the usage-error contract: exit 2, empty stdout, message on stderr.
fn assert_usage_error(command: &mut Command, message: &str) {
    let (code, stdout, stderr) = run(command);
    assert_eq!(code, 2, "stderr: {stderr}");
    assert_eq!(stdout, "");
    assert_eq!(stderr, format!("{message}\n"));
}

#[test]
fn vacancy_trajectory_output() {
    let (code, stdout, _) = run(example("vacancy_trajectory").args(["7", "3"]));
    assert_eq!(code, 0);
    assert_eq!(
        stdout,
        "rate 803859640.3012047\n\
         0.0 0.0\n\
         1.2903598491084096e-11 6.162616687499999e-20\n\
         1.1060356465215406e-10 8.216822249999999e-20\n\
         1.66623772633431e-10 6.162616687499999e-20\n"
    );
}

#[test]
fn vacancy_trajectory_bad_arguments() {
    let usage = "usage: vacancy_trajectory SEED STEPS";
    assert_usage_error(&mut example("vacancy_trajectory"), usage);
    assert_usage_error(example("vacancy_trajectory").arg("7"), usage);
    assert_usage_error(example("vacancy_trajectory").args(["7", "x"]), usage);
    assert_usage_error(example("vacancy_trajectory").args(["7", "3", "1"]), usage);
}

#[test]
fn vacancy_msd_small() {
    let (code, stdout, stderr) = run(example("vacancy_msd").args(["4", "7"]));
    assert_eq!(code, 0);
    assert_eq!(
        stdout,
        "parameterization = illustrative parameters, not a validated Fe parameterization\n\
         cells = [4, 4, 4]\n\
         nu = 1e13 s^-1\n\
         Em = 0.65 eV\n\
         T = 800 K\n\
         a = 2.8665e-10 m\n\
         r = 8.038596403012e8 s^-1\n\
         T_end = 1.554997834612e-7 s\n\
         D_theory = 6.605171778304e-11 m^2/s\n\
         d_hat = 4.072503011674e-11 m^2/s\n\
         se = 1.563969594909e-11 m^2/s\n\
         CI_95 = [1.007178932736e-11, 7.137827090612e-11] m^2/s\n\
         z_score_vs_theory = -1.619384913156\n\
         relative_error = 3.834372294372e-1\n\
         runs = 4\n\
         seed_base = 7\n\
         total_fired_steps = 3988\n"
    );
    assert!(stderr.starts_with("threads = "), "stderr: {stderr}");
}

#[test]
fn vacancy_msd_bad_arguments() {
    let usage = "usage: vacancy_msd [RUNS] [SEED_BASE]";
    assert_usage_error(example("vacancy_msd").arg("1"), usage);
    assert_usage_error(example("vacancy_msd").args(["4", "x"]), usage);
    assert_usage_error(example("vacancy_msd").args(["4", "7", "1"]), usage);
    assert_usage_error(
        example("vacancy_msd").args(["4", &u64::MAX.to_string()]),
        usage,
    );
}

/// The minimum-size run: about 6 s in debug.
#[test]
#[ignore = "full tier: e2e"]
fn fecr_diffusion_small() {
    let (code, stdout, stderr) = run(example("fecr_diffusion")
        .env("RUNS_F", "2")
        .env("RUNS", "2"));
    assert_eq!(code, 0, "stderr: {stderr}");
    assert_eq!(stdout, include_str!("data/fecr_diffusion_runs2.txt"));
}

/// Fast-tier smoke test: the binary starts and rejects a bad `RUNS_F` before any work.
#[test]
fn fecr_diffusion_bad_environment() {
    assert_usage_error(
        example("fecr_diffusion").env("RUNS_F", "1"),
        "RUNS_F must be an integer >= 2",
    );
}

/// Fast-tier smoke test: the binary rejects a non-positive `ANNEAL_SWEEPS` before any work.
#[test]
fn fecr_diffusion_bad_anneal_sweeps() {
    assert_usage_error(
        example("fecr_diffusion").env("ANNEAL_SWEEPS", "0"),
        "ANNEAL_SWEEPS must be an integer >= 1",
    );
}

#[test]
fn fecr_diffusion_reports_anneal_allocation_error() {
    let (code, stdout, stderr) = run(example("fecr_diffusion")
        .env("RUNS_F", "2")
        .env("RUNS", "2")
        .env("ANNEAL_SWEEPS", u64::MAX.to_string()));
    assert_eq!(code, 1, "stderr: {stderr}");
    assert_eq!(stdout, "");
    assert!(
        stderr.starts_with("error: cannot allocate anneal trace"),
        "stderr: {stderr}"
    );
}

#[test]
fn fecr_observables_small() {
    let (code, stdout, stderr) = run(example("fecr_observables").args(["22", "7", "2", "20"]));
    assert_eq!(code, 0, "stderr: {stderr}");
    assert_eq!(stdout, include_str!("data/fecr_observables_small.txt"));
}

#[test]
fn fecr_observables_bad_args() {
    assert_usage_error(
        example("fecr_observables").args(["22", "7", "1", "0"]),
        "usage: fecr_observables <cr_count> <seed_base> <runs>=2 <anneal_sweeps>",
    );
}

/// Removes every `us_per_step=<value>` field: it is a wall-clock time and not deterministic.
/// Applied to both the output and the expected file.
fn without_wall_time(text: &str) -> String {
    text.lines()
        .map(|line| {
            let Some(start) = line.find("us_per_step=") else {
                return line.to_string();
            };
            let value = &line[start + "us_per_step=".len()..];
            let end = value
                .find(' ')
                .map_or(line.len(), |n| line.len() - value.len() + n);
            format!("{}{}", &line[..start], &line[end..])
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// About 0.05 s in debug.
#[test]
fn fecr_ris_pilot_small() {
    let (code, stdout, stderr) = run(example("fecr_ris_pilot").args([
        "--cells",
        "6",
        "6",
        "6",
        "--temperature",
        "773",
        "--k0",
        "1e-3",
        "--dpa-end",
        "0.3",
        "--capacity",
        "64",
        "--seed",
        "7",
        "--report-every-steps",
        "200",
    ]));
    assert_eq!(code, 0, "stderr: {stderr}");
    assert!(stdout.contains("us_per_step="));
    assert_eq!(
        without_wall_time(&stdout),
        without_wall_time(include_str!("data/fecr_ris_pilot_small.txt"))
    );
}

#[test]
fn fecr_ris_pilot_bad_args() {
    let usage = "usage: fecr_ris_pilot --temperature K --k0 dpa_per_s --dpa-end x --capacity P --seed s [--cells nx ny nz] [--report-every-steps n]";
    assert_usage_error(
        example("fecr_ris_pilot").args([
            "--temperature",
            "773",
            "--k0",
            "1e-3",
            "--dpa-end",
            "0.3",
            "--capacity",
            "64",
        ]),
        usage,
    );
    for bad_dpa in ["inf", "-inf", "NaN"] {
        assert_usage_error(
            example("fecr_ris_pilot").args([
                "--temperature",
                "773",
                "--k0",
                "1e-3",
                "--dpa-end",
                bad_dpa,
                "--capacity",
                "64",
                "--seed",
                "7",
            ]),
            usage,
        );
    }
}

/// About 3.5 s in debug.
#[test]
#[ignore = "full tier: e2e"]
fn fecr_observables_martinez_small() {
    let (code, stdout, stderr) =
        run(example("fecr_observables_martinez").args(["22", "7", "2", "20"]));
    assert_eq!(code, 0, "stderr: {stderr}");
    assert_eq!(
        stdout,
        include_str!("data/fecr_observables_martinez_small.txt")
    );
}

#[test]
fn fecr_observables_martinez_bad_args() {
    assert_usage_error(
        example("fecr_observables_martinez").args(["22", "7", "1", "0"]),
        "usage: fecr_observables_martinez <cr_count> <seed_base> <runs>=2 <anneal_sweeps>",
    );
}

#[test]
fn multi_vacancy_scaling_small() {
    let (code, stdout, stderr) = run(example("multi_vacancy_scaling")
        .env("MAX_N", "4")
        .env("STEP_BUDGET", "1"));
    assert_eq!(code, 0, "stderr: {stderr}");
    let mut lines = stdout.lines();
    assert_eq!(
        lines.next(),
        Some(
            "N,M,steps,direct_median_ns,direct_min_ns,direct_max_ns,\
             incremental_median_ns,incremental_min_ns,incremental_max_ns,speedup"
        )
    );
    let rows: Vec<Vec<&str>> = lines.map(|line| line.split(',').collect()).collect();
    assert_eq!(rows.len(), 2);
    for (row, (n, m)) in rows.iter().zip([("1", "8"), ("4", "32")]) {
        assert_eq!(row[..3], [n, m, "2000"]);
        let ns: Vec<f64> = row[3..].iter().map(|v| v.parse().unwrap()).collect();
        assert_eq!(ns.len(), 7);
        assert!(ns[1] <= ns[0] && ns[0] <= ns[2], "{row:?}");
        assert!(ns[4] <= ns[3] && ns[3] <= ns[5], "{row:?}");
    }
}

#[test]
fn multi_vacancy_scaling_bad_environment() {
    assert_usage_error(
        example("multi_vacancy_scaling").env("MAX_N", "-1"),
        "MAX_N must be an integer >= 1",
    );
    assert_usage_error(
        example("multi_vacancy_scaling").env("STEP_BUDGET", "0"),
        "STEP_BUDGET must be an integer >= 1",
    );
}

fn without_wall_s(text: &str) -> String {
    text.lines()
        .filter(|line| !line.starts_with("wall_s="))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn fecr_ris_transport_small() {
    let (code, stdout, stderr) = run(example("fecr_ris_transport").args([
        "--defect",
        "dumbbell",
        "--temperature",
        "650",
        "--cr-count",
        "43",
        "--runs",
        "4",
        "--t-end",
        "5e-11",
        "--samples",
        "8",
        "--seed",
        "7",
        "--cells",
        "6",
        "6",
        "6",
    ]));
    assert_eq!(code, 0, "stderr: {stderr}");
    assert!(stdout.contains("wall_s="));
    assert_eq!(
        without_wall_s(&stdout),
        without_wall_s(include_str!("data/fecr_ris_transport_small.txt"))
    );
}

#[test]
fn fecr_ris_transport_bad_args() {
    let usage = "usage: fecr_ris_transport --defect <vacancy|dumbbell> --temperature K --cr-count N --runs R --t-end s --samples K --seed S [--t-equil s] [--cells nx ny nz] [--lattice-a m] [--max-steps N] [--fit <ols|origin>]";
    assert_usage_error(&mut example("fecr_ris_transport"), usage);
    assert_usage_error(
        example("fecr_ris_transport").args([
            "--defect",
            "void",
            "--temperature",
            "650",
            "--cr-count",
            "43",
            "--runs",
            "4",
            "--t-end",
            "5e-11",
            "--samples",
            "8",
            "--seed",
            "7",
        ]),
        usage,
    );
    assert_usage_error(
        example("fecr_ris_transport").args([
            "--defect",
            "vacancy",
            "--temperature",
            "650",
            "--cr-count",
            "43",
            "--runs",
            "4",
            "--t-end",
            "5e-11",
            "--samples",
            "8",
            "--seed",
            "7",
            "--max-steps",
            "0",
        ]),
        usage,
    );
}

fn is_segregation_fit_line(line: &str) -> bool {
    line.starts_with("phi_rel_s_cr_tot=") || line.starts_with("s_cr_tot_inf_nm=")
}

fn without_timing_and_fit(text: &str) -> String {
    text.lines()
        .filter(|line| {
            !line.starts_with("wall_s=")
                && !line.starts_with("us_per_step=")
                && !is_segregation_fit_line(line)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn segregation_fit_lines(text: &str) -> Vec<&str> {
    text.lines()
        .filter(|line| is_segregation_fit_line(line))
        .collect()
}

#[test]
fn fecr_ris_segregation_small() {
    let ts_path = std::env::temp_dir().join(format!("okrates_seg_ts_{}.csv", std::process::id()));
    let prof_path =
        std::env::temp_dir().join(format!("okrates_seg_prof_{}.csv", std::process::id()));
    let ts_arg = ts_path.to_str().unwrap().to_string();
    let prof_arg = prof_path.to_str().unwrap().to_string();

    let (code, stdout, stderr) = run(example("fecr_ris_segregation").args([
        "--temperature",
        "773",
        "--k0",
        "1e3",
        "--cr-count",
        "43",
        "--dpa-end",
        "2e-3",
        "--capacity",
        "32",
        "--runs",
        "2",
        "--samples",
        "4",
        "--seed",
        "7",
        "--cells",
        "8",
        "6",
        "6",
        "--timeseries-out",
        &ts_arg,
        "--profile-out",
        &prof_arg,
    ]));
    assert_eq!(code, 0, "stderr: {stderr}");
    assert!(stdout.contains("wall_s="));
    assert!(stdout.contains("us_per_step="));
    let expected = include_str!("data/fecr_ris_segregation_small.txt");
    assert_eq!(
        without_timing_and_fit(&stdout),
        without_timing_and_fit(expected)
    );
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    assert_eq!(
        segregation_fit_lines(&stdout),
        segregation_fit_lines(expected),
        "x86_64-linux relaxation fit output lines differ"
    );
    // aarch64-macos values printed by macOS CI (PR #25, run 44605084477);
    // all 41 physics observable and counter lines match x86_64-linux bit-for-bit,
    // while platform libm exp_m1/ln shifts the 1D golden-section SSE argmin by ~3e-8 relative.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    assert_eq!(
        segregation_fit_lines(&stdout),
        vec![
            "phi_rel_s_cr_tot=1.154588383693e-3",
            "s_cr_tot_inf_nm=1.818526045640e-2",
        ],
        "aarch64-macos relaxation fit output lines differ"
    );
    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "macos")
    )))]
    {
        let got = segregation_fit_lines(&stdout);
        let exp = segregation_fit_lines(expected);
        assert_eq!(got.len(), 2, "expected 2 fit lines in {stdout}");
        assert_eq!(exp.len(), 2);
        for (g_line, e_line) in got.iter().zip(&exp) {
            let (gk, gv) = g_line.split_once('=').unwrap();
            let (ek, ev) = e_line.split_once('=').unwrap();
            assert_eq!(gk, ek);
            let g: f64 = gv.parse().unwrap();
            let e: f64 = ev.parse().unwrap();
            assert!(
                ((g - e) / e).abs() <= 1e-6,
                "{gk} differs beyond 1e-6 relative: got {g}, expected {e}"
            );
        }
    }

    let ts_csv = std::fs::read_to_string(&ts_path).unwrap();
    let ts_lines: Vec<&str> = ts_csv.lines().collect();
    assert_eq!(
        ts_lines.first().copied(),
        Some(
            "dose_dpa,c_cr_gb,c_cr_gb_se,c_cr_bulk,c_cr_bulk_se,s_cr_tot_nm,s_cr_tot_nm_se,s_cr_linear_nm,s_cr_linear_nm_se"
        )
    );
    assert_eq!(ts_lines.len(), 5);

    let prof_csv = std::fs::read_to_string(&prof_path).unwrap();
    let prof_lines: Vec<&str> = prof_csv.lines().collect();
    assert_eq!(
        prof_lines.first().copied(),
        Some("d_planes,distance_nm,c_cr_mean,c_cr_se,c_v_mean,c_v_se,c_i_mean,c_i_se")
    );
    assert_eq!(prof_lines.len(), 17);

    let _ = std::fs::remove_file(&ts_path);
    let _ = std::fs::remove_file(&prof_path);
}

#[test]
fn fecr_ris_segregation_bad_args() {
    let usage = "usage: fecr_ris_segregation --temperature K --k0 dpa_per_s (--cr-frac f | --cr-count N) --dpa-end x --capacity P --runs R --samples K --seed S [--cells nx ny nz] [--sink-plane x] [--bulk-min-planes n] [--steady-start-dpa x] [--lattice-a m] [--max-steps N] [--timeseries-out path] [--profile-out path]";
    assert_usage_error(&mut example("fecr_ris_segregation"), usage);
    assert_usage_error(
        example("fecr_ris_segregation").args([
            "--temperature",
            "773",
            "--k0",
            "1e3",
            "--cr-frac",
            "0.1",
            "--cr-count",
            "43",
            "--dpa-end",
            "2e-3",
            "--capacity",
            "32",
            "--runs",
            "2",
            "--samples",
            "4",
            "--seed",
            "7",
        ]),
        usage,
    );
    assert_usage_error(
        example("fecr_ris_segregation").args([
            "--temperature",
            "773",
            "--k0",
            "1e3",
            "--cr-frac",
            "1.5",
            "--dpa-end",
            "2e-3",
            "--capacity",
            "32",
            "--runs",
            "2",
            "--samples",
            "4",
            "--seed",
            "7",
        ]),
        usage,
    );
    assert_usage_error(
        example("fecr_ris_segregation").args([
            "--temperature",
            "773",
            "--k0",
            "1e3",
            "--cr-count",
            "43",
            "--dpa-end",
            "2e-3",
            "--steady-start-dpa",
            "2e-3",
            "--capacity",
            "32",
            "--runs",
            "2",
            "--samples",
            "4",
            "--seed",
            "7",
        ]),
        usage,
    );
    assert_usage_error(
        example("fecr_ris_segregation").args([
            "--temperature",
            "773",
            "--k0",
            "1e3",
            "--cr-count",
            "43",
            "--dpa-end",
            "2e-3",
            "--capacity",
            "32",
            "--runs",
            "2",
            "--samples",
            "4",
            "--seed",
            "7",
            "--max-steps",
            "0",
        ]),
        usage,
    );

    let same_path =
        std::env::temp_dir().join(format!("okrates_seg_same_{}.csv", std::process::id()));
    let _ = std::fs::remove_file(&same_path);
    let same_arg = same_path.to_str().unwrap().to_string();
    let eq_arg = same_path
        .parent()
        .unwrap()
        .join(".")
        .join(same_path.file_name().unwrap())
        .to_str()
        .unwrap()
        .to_string();
    for second_arg in [&same_arg, &eq_arg] {
        let (code, stdout, stderr) = run(example("fecr_ris_segregation").args([
            "--temperature",
            "773",
            "--k0",
            "1e3",
            "--cr-count",
            "43",
            "--dpa-end",
            "2e-3",
            "--capacity",
            "32",
            "--runs",
            "2",
            "--samples",
            "4",
            "--seed",
            "7",
            "--timeseries-out",
            &same_arg,
            "--profile-out",
            second_arg,
        ]));
        assert_eq!(code, 1, "stderr: {stderr}");
        assert_eq!(stdout, "");
        assert!(
            stderr.contains("must refer to different paths"),
            "stderr: {stderr}"
        );
        assert!(!same_path.exists());
    }
}
