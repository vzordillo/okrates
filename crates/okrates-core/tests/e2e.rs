//! End-to-end tests: run the okrates-core example binaries and check exit
//! codes, output format and seeded values.
//!
//! Expected outputs were produced by running the examples; never edit them by hand.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

const PACKAGE: &str = "okrates-core";

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

/// The README command `cargo run --example random_walk -- 42 5`.
#[test]
fn random_walk_readme_output() {
    let (code, stdout, _) = run(example("random_walk").args(["42", "5"]));
    assert_eq!(code, 0);
    assert_eq!(
        stdout,
        "step 1 t=0.912933 0 -> 1\n\
         step 2 t=1.101330 1 -> 0\n\
         step 3 t=2.071264 0 -> 1\n\
         step 4 t=3.343817 1 -> 2\n\
         step 5 t=4.358610 2 -> 1\n\
         steps=5 time=4.358610\n"
    );
}

#[test]
fn random_walk_defaults_are_seed_42_and_10_steps() {
    let (code, default, _) = run(&mut example("random_walk"));
    let (_, explicit, _) = run(example("random_walk").args(["42", "10"]));
    assert_eq!(code, 0);
    assert_eq!(default, explicit);
    assert_eq!(default.lines().count(), 11);
}

#[test]
fn random_walk_bad_arguments() {
    let usage = "usage: random_walk [SEED] [STEPS]";
    assert_usage_error(example("random_walk").arg("x"), usage);
    assert_usage_error(example("random_walk").args(["42", "-1"]), usage);
    assert_usage_error(example("random_walk").args(["1", "2", "3"]), usage);
}

#[test]
fn catalog_scaling_small() {
    let (code, stdout, stderr) = run(example("catalog_scaling")
        .env("MAX_M", "32")
        .env("STEP_BUDGET", "1"));
    assert_eq!(code, 0, "stderr: {stderr}");
    let mut lines = stdout.lines();
    assert_eq!(
        lines.next(),
        Some(
            "backend,M,rho_update,rate_skew,bytes_per_slot,steps,reps,median_ns_per_step,min_ns_per_step,max_ns_per_step"
        )
    );
    let rows: Vec<Vec<&str>> = lines.map(|line| line.split(',').collect()).collect();
    assert_eq!(rows.len(), 4);
    for (row, (backend, m)) in rows.iter().zip([
        ("direct", "8"),
        ("sum_tree", "8"),
        ("direct", "32"),
        ("sum_tree", "32"),
    ]) {
        assert_eq!(row[0], backend);
        assert_eq!(row[1], m);
        assert_eq!(row[5..7], ["1000", "5"]);
        let ns: Vec<f64> = row[7..].iter().map(|v| v.parse().unwrap()).collect();
        assert_eq!(ns.len(), 3);
        assert!(ns[1] <= ns[0] && ns[0] <= ns[2], "{row:?}");
    }
}

#[test]
fn catalog_scaling_bad_environment() {
    assert_usage_error(
        example("catalog_scaling").env("MAX_M", "0"),
        "MAX_M must be an integer >= 1",
    );
    assert_usage_error(
        example("catalog_scaling").env("STEP_BUDGET", "x"),
        "STEP_BUDGET must be an integer >= 1",
    );
}
