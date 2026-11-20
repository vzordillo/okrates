//! Well-mixed Michaelis-Menten enzyme kinetics (`E + S <-> ES -> E + P`) driven by [`ReactionNetwork`].

use std::env;
use std::process::exit;

use okrates_core::{
    AbsorptionPolicy, IncrementalSimulation, SampleGrid, StreamRole, run_ensemble,
    sample_trajectory,
};
use okrates_models::reaction_network::{ReactionChannel, ReactionNetwork};

const USAGE: &str = "usage: reaction_network --e0 N --s0 N --k-bind F --k-unbind F --k-cat F --t-end F --runs R --seed S";

fn usage() -> ! {
    eprintln!("{USAGE}");
    exit(2);
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("error: {message}");
    exit(1);
}

struct Args {
    e0: u64,
    s0: u64,
    k_bind: f64,
    k_unbind: f64,
    k_cat: f64,
    t_end: f64,
    runs: usize,
    seed: u64,
}

fn parse() -> Args {
    let argv: Vec<String> = env::args().skip(1).collect();
    let (mut e0, mut s0, mut k_bind, mut k_unbind, mut k_cat, mut t_end, mut runs, mut seed) =
        (None, None, None, None, None, None, None, None);

    let mut i = 0;
    let next = |i: &mut usize| -> &str {
        *i += 1;
        argv.get(*i).map(String::as_str).unwrap_or_else(|| usage())
    };

    while i < argv.len() {
        match argv[i].as_str() {
            "--e0" => e0 = Some(next(&mut i).parse::<u64>().unwrap_or_else(|_| usage())),
            "--s0" => s0 = Some(next(&mut i).parse::<u64>().unwrap_or_else(|_| usage())),
            "--k-bind" => k_bind = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage())),
            "--k-unbind" => {
                k_unbind = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage()))
            }
            "--k-cat" => k_cat = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage())),
            "--t-end" => t_end = Some(next(&mut i).parse::<f64>().unwrap_or_else(|_| usage())),
            "--runs" => runs = Some(next(&mut i).parse::<usize>().unwrap_or_else(|_| usage())),
            "--seed" => seed = Some(next(&mut i).parse::<u64>().unwrap_or_else(|_| usage())),
            _ => usage(),
        }
        i += 1;
    }

    let args = Args {
        e0: e0.unwrap_or_else(|| usage()),
        s0: s0.unwrap_or_else(|| usage()),
        k_bind: k_bind.unwrap_or_else(|| usage()),
        k_unbind: k_unbind.unwrap_or_else(|| usage()),
        k_cat: k_cat.unwrap_or_else(|| usage()),
        t_end: t_end.unwrap_or_else(|| usage()),
        runs: runs.unwrap_or_else(|| usage()),
        seed: seed.unwrap_or_else(|| usage()),
    };
    if args.runs == 0 || !args.t_end.is_finite() || args.t_end <= 0.0 {
        usage();
    }
    args
}

struct ReplicaOutcome {
    final_counts: Vec<u64>,
    fired_counts: Vec<u64>,
    steps: u64,
}

fn main() {
    let args = parse();
    let channels = vec![
        ReactionChannel::new(vec![(0, 1), (1, 1)], vec![(2, 1)], args.k_bind).with_reverse(1),
        ReactionChannel::new(vec![(2, 1)], vec![(0, 1), (1, 1)], args.k_unbind).with_reverse(0),
        ReactionChannel::new(vec![(2, 1)], vec![(0, 1), (3, 1)], args.k_cat),
    ];
    let model = ReactionNetwork::new(4, channels)
        .and_then(|m| m.with_conservation_laws(vec![vec![1, 0, 1, 0], vec![0, 1, 1, 1]]))
        .unwrap_or_else(|err| fail(err));
    let init = model
        .initial_state(vec![args.e0, args.s0, 0, 0])
        .unwrap_or_else(|err| fail(err));
    let grid = SampleGrid::from_origin(args.t_end, 1).unwrap_or_else(|err| fail(err));

    let outcomes = run_ensemble(args.seed, args.runs as u64, |ctx| {
        let mut sim = IncrementalSimulation::with_rng(
            model.clone(),
            init.clone(),
            ctx.rng(StreamRole::Dynamics),
        )
        .map_err(|err| err.to_string())?;
        let traj = sample_trajectory(
            &mut sim,
            &grid,
            None,
            AbsorptionPolicy::HoldFinal,
            |m, s| m.check_conservation(&init, s).map(|()| s.clone()),
        )
        .map_err(|err| err.to_string())?;
        let final_state = traj
            .values
            .into_iter()
            .next()
            .ok_or_else(|| "missing sampled endpoint".to_string())?
            .map_err(|err| err.to_string())?;
        Ok::<ReplicaOutcome, String>(ReplicaOutcome {
            final_counts: final_state.counts,
            fired_counts: final_state.fired_counts,
            steps: traj.fired_steps,
        })
    });

    let mut sum_counts = [0_u128; 4];
    let mut sum_fired = [0_u128; 3];
    let mut sum_steps = 0_u128;
    for outcome in outcomes {
        let outcome = outcome.unwrap_or_else(|err| fail(err));
        for (acc, &c) in sum_counts.iter_mut().zip(&outcome.final_counts) {
            *acc += u128::from(c);
        }
        for (acc, &f) in sum_fired.iter_mut().zip(&outcome.fired_counts) {
            *acc += u128::from(f);
        }
        sum_steps += u128::from(outcome.steps);
    }

    let n = args.runs as f64;
    println!("species = [E, S, ES, P]");
    println!(
        "initial = [{}, {}, 0, 0] k_bind = {} k_unbind = {} k_cat = {}",
        args.e0, args.s0, args.k_bind, args.k_unbind, args.k_cat
    );
    println!(
        "t_end = {} runs = {} seed = {}",
        args.t_end, args.runs, args.seed
    );
    println!(
        "mean_counts = [{:.6}, {:.6}, {:.6}, {:.6}]",
        (sum_counts[0] as f64) / n,
        (sum_counts[1] as f64) / n,
        (sum_counts[2] as f64) / n,
        (sum_counts[3] as f64) / n,
    );
    println!(
        "mean_firings = [{:.6}, {:.6}, {:.6}] mean_steps = {:.6}",
        (sum_fired[0] as f64) / n,
        (sum_fired[1] as f64) / n,
        (sum_fired[2] as f64) / n,
        (sum_steps as f64) / n,
    );
    println!("conservation = verified");
}
