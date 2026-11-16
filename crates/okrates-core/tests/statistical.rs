use std::sync::OnceLock;

use okrates_core::reference::GraphRandomWalk;
use okrates_core::{Simulation, Step};

const N_HUB: usize = 200_000;
const N_LEAF: usize = 200_000;

// Each test uses alpha = 1e-6: chi-square df=3 uses its upper 1e-6 quantile,
// normal means use a two-sided alpha=1e-6 cutoff, conditioned means apply
// Bonferroni over four targets, and KS uses the asymptotic Kolmogorov cutoff.
const CHI2_DF3_CRIT: f64 = 30.6648497062136;
const Z_CRIT: f64 = 4.89163847569859;
const Z_CRIT_BONF4: f64 = 5.157701312198881;
const KS_LAMBDA_CRIT: f64 = 2.6933861344527097;

struct Sample {
    hub_departures: Vec<(usize, f64)>,
    leaf_waiting_times: Vec<f64>,
}

fn star() -> GraphRandomWalk {
    GraphRandomWalk::new(vec![
        vec![(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 0.0)],
        vec![(0, 0.5)],
        vec![(0, 0.5)],
        vec![(0, 0.5)],
        vec![(0, 0.5)],
        vec![(0, 0.5)],
    ])
    .unwrap()
}

fn sample() -> &'static Sample {
    static SAMPLE: OnceLock<Sample> = OnceLock::new();
    SAMPLE.get_or_init(|| {
        let mut simulation = Simulation::new(star(), 0, 20261116);
        let mut hub_departures = Vec::with_capacity(N_HUB);
        let mut leaf_waiting_times = Vec::with_capacity(N_LEAF);
        for _ in 0..(N_HUB + N_LEAF) {
            let before = *simulation.state();
            let step = simulation.step().unwrap();
            // Invariants after every event: the event was enumerated from the
            // current state, apply moved the walker to its target, and the
            // state stays inside the graph.
            if let Step::Fired { event, .. } = &step {
                assert_eq!(event.from, before);
                assert_eq!(*simulation.state(), event.to);
            }
            assert!(*simulation.state() < simulation.model().node_count());
            match step {
                Step::Fired { event, dt } if event.from == 0 => {
                    hub_departures.push((event.to, dt));
                }
                Step::Fired { dt, .. } => leaf_waiting_times.push(dt),
                Step::Absorbed => panic!("star graph unexpectedly absorbed"),
            }
        }
        assert_eq!(hub_departures.len(), N_HUB);
        assert_eq!(leaf_waiting_times.len(), N_LEAF);
        Sample {
            hub_departures,
            leaf_waiting_times,
        }
    })
}

fn mean(values: impl Iterator<Item = f64>, n: usize) -> f64 {
    values.sum::<f64>() / n as f64
}

fn ks_exponential(samples: &[f64], rate: f64) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let nf = n as f64;
    let mut d: f64 = 0.0;
    for (index, &x) in sorted.iter().enumerate() {
        let i = (index + 1) as f64;
        let cdf = -(-rate * x).exp_m1();
        d = d.max(i / nf - cdf).max(cdf - (i - 1.0) / nf);
    }
    nf.sqrt() * d
}

#[test]
fn selection_frequencies_match_rates() {
    let sample = sample();
    let weights = [1.0, 2.0, 3.0, 4.0];
    let mut chi2 = 0.0;
    for (index, weight) in weights.into_iter().enumerate() {
        let target = index + 1;
        let observed = sample
            .hub_departures
            .iter()
            .filter(|&&(to, _)| to == target)
            .count() as f64;
        let expected = N_HUB as f64 * weight / 10.0;
        chi2 += (observed - expected).powi(2) / expected;
    }
    assert_eq!(
        sample
            .hub_departures
            .iter()
            .filter(|&&(to, _)| to == 5)
            .count(),
        0
    );
    assert!(chi2 <= CHI2_DF3_CRIT, "chi2={chi2}");
    println!("chi2={chi2}");
}

#[test]
fn hub_waiting_time_mean() {
    let sample = sample();
    let observed_mean = mean(sample.hub_departures.iter().map(|&(_, dt)| dt), N_HUB);
    let z = (observed_mean - 0.1) / (0.1 / (N_HUB as f64).sqrt());
    assert!(z.abs() <= Z_CRIT, "z={z}");
    println!("hub z={z}");
}

#[test]
fn hub_waiting_time_ks() {
    let sample = sample();
    let waiting_times: Vec<f64> = sample.hub_departures.iter().map(|&(_, dt)| dt).collect();
    let ks = ks_exponential(&waiting_times, 10.0);
    assert!(ks <= KS_LAMBDA_CRIT, "ks={ks}");
    println!("hub ks={ks}");
}

#[test]
fn leaf_waiting_time_mean_and_ks() {
    let sample = sample();
    let observed_mean = mean(sample.leaf_waiting_times.iter().copied(), N_LEAF);
    let z = (observed_mean - 2.0) / (2.0 / (N_LEAF as f64).sqrt());
    assert!(z.abs() <= Z_CRIT, "leaf z={z}");
    let ks = ks_exponential(&sample.leaf_waiting_times, 0.5);
    assert!(ks <= KS_LAMBDA_CRIT, "leaf ks={ks}");
    println!("leaf z={z}");
    println!("leaf ks={ks}");
}

#[test]
fn waiting_time_independent_of_selected_event() {
    let sample = sample();
    let mut z_values = Vec::with_capacity(4);
    for target in 1..=4 {
        let values: Vec<f64> = sample
            .hub_departures
            .iter()
            .filter_map(|&(to, dt)| (to == target).then_some(dt))
            .collect();
        let n = values.len();
        let observed_mean = mean(values.iter().copied(), n);
        let z = (observed_mean - 0.1) / (0.1 / (n as f64).sqrt());
        assert!(z.abs() <= Z_CRIT_BONF4, "target={target} z={z}");
        z_values.push(z);
    }
    println!("conditioned z={z_values:?}");
}
