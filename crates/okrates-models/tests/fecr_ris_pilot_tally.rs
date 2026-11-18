//! Tests of the residence-time tally of the `fecr_ris_pilot` example: the count before
//! each event is weighted by the sampled waiting time `dt`, and the mean is normalised by
//! the summed `dt`.

#[path = "../examples/support/tally.rs"]
mod tally;

use tally::Tally;

/// Feed `(before, after, dt)` events to a fresh tally.
fn run(events: &[(u64, u64, f64)]) -> Tally {
    let mut t = Tally::default();
    for &(before, after, dt) in events {
        t.update(before, after, dt);
    }
    t
}

#[test]
fn mean_weights_the_count_before_each_event() {
    // counts before: 1, 2, 0; after: 2, 0, 5. (1·0.5 + 2·0.25 + 0·0.25) / 1.0 = 1.0.
    let t = run(&[(1, 2, 0.5), (2, 0, 0.25), (0, 5, 0.25)]);
    assert_eq!(t.mean(), 1.0);
    assert_eq!(t.peak, 5);
}

#[test]
fn mean_is_normalised_by_elapsed_not_by_step_count() {
    // One count of 4 held for 0.5, one of 0 held for 3.5: 4·0.5 / 4.0 = 0.5. Dividing by
    // the step count (2) would give 1.0.
    let t = run(&[(4, 0, 0.5), (0, 4, 3.5)]);
    assert_eq!(t.mean(), 0.5);
}

#[test]
fn zero_elapsed_gives_zero() {
    assert_eq!(Tally::default().mean(), 0.0);
    assert_eq!(run(&[(3, 3, 0.0)]).mean(), 0.0);
}

#[test]
fn tiny_dt_at_a_large_clock_is_still_counted() {
    // 1e-11 s is below the spacing of f64 at 2^20 s (about 2.3e-10 s), so a clock
    // difference would read 0; the sampled dt must be summed instead.
    let clock = (1u64 << 20) as f64;
    assert_eq!(clock + 1e-11, clock);
    let events = [(2, 2, 1e-11), (4, 4, 1e-11)];
    let t = run(&events);
    let expected = (2.0 * 1e-11 + 4.0 * 1e-11) / (2.0 * 1e-11);
    assert!((t.mean() - 3.0).abs() < 1e-12, "{} vs {expected}", t.mean());
}

#[test]
fn short_waits_after_a_long_interval_are_kept() {
    // After a wait of 2^20, each 1e-11 is below half the spacing of f64 there, so plain
    // summation drops all of them and the mean reads exactly 1.0.
    let long = (1u64 << 20) as f64;
    let mut t = Tally::default();
    t.update(1, 1, long);
    for _ in 0..1_000_000 {
        t.update(3, 3, 1e-11);
    }
    // Exact mean = (2^20 + 3e-5) / (2^20 + 1e-5), so mean - 1 = 2e-5 / (2^20 + 1e-5).
    let expected = 2e-5 / (long + 1e-5);
    let got = t.mean() - 1.0;
    // f64 spacing near 1 is about 2.2e-16, i.e. about 6e-6 relative to `expected` (1.9e-11),
    // so 1e-4 is the tightest sensible tolerance; plain summation is 100 % off.
    assert!(
        ((got - expected) / expected).abs() < 1e-4,
        "{got:e} vs {expected:e}"
    );
}
