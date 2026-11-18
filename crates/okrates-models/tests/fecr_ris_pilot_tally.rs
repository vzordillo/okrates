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
    // One count of 4 held for 0.5, one of 0 held for 1.5: 4·0.5 / 2.0 = 1.0 (not 2.0).
    let t = run(&[(4, 0, 0.5), (0, 4, 1.5)]);
    assert_eq!(t.mean(), 1.0);
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
