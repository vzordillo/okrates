//! Simulation clock using exact TwoSum compensated summation.

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

/// Elapsed simulation time `(time, comp)` maintained as a normalized TwoSum pair.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Clock {
    time: f64,
    comp: f64,
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let a_prime = s - b;
    let b_prime = s - a_prime;
    (s, (a - a_prime) + (b - b_prime))
}

impl Clock {
    pub fn new() -> Self {
        Self {
            time: 0.0,
            comp: 0.0,
        }
    }

    /// Restores a clock from checkpointed `(time, comp)` values.
    pub fn from_parts(time: f64, comp: f64) -> Result<Self, ClockError> {
        if !time.is_finite() || time.is_sign_negative() || !comp.is_finite() {
            return Err(ClockError::InvalidState { time, comp });
        }
        let norm_comp = if comp == 0.0 { 0.0 } else { comp };
        if (time == 0.0 && norm_comp != 0.0) || time + norm_comp < 0.0 {
            return Err(ClockError::InvalidState { time, comp });
        }
        let (s, e) = two_sum(time, norm_comp);
        let norm_e = if e == 0.0 { 0.0 } else { e };
        if s.to_bits() != time.to_bits() || norm_e.to_bits() != norm_comp.to_bits() {
            return Err(ClockError::InvalidState { time, comp });
        }
        Ok(Self {
            time,
            comp: norm_comp,
        })
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn compensation(&self) -> f64 {
        self.comp
    }

    /// Advances by a finite `dt > 0`, failing atomically on overflow or underflow stagnation.
    pub fn advance(&mut self, dt: f64) -> Result<(), ClockError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ClockError::InvalidIncrement(dt));
        }
        let (s, e) = two_sum(self.time, dt);
        if !s.is_finite() {
            return Err(ClockError::Overflow);
        }
        let c = self.comp + e;
        if !c.is_finite() {
            return Err(ClockError::Overflow);
        }
        let (next_time, next_comp) = two_sum(s, c);
        if !next_time.is_finite() || !next_comp.is_finite() {
            return Err(ClockError::Overflow);
        }
        if next_time == self.time && next_comp == self.comp {
            return Err(ClockError::Stagnation {
                time: self.time,
                dt,
            });
        }
        self.time = next_time;
        self.comp = if next_comp == 0.0 { 0.0 } else { next_comp };
        Ok(())
    }

    /// Orders `time + comp` against `target` without collapsing `comp`.
    pub fn cmp_f64(&self, target: f64) -> Ordering {
        if !target.is_finite() {
            return self.time.total_cmp(&target);
        }
        if target < 0.0 {
            return Ordering::Greater;
        }
        let (diff, err) = two_sum(self.time, -target);
        let (hi, lo) = two_sum(diff, err + self.comp);
        if hi > 0.0 || (hi == 0.0 && lo > 0.0) {
            Ordering::Greater
        } else if hi < 0.0 || (hi == 0.0 && lo < 0.0) {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClockError {
    InvalidIncrement(f64),
    Overflow,
    /// `dt` underflowed both `time` and `comp` (~106-bit significand precision).
    Stagnation {
        time: f64,
        dt: f64,
    },
    InvalidState {
        time: f64,
        comp: f64,
    },
}

impl fmt::Display for ClockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIncrement(dt) => write!(f, "invalid clock increment: {dt}"),
            Self::Overflow => write!(f, "clock time overflowed"),
            Self::Stagnation { time, dt } => {
                write!(f, "clock stagnated at t={time:e} for dt={dt:e}")
            }
            Self::InvalidState { time, comp } => {
                write!(f, "invalid clock state (time={time}, comp={comp})")
            }
        }
    }
}

impl Error for ClockError {}

#[cfg(test)]
mod tests {
    use super::{Clock, ClockError};

    #[test]
    fn rejects_invalid_increments_without_mutation() {
        let mut clock = Clock::new();
        for dt in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY] {
            match clock.advance(dt) {
                Err(ClockError::InvalidIncrement(actual)) if dt.is_nan() => {
                    assert!(actual.is_nan());
                }
                Err(ClockError::InvalidIncrement(actual)) => assert_eq!(actual, dt),
                result => panic!("expected invalid increment for {dt}, got {result:?}"),
            }
            assert_eq!(clock.time(), 0.0);
            assert_eq!(clock.compensation(), 0.0);
        }
    }

    #[test]
    fn preserves_state_on_overflow() {
        let mut large = Clock::new();
        large.advance(f64::MAX).unwrap();
        let snapshot = large;
        assert_eq!(large.advance(f64::MAX), Err(ClockError::Overflow));
        assert_eq!(large, snapshot);
    }

    #[test]
    fn compensated_summation_accumulates_sub_ulp_increments() {
        // At t = 10_000.0, ulp(t) ≈ 1.82e-12, so naive `10_000.0 + 1e-15 == 10_000.0`.
        let mut clock = Clock::new();
        clock.advance(10_000.0).unwrap();
        for _ in 0..10_000 {
            clock.advance(1e-15).unwrap();
        }
        let elapsed = clock.time() - 10_000.0 + clock.compensation();
        assert!((elapsed - 1e-11).abs() <= 1e-24, "elapsed={elapsed:e}");
        assert!(clock.time() > 10_000.0);
    }

    #[test]
    fn detects_extreme_underflow_stagnation() {
        let mut clock = Clock::from_parts(1e200, 1e180).unwrap();
        let before = clock;
        assert_eq!(
            clock.advance(1e-200),
            Err(ClockError::Stagnation {
                time: 1e200,
                dt: 1e-200
            })
        );
        assert_eq!(clock, before);
    }

    #[test]
    fn from_parts_enforces_canonical_twosum_and_nonnegative_time() {
        let zero = Clock::from_parts(0.0, -0.0).unwrap();
        assert_eq!(zero.compensation().to_bits(), 0.0_f64.to_bits());

        for (time, comp) in [
            (-0.0, 0.0),
            (0.0, 1e-300),
            (0.0, -1e-300),
            (1e-320, -1e-300),
            (10.0, 1.0),
            (f64::MAX, f64::MAX),
            (1.0, f64::NAN),
        ] {
            assert!(
                matches!(
                    Clock::from_parts(time, comp),
                    Err(ClockError::InvalidState { .. })
                ),
                "accepted invalid clock parts ({time:e}, {comp:e})"
            );
        }
    }

    #[test]
    fn cmp_f64_orders_sub_ulp_compensations_and_binade_midpoints() {
        use std::cmp::Ordering;

        let prev = 1.0 - 2.0_f64.powi(-53);
        let below = Clock::from_parts(1.0, -2.0_f64.powi(-54)).unwrap();
        let exact = Clock::from_parts(1.0, 0.0).unwrap();
        let above = Clock::from_parts(1.0, 2.0_f64.powi(-54)).unwrap();

        assert_eq!(below.cmp_f64(prev), Ordering::Greater);
        assert_eq!(below.cmp_f64(1.0), Ordering::Less);
        assert_eq!(exact.cmp_f64(1.0), Ordering::Equal);
        assert_eq!(above.cmp_f64(1.0), Ordering::Greater);
        assert_eq!(
            Clock::from_parts(prev, 0.0).unwrap().cmp_f64(prev),
            Ordering::Equal
        );
        assert_eq!(Clock::new().cmp_f64(-0.0), Ordering::Equal);
        assert_eq!(Clock::new().cmp_f64(-1.0), Ordering::Greater);
        assert_eq!(
            Clock::from_parts(f64::MAX, 0.0).unwrap().cmp_f64(-f64::MAX),
            Ordering::Greater
        );
        assert_eq!(exact.cmp_f64(f64::INFINITY), Ordering::Less);
        assert_eq!(exact.cmp_f64(f64::NEG_INFINITY), Ordering::Greater);
    }
}
