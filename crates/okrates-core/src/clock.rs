//! Simulation clock using exact TwoSum compensated summation.

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
        let comp = if comp == 0.0 { 0.0 } else { comp };
        if (time + comp).to_bits() != time.to_bits() {
            return Err(ClockError::InvalidState { time, comp });
        }
        Ok(Self { time, comp })
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
}
