//! Simulation time and checked clock advancement.

use std::error::Error;
use std::fmt;

/// The elapsed simulation time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Clock {
    time: f64,
}

impl Clock {
    /// Creates a clock at time zero.
    pub fn new() -> Self {
        Self { time: 0.0 }
    }

    /// Returns the current time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Advances by a finite, strictly positive increment.
    pub fn advance(&mut self, dt: f64) -> Result<(), ClockError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ClockError::InvalidIncrement(dt));
        }
        let time = self.time + dt;
        if !time.is_finite() {
            return Err(ClockError::Overflow);
        }
        self.time = time;
        Ok(())
    }
}

/// An invalid clock increment or an overflowing clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClockError {
    /// The increment was not finite and strictly positive.
    InvalidIncrement(f64),
    /// Adding the increment would make the clock non-finite.
    Overflow,
}

impl fmt::Display for ClockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIncrement(dt) => write!(f, "invalid clock increment: {dt}"),
            Self::Overflow => write!(f, "clock time overflowed"),
        }
    }
}

impl Error for ClockError {}

#[cfg(test)]
mod tests {
    use super::{Clock, ClockError};

    #[test]
    fn rejects_invalid_increments_without_changing_time() {
        let mut clock = Clock::new();
        for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            match clock.advance(dt) {
                Err(ClockError::InvalidIncrement(actual)) if dt.is_nan() => {
                    assert!(actual.is_nan());
                }
                Err(ClockError::InvalidIncrement(actual)) => assert_eq!(actual, dt),
                result => panic!("expected invalid increment for {dt}, got {result:?}"),
            }
            assert_eq!(clock.time(), 0.0);
        }
    }

    #[test]
    fn advances_and_preserves_time_on_overflow() {
        let mut clock = Clock::new();
        clock.advance(1.5).unwrap();
        clock.advance(0.25).unwrap();
        assert_eq!(clock.time(), 1.75);

        let mut large = Clock::new();
        large.advance(f64::MAX).unwrap();
        assert_eq!(large.advance(f64::MAX), Err(ClockError::Overflow));
        assert_eq!(large.time(), f64::MAX);
    }
}
