//! Validated non-negative finite event rates.

use std::error::Error;
use std::fmt;

/// Non-negative finite event rate in inverse time units (core is unit-agnostic).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Rate(f64);

impl Rate {
    pub const ZERO: Self = Self(0.0);

    /// Validates `value >= 0` and finite, normalizing `-0.0` to `+0.0`.
    pub fn new(value: f64) -> Result<Self, RateError> {
        if value.is_finite() && value >= 0.0 {
            Ok(Self(if value == 0.0 { 0.0 } else { value }))
        } else if value.is_nan() {
            Err(RateError::NaN)
        } else if value.is_infinite() {
            Err(RateError::Infinite(value))
        } else {
            Err(RateError::Negative(value))
        }
    }

    pub fn get(self) -> f64 {
        self.0
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RateError {
    NaN,
    Infinite(f64),
    Negative(f64),
}

impl fmt::Display for RateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NaN => write!(f, "rate is NaN"),
            Self::Infinite(value) => write!(f, "rate is infinite: {value}"),
            Self::Negative(value) => write!(f, "rate is negative: {value}"),
        }
    }
}

impl Error for RateError {}

#[cfg(test)]
mod tests {
    use super::{Rate, RateError};

    #[test]
    fn accepts_finite_nonnegative_values_and_normalizes_negative_zero() {
        assert_eq!(Rate::new(0.0).unwrap().get(), 0.0);
        assert!(Rate::new(-0.0).unwrap().get().is_sign_positive());
        assert_eq!(Rate::new(5e-324).unwrap().get(), 5e-324);
        assert_eq!(Rate::new(1e-300).unwrap().get(), 1e-300);
        assert_eq!(Rate::new(f64::MAX).unwrap().get(), f64::MAX);
    }

    #[test]
    fn rejects_nan_infinities_and_finite_negative_values() {
        assert_eq!(Rate::new(f64::NAN).unwrap_err(), RateError::NaN);
        assert_eq!(
            Rate::new(f64::INFINITY).unwrap_err(),
            RateError::Infinite(f64::INFINITY)
        );
        assert_eq!(
            Rate::new(f64::NEG_INFINITY).unwrap_err(),
            RateError::Infinite(f64::NEG_INFINITY)
        );
        assert_eq!(
            Rate::new(-1e-300).unwrap_err(),
            RateError::Negative(-1e-300)
        );
    }
}
