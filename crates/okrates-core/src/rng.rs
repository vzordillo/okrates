//! Seeded random numbers for event and waiting-time sampling.

use rand_chacha::ChaCha12Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};

/// A reproducible random stream for kinetic Monte Carlo sampling.
#[derive(Clone, Debug)]
pub struct KmcRng {
    inner: ChaCha12Rng,
}

impl KmcRng {
    /// Creates a random stream from a 64-bit seed.
    pub fn from_seed(seed: u64) -> Self {
        let mut bytes = [0_u8; 32];
        bytes[..8].copy_from_slice(&seed.to_le_bytes());
        Self {
            inner: ChaCha12Rng::from_seed(bytes),
        }
    }

    /// Samples a floating-point value strictly between zero and one.
    pub fn open01(&mut self) -> f64 {
        open01_from_bits(self.inner.next_u64())
    }

    /// Exports the seed, stream and word position that determine all later output.
    pub(crate) fn export_state(&self) -> ([u8; 32], u64, u128) {
        (
            self.inner.get_seed(),
            self.inner.get_stream(),
            self.inner.get_word_pos(),
        )
    }

    /// Rebuilds a stream from the output of [`KmcRng::export_state`].
    pub(crate) fn from_state(seed: [u8; 32], stream: u64, word_pos: u128) -> Self {
        let mut inner = ChaCha12Rng::from_seed(seed);
        inner.set_stream(stream);
        inner.set_word_pos(word_pos);
        Self { inner }
    }
}

fn open01_from_bits(bits: u64) -> f64 {
    // The result lies strictly inside (0,1); both extremes are exactly
    // representable, while a 53-bit variant can round to 1.0.
    ((bits >> 12) as f64 + 0.5) * f64::EPSILON
}

#[cfg(test)]
mod tests {
    use super::{KmcRng, open01_from_bits};

    #[test]
    fn maps_bit_extremes_inside_the_open_unit_interval() {
        let low = open01_from_bits(0);
        assert_eq!(low, 0.5 * f64::EPSILON);
        assert!(low > 0.0);

        let high = open01_from_bits(u64::MAX);
        assert_eq!(high, 1.0 - 0.5 * f64::EPSILON);
        assert!(high < 1.0);
    }

    #[test]
    fn golden_stream_seed_42() {
        // Regression snapshot of the ChaCha12 stream under rand_chacha 0.10.0,
        // not a correctness oracle.
        let mut rng = KmcRng::from_seed(42);
        let bits = [
            rng.open01().to_bits(),
            rng.open01().to_bits(),
            rng.open01().to_bits(),
        ];
        assert_eq!(
            bits,
            [0x3fdba660942f2b1e, 0x3fc49e34dfc8e074, 0x3fe5824d8356d36d]
        );
    }

    #[test]
    fn state_round_trip_at_many_positions() {
        for n in 0..=70 {
            let mut original = KmcRng::from_seed(42);
            original.inner.set_stream(5);
            for _ in 0..n {
                original.open01();
            }
            let (seed, stream, word_pos) = original.export_state();
            assert_eq!(stream, 5);
            let mut rebuilt = KmcRng::from_state(seed, stream, word_pos);
            for _ in 0..40 {
                assert_eq!(original.open01().to_bits(), rebuilt.open01().to_bits());
            }
        }
    }
}
