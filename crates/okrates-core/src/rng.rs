//! Seeded ChaCha12 streams with domain-separated replica substreams
//! (Bernstein, "ChaCha, a variant of Salsa20", *Workshop Record of SASC 2008*, pp. 273–278 (2008),
//! <https://cr.yp.to/chacha.html>).

use rand_chacha::ChaCha12Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};

/// Sub-stream domain tag encoded into bytes `16..24` of the 256-bit ChaCha12 key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StreamRole {
    Dynamics,
    InitialState,
    Equilibration,
    Custom(u16),
}

impl StreamRole {
    fn tag(self) -> u64 {
        match self {
            Self::Dynamics => 1,
            Self::InitialState => 2,
            Self::Equilibration => 3,
            Self::Custom(id) => 0x1_0000 | u64::from(id),
        }
    }
}

/// Deterministic `ChaCha12Rng` wrapper emitting samples strictly inside `(0, 1)`.
#[derive(Clone, Debug)]
pub struct KmcRng {
    inner: ChaCha12Rng,
}

impl KmcRng {
    /// Stream `0` with `seed` in key bytes `0..8` and zeros in `8..32`.
    pub fn from_seed(seed: u64) -> Self {
        Self::with_stream(seed, 0)
    }

    /// Selects an independent 64-bit ChaCha12 stream under `seed`.
    pub fn with_stream(seed: u64, stream: u64) -> Self {
        let mut bytes = [0_u8; 32];
        bytes[..8].copy_from_slice(&seed.to_le_bytes());
        let mut inner = ChaCha12Rng::from_seed(bytes);
        inner.set_stream(stream);
        Self { inner }
    }

    /// Collision-free stream for `(base_seed, replica_id, role)` with no `u64` addition overflow.
    pub fn for_replica(base_seed: u64, replica_id: u64, role: StreamRole) -> Self {
        let mut bytes = [0_u8; 32];
        bytes[..8].copy_from_slice(&base_seed.to_le_bytes());
        bytes[8..16].copy_from_slice(&replica_id.to_le_bytes());
        bytes[16..24].copy_from_slice(&role.tag().to_le_bytes());
        Self {
            inner: ChaCha12Rng::from_seed(bytes),
        }
    }

    /// Uniform sample strictly inside `(0, 1)`.
    pub fn open01(&mut self) -> f64 {
        open01_from_bits(self.inner.next_u64())
    }

    /// Uniform index in `0..len`. Panics if `len == 0`.
    pub fn index(&mut self, len: usize) -> usize {
        assert!(len > 0, "cannot sample index from empty range");
        ((self.open01() * len as f64) as usize).min(len - 1)
    }

    pub(crate) fn export_state(&self) -> ([u8; 32], u64, u128) {
        (
            self.inner.get_seed(),
            self.inner.get_stream(),
            self.inner.get_word_pos(),
        )
    }

    pub(crate) fn from_state(seed: [u8; 32], stream: u64, word_pos: u128) -> Self {
        let mut inner = ChaCha12Rng::from_seed(seed);
        inner.set_stream(stream);
        inner.set_word_pos(word_pos);
        Self { inner }
    }
}

fn open01_from_bits(bits: u64) -> f64 {
    // 52-bit mantissa grid shifted by 0.5 ulp stays strictly inside (0, 1),
    // whereas a 53-bit variant rounds `u64::MAX` to 1.0.
    ((bits >> 12) as f64 + 0.5) * f64::EPSILON
}

#[cfg(test)]
mod tests {
    use super::{KmcRng, StreamRole, open01_from_bits};

    #[test]
    fn maps_bit_extremes_inside_open_unit_interval() {
        let low = open01_from_bits(0);
        assert_eq!(low, 0.5 * f64::EPSILON);
        assert!(low > 0.0);

        let high = open01_from_bits(u64::MAX);
        assert_eq!(high, 1.0 - 0.5 * f64::EPSILON);
        assert!(high < 1.0);
    }

    #[test]
    fn golden_stream_seed_42() {
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
    fn replica_roles_and_max_seeds_are_distinct_and_round_trip() {
        let mut a = KmcRng::for_replica(u64::MAX, u64::MAX, StreamRole::Dynamics);
        let mut b = KmcRng::for_replica(u64::MAX, u64::MAX, StreamRole::InitialState);
        let mut c = KmcRng::for_replica(u64::MAX, u64::MAX - 1, StreamRole::Dynamics);
        let (va, vb, vc) = (
            a.open01().to_bits(),
            b.open01().to_bits(),
            c.open01().to_bits(),
        );
        assert_ne!(va, vb);
        assert_ne!(va, vc);

        let (seed, stream, pos) = a.export_state();
        let mut rebuilt = KmcRng::from_state(seed, stream, pos);
        assert_eq!(a.open01().to_bits(), rebuilt.open01().to_bits());
    }

    #[test]
    fn state_round_trip_at_many_positions() {
        for n in 0..=70 {
            let mut original = KmcRng::with_stream(42, 5);
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
