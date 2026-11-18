//! Deterministic parallel replica ensembles on the active Rayon thread pool.
//!
//! Results are collected in replica `id` order (`0..count`) so reductions are
//! independent of thread count and work-stealing schedule.

use std::fmt;

use rayon::prelude::*;

use crate::rng::{KmcRng, StreamRole};

#[derive(Clone, Debug, PartialEq)]
pub struct Replica<T> {
    pub id: u64,
    pub seed: u64,
    pub value: T,
}

/// Domain-separated replica handle supporting arbitrary `u64` seeds without overflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplicaContext {
    pub id: u64,
    pub base_seed: u64,
}

impl ReplicaContext {
    pub fn rng(&self, role: StreamRole) -> KmcRng {
        KmcRng::for_replica(self.base_seed, self.id, role)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnsembleError {
    SeedOverflow { base_seed: u64, count: u64 },
}

impl fmt::Display for EnsembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SeedOverflow { base_seed, count } => write!(
                f,
                "seed overflow: base seed {base_seed} plus {count} replicas exceeds u64"
            ),
        }
    }
}

impl std::error::Error for EnsembleError {}

pub fn replica_seed(base_seed: u64, id: u64) -> Option<u64> {
    base_seed.checked_add(id)
}

pub fn thread_count() -> usize {
    rayon::current_num_threads()
}

/// Runs `count` replicas in parallel using additive seed `base_seed + id`.
pub fn run_replicas<T, F>(
    base_seed: u64,
    count: u64,
    f: F,
) -> Result<Vec<Replica<T>>, EnsembleError>
where
    T: Send,
    F: Fn(u64, u64) -> T + Sync,
{
    if count > 0 && replica_seed(base_seed, count - 1).is_none() {
        return Err(EnsembleError::SeedOverflow { base_seed, count });
    }
    Ok((0..count)
        .into_par_iter()
        .map(|id| {
            let seed = base_seed + id;
            Replica {
                id,
                seed,
                value: f(id, seed),
            }
        })
        .collect::<Vec<_>>())
}

/// Runs `count` replicas in parallel with collision-free [`ReplicaContext`] substreams.
pub fn run_ensemble<T, F>(base_seed: u64, count: u64, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(ReplicaContext) -> T + Sync,
{
    (0..count)
        .into_par_iter()
        .map(|id| f(ReplicaContext { id, base_seed }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_overflow_is_detected() {
        assert_eq!(replica_seed(u64::MAX, 1), None);
        assert_eq!(replica_seed(u64::MAX, 0), Some(u64::MAX));
    }

    #[test]
    fn overflowing_ensemble_is_rejected() {
        assert_eq!(
            run_replicas(u64::MAX, 2, |id, _| id),
            Err(EnsembleError::SeedOverflow {
                base_seed: u64::MAX,
                count: 2
            })
        );
    }

    #[test]
    fn run_ensemble_handles_u64_max_seed_without_overflow() {
        let vals = run_ensemble(u64::MAX, 4, |ctx| {
            ctx.rng(StreamRole::Dynamics).open01().to_bits()
        });
        assert_eq!(vals.len(), 4);
        assert_ne!(vals[0], vals[1]);
    }

    #[test]
    fn replicas_come_back_in_id_order() {
        let replicas = run_replicas(10, 5, |id, seed| (id, seed)).unwrap();
        assert_eq!(replicas.len(), 5);
        for (index, replica) in replicas.iter().enumerate() {
            let id = index as u64;
            assert_eq!(replica.id, id);
            assert_eq!(replica.seed, 10 + id);
            assert_eq!(replica.value, (id, 10 + id));
        }
    }

    #[test]
    fn zero_replicas_give_an_empty_vec() {
        assert!(run_replicas(3, 0, |id, _| id).unwrap().is_empty());
    }
}
