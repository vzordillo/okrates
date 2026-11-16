//! Parallel ensembles of independent replicas.
//!
//! Independent replicas run in parallel on the current rayon thread pool.
//! Replica `id` uses seed `base_seed + id`. Results are returned in id order,
//! so they do not depend on the number of threads or on scheduling. Callers
//! must aggregate serially in id order.

use std::fmt;

use rayon::prelude::*;

/// One replica's result with its id and seed.
#[derive(Clone, Debug, PartialEq)]
pub struct Replica<T> {
    /// Replica index, from 0 to `count - 1`.
    pub id: u64,
    /// Seed used for this replica: `base_seed + id`.
    pub seed: u64,
    /// Value computed by the replica function.
    pub value: T,
}

/// Error: the seed of the last replica does not fit in u64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnsembleError {
    /// `base_seed + (count - 1)` overflows `u64`.
    SeedOverflow {
        /// Base seed of the ensemble.
        base_seed: u64,
        /// Number of replicas requested.
        count: u64,
    },
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

/// The seed of replica `id`: `base_seed + id`, or `None` on overflow.
pub fn replica_seed(base_seed: u64, id: u64) -> Option<u64> {
    base_seed.checked_add(id)
}

/// Number of threads in the current rayon pool.
pub fn thread_count() -> usize {
    rayon::current_num_threads()
}

/// Runs `count` replicas in parallel; `f(id, seed)` computes one replica.
///
/// Returns the replicas in id order. Fails with
/// [`EnsembleError::SeedOverflow`] if the seed of the last replica does not
/// fit in `u64`.
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
    fn last_possible_seed_is_accepted() {
        let replicas = run_replicas(u64::MAX, 1, |_, seed| seed).unwrap();
        assert_eq!(replicas.len(), 1);
        assert_eq!(replicas[0].seed, u64::MAX);
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
