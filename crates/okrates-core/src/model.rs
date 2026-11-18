//! Core model and reversible transition contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;

use crate::checkpoint::Checkpointable;
use crate::incremental::IncrementalModel;

/// Baseline KMC model contract: enumerates `(event, rate)` pairs and applies transitions.
///
/// Rates are in inverse time units; zero-rate events are ignored by the engine.
/// Implementations must leave `state` logically unchanged when `apply` returns `Err`.
pub trait Model {
    type State;
    type Event;
    type Error: Error;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64));

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error>;
}

/// Extension of [`Model`] with a rollback token so post-transition catalog validation
/// failures restore `state` unchanged; `apply_reversible` must also leave `state` unchanged on `Err`.
pub trait ReversibleModel: Model {
    type Undo;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error>;

    fn undo(&self, state: &mut Self::State, undo: Self::Undo);
}

/// Owned value snapshot with no shared interior mutability, ensuring rollback cannot alias mutated state.
pub trait IndependentSnapshot: Sized {
    fn snapshot(&self) -> Self;
}

macro_rules! impl_copy_snapshot {
    ($($t:ty),* $(,)?) => {
        $(
            impl IndependentSnapshot for $t {
                fn snapshot(&self) -> Self {
                    *self
                }
            }
        )*
    };
}

impl_copy_snapshot!(
    (),
    bool,
    char,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    f32,
    f64,
);

impl<T: IndependentSnapshot, const N: usize> IndependentSnapshot for [T; N] {
    fn snapshot(&self) -> Self {
        std::array::from_fn(|i| self[i].snapshot())
    }
}

macro_rules! impl_tuple_snapshot {
    ($(($($idx:tt : $T:ident),+)),+ $(,)?) => {
        $(
            impl<$($T: IndependentSnapshot),+> IndependentSnapshot for ($($T,)+) {
                fn snapshot(&self) -> Self {
                    ($(self.$idx.snapshot(),)+)
                }
            }
        )+
    };
}

impl_tuple_snapshot!(
    (0: A),
    (0: A, 1: B),
    (0: A, 1: B, 2: C),
    (0: A, 1: B, 2: C, 3: D),
    (0: A, 1: B, 2: C, 3: D, 4: E),
    (0: A, 1: B, 2: C, 3: D, 4: E, 5: F),
);

impl<T: IndependentSnapshot> IndependentSnapshot for Option<T> {
    fn snapshot(&self) -> Self {
        self.as_ref().map(IndependentSnapshot::snapshot)
    }
}

impl<T: IndependentSnapshot> IndependentSnapshot for Vec<T> {
    fn snapshot(&self) -> Self {
        self.iter().map(IndependentSnapshot::snapshot).collect()
    }
}

impl<T: IndependentSnapshot> IndependentSnapshot for Box<T> {
    fn snapshot(&self) -> Self {
        Box::new((**self).snapshot())
    }
}

impl<T: IndependentSnapshot> IndependentSnapshot for Box<[T]> {
    fn snapshot(&self) -> Self {
        self.iter().map(IndependentSnapshot::snapshot).collect()
    }
}

impl IndependentSnapshot for String {
    fn snapshot(&self) -> Self {
        self.clone()
    }
}

impl<K: IndependentSnapshot + Ord, V: IndependentSnapshot> IndependentSnapshot for BTreeMap<K, V> {
    fn snapshot(&self) -> Self {
        self.iter()
            .map(|(k, v)| (k.snapshot(), v.snapshot()))
            .collect()
    }
}

impl<T: IndependentSnapshot + Ord> IndependentSnapshot for BTreeSet<T> {
    fn snapshot(&self) -> Self {
        self.iter().map(IndependentSnapshot::snapshot).collect()
    }
}

/// Adapter supplying [`ReversibleModel`] for any [`Model`] whose `State: IndependentSnapshot` (`O(|State|)` per step).
///
/// Shared interior-mutable handles such as `Rc<Cell<_>>` are statically rejected:
///
/// ```compile_fail,E0277
/// use std::cell::Cell;
/// use std::convert::Infallible;
/// use std::rc::Rc;
/// use okrates_core::{CloneRollback, Model, ReversibleModel};
///
/// struct OwnedModel;
/// impl Model for OwnedModel {
///     type State = u64;
///     type Event = ();
///     type Error = Infallible;
///     fn enumerate_events(&self, _: &u64, _: &mut dyn FnMut((), f64)) {}
///     fn apply(&self, s: &mut u64, (): &()) -> Result<(), Infallible> { *s += 1; Ok(()) }
/// }
///
/// struct SharedModel;
/// impl Model for SharedModel {
///     type State = Rc<Cell<u64>>;
///     type Event = ();
///     type Error = Infallible;
///     fn enumerate_events(&self, _: &Self::State, _: &mut dyn FnMut((), f64)) {}
///     fn apply(&self, s: &mut Self::State, (): &()) -> Result<(), Infallible> { s.set(1); Ok(()) }
/// }
///
/// fn require_reversible<M: ReversibleModel>(_: M) {}
/// require_reversible(CloneRollback(OwnedModel));
/// require_reversible(CloneRollback(SharedModel));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CloneRollback<M>(pub M);

impl<M> Model for CloneRollback<M>
where
    M: Model,
    M::State: IndependentSnapshot,
{
    type State = M::State;
    type Event = M::Event;
    type Error = M::Error;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        self.0.enumerate_events(state, emit);
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        let snapshot = state.snapshot();
        if let Err(err) = self.0.apply(state, event) {
            *state = snapshot;
            return Err(err);
        }
        Ok(())
    }
}

impl<M> ReversibleModel for CloneRollback<M>
where
    M: Model,
    M::State: IndependentSnapshot,
{
    type Undo = M::State;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        let snapshot = state.snapshot();
        if let Err(err) = self.0.apply(state, event) {
            *state = snapshot;
            return Err(err);
        }
        Ok(snapshot)
    }

    fn undo(&self, state: &mut Self::State, undo: Self::Undo) {
        *state = undo;
    }
}

impl<M> IncrementalModel for CloneRollback<M>
where
    M: IncrementalModel,
    M::State: IndependentSnapshot,
{
    fn slot_count(&self, state: &Self::State) -> usize {
        self.0.slot_count(state)
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        self.0.slot_rate(state, slot)
    }

    fn slot_event(&self, state: &Self::State, slot: usize) -> Self::Event {
        self.0.slot_event(state, slot)
    }

    fn affected_slots(&self, state: &Self::State, event: &Self::Event, out: &mut Vec<usize>) {
        self.0.affected_slots(state, event, out);
    }
}

impl<M> Checkpointable for CloneRollback<M>
where
    M: Checkpointable,
    M::State: IndependentSnapshot,
{
    const MODEL_ID: &'static str = M::MODEL_ID;

    fn encode_params(&self, out: &mut Vec<u8>) {
        self.0.encode_params(out);
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        self.0.encode_state(state, out);
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        self.0.decode_state(bytes)
    }
}
