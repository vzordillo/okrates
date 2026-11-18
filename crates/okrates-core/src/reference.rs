//! Directed graph random-walk reference model and decoupled topology/rate components.

use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::checkpoint::Checkpointable;
use crate::incremental::IncrementalModel;
use crate::model::{Model, ReversibleModel};
use crate::topology::{
    CoupledModel, EnumeratedTopology, RateModel, SlottedTopology, TransitionTopology,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hop {
    pub from: usize,
    pub to: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedHop {
    pub edge: usize,
    pub from: usize,
    pub to: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    TargetOutOfRange {
        from: usize,
        to: usize,
        nodes: usize,
    },
    RateCountMismatch {
        expected: usize,
        found: usize,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetOutOfRange { from, to, nodes } => write!(
                f,
                "edge from node {from} targets node {to}, outside {nodes} nodes"
            ),
            Self::RateCountMismatch { expected, found } => {
                write!(f, "expected {expected} edge rates, found {found}")
            }
        }
    }
}

impl Error for GraphError {}

/// Directed multi-graph topology supporting parallel edges and self-loops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphTopology {
    targets: Arc<Vec<Vec<usize>>>,
    offsets: Vec<usize>,
    slot_source: Vec<usize>,
}

impl GraphTopology {
    pub fn new(targets: Vec<Vec<usize>>) -> Result<Self, GraphError> {
        let nodes = targets.len();
        for (from, edges) in targets.iter().enumerate() {
            for &to in edges {
                if to >= nodes {
                    return Err(GraphError::TargetOutOfRange { from, to, nodes });
                }
            }
        }
        let mut offsets = Vec::with_capacity(nodes + 1);
        let mut slot_source = Vec::new();
        offsets.push(0);
        for (from, edges) in targets.iter().enumerate() {
            slot_source.extend(std::iter::repeat_n(from, edges.len()));
            offsets.push(slot_source.len());
        }
        Ok(Self {
            targets: Arc::new(targets),
            offsets,
            slot_source,
        })
    }

    pub fn node_count(&self) -> usize {
        self.targets.len()
    }

    pub fn edge_count(&self) -> usize {
        self.slot_source.len()
    }
}

impl TransitionTopology for GraphTopology {
    type State = usize;
    type Event = IndexedHop;
    type Undo = usize;
    type Error = Infallible;

    fn apply(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        let prev = *state;
        *state = event.to;
        Ok(prev)
    }

    fn undo(&self, state: &mut Self::State, undo: Self::Undo) {
        *state = undo;
    }
}

impl EnumeratedTopology for GraphTopology {
    fn enumerate_candidates(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event)) {
        let Some(edges) = self.targets.get(*state) else {
            return;
        };
        let base = self.offsets[*state];
        for (local, &to) in edges.iter().enumerate() {
            emit(IndexedHop {
                edge: base + local,
                from: *state,
                to,
            });
        }
    }
}

impl SlottedTopology for GraphTopology {
    fn slot_count(&self, _state: &Self::State) -> usize {
        self.slot_source.len()
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        let Some(&from) = self.slot_source.get(slot) else {
            return IndexedHop {
                edge: slot,
                from: usize::MAX,
                to: usize::MAX,
            };
        };
        IndexedHop {
            edge: slot,
            from,
            to: self.targets[from][slot - self.offsets[from]],
        }
    }

    fn is_slot_active(&self, state: &Self::State, slot: usize) -> bool {
        self.slot_source.get(slot) == Some(state)
    }

    fn affected_slots(&self, _state: &Self::State, event: &Self::Event, out: &mut Vec<usize>) {
        for node in [event.from, event.to] {
            if node < self.targets.len() {
                out.extend(self.offsets[node]..self.offsets[node + 1]);
            }
        }
    }
}

/// Per-edge rate table pairing with [`GraphTopology`] via [`CoupledModel`].
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeRateTable {
    targets: Arc<Vec<Vec<usize>>>,
    rates: Vec<f64>,
}

impl EdgeRateTable {
    pub fn new(topology: &GraphTopology, rates: Vec<f64>) -> Result<Self, GraphError> {
        if rates.len() != topology.edge_count() {
            return Err(GraphError::RateCountMismatch {
                expected: topology.edge_count(),
                found: rates.len(),
            });
        }
        Ok(Self {
            targets: Arc::clone(&topology.targets),
            rates,
        })
    }
}

impl RateModel<GraphTopology> for EdgeRateTable {
    fn rate(&self, topology: &GraphTopology, _state: &usize, event: &IndexedHop) -> f64 {
        if self.targets != topology.targets {
            return f64::NAN;
        }
        let Some(&from) = topology.slot_source.get(event.edge) else {
            return f64::NAN;
        };
        let to = topology.targets[from][event.edge - topology.offsets[from]];
        if from != event.from || to != event.to {
            return f64::NAN;
        }
        self.rates.get(event.edge).copied().unwrap_or(f64::NAN)
    }
}

pub type CoupledGraphWalk = CoupledModel<GraphTopology, EdgeRateTable>;

/// Weighted directed graph random walk with `(target, rate)` adjacency lists.
///
/// Parallel edges and self-loops are distinct events.
#[derive(Clone, Debug)]
pub struct GraphRandomWalk {
    adjacency: Vec<Vec<(usize, f64)>>,
    offsets: Vec<usize>,
    slot_source: Vec<usize>,
}

impl GraphRandomWalk {
    pub fn new(adjacency: Vec<Vec<(usize, f64)>>) -> Result<Self, GraphError> {
        let nodes = adjacency.len();
        for (from, edges) in adjacency.iter().enumerate() {
            for &(to, _) in edges {
                if to >= nodes {
                    return Err(GraphError::TargetOutOfRange { from, to, nodes });
                }
            }
        }
        let mut offsets = Vec::with_capacity(nodes + 1);
        let mut slot_source = Vec::new();
        offsets.push(0);
        for (from, edges) in adjacency.iter().enumerate() {
            slot_source.extend(std::iter::repeat_n(from, edges.len()));
            offsets.push(slot_source.len());
        }
        Ok(Self {
            adjacency,
            offsets,
            slot_source,
        })
    }

    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }
}

impl Model for GraphRandomWalk {
    type State = usize;
    type Event = Hop;
    type Error = Infallible;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        if let Some(edges) = self.adjacency.get(*state) {
            for &(to, rate) in edges {
                emit(Hop { from: *state, to }, rate);
            }
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        *state = event.to;
        Ok(())
    }
}

impl ReversibleModel for GraphRandomWalk {
    type Undo = usize;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        let prev = *state;
        *state = event.to;
        Ok(prev)
    }

    fn undo(&self, state: &mut Self::State, undo: Self::Undo) {
        *state = undo;
    }
}

impl IncrementalModel for GraphRandomWalk {
    fn slot_count(&self, _state: &Self::State) -> usize {
        self.slot_source.len()
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        let Some(&from) = self.slot_source.get(slot) else {
            return f64::NAN;
        };
        if from == *state {
            self.adjacency[from][slot - self.offsets[from]].1
        } else {
            0.0
        }
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        let Some(&from) = self.slot_source.get(slot) else {
            return Hop {
                from: usize::MAX,
                to: usize::MAX,
            };
        };
        Hop {
            from,
            to: self.adjacency[from][slot - self.offsets[from]].0,
        }
    }

    fn affected_slots(&self, _state: &Self::State, event: &Self::Event, out: &mut Vec<usize>) {
        for node in [event.from, event.to] {
            if node < self.adjacency.len() {
                out.extend(self.offsets[node]..self.offsets[node + 1]);
            }
        }
    }
}

impl Checkpointable for GraphRandomWalk {
    const MODEL_ID: &'static str = "okrates.graph_random_walk";

    fn encode_params(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.adjacency.len() as u64).to_le_bytes());
        for edges in &self.adjacency {
            out.extend_from_slice(&(edges.len() as u64).to_le_bytes());
            for &(to, rate) in edges {
                out.extend_from_slice(&(to as u64).to_le_bytes());
                out.extend_from_slice(&rate.to_bits().to_le_bytes());
            }
        }
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        out.extend_from_slice(&(*state as u64).to_le_bytes());
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        let bytes: [u8; 8] = bytes.try_into().map_err(|_| "state must be 8 bytes")?;
        let node = u64::from_le_bytes(bytes);
        match usize::try_from(node) {
            Ok(node) if node < self.node_count() => Ok(node),
            _ => Err(format!("node {node} is outside {} nodes", self.node_count()).into()),
        }
    }
}
