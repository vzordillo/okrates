//! A simple directed graph random walk model.

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use crate::checkpoint::Checkpointable;
use crate::incremental::IncrementalModel;
use crate::model::Model;

/// A hop from one graph node to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hop {
    /// The node the walker leaves.
    pub from: usize,
    /// The node the walker enters.
    pub to: usize,
}

/// A random walk on a weighted directed graph.
///
/// Each adjacency entry is a `(target, rate)` pair. Parallel edges and
/// self-loops are separate events.
#[derive(Clone, Debug)]
pub struct GraphRandomWalk {
    adjacency: Vec<Vec<(usize, f64)>>,
    offsets: Vec<usize>,
    slot_source: Vec<usize>,
}

impl GraphRandomWalk {
    /// Creates a graph, rejecting any edge whose target is outside the graph.
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

    /// Returns the number of nodes in this graph.
    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }
}

/// An invalid graph edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// An edge target is outside the graph's node range.
    TargetOutOfRange {
        /// The source node of the invalid edge.
        from: usize,
        /// The invalid target node.
        to: usize,
        /// The number of nodes in the graph.
        nodes: usize,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetOutOfRange { from, to, nodes } => write!(
                f,
                "edge from node {from} targets node {to}, outside {nodes} nodes"
            ),
        }
    }
}

impl Error for GraphError {}

impl Model for GraphRandomWalk {
    type State = usize;
    type Event = Hop;
    type Error = Infallible;

    /// Emits outgoing edges in adjacency order.
    ///
    /// # Panics
    ///
    /// Panics if `state >= node_count()`.
    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        for &(to, rate) in &self.adjacency[*state] {
            emit(Hop { from: *state, to }, rate);
        }
    }

    /// Sets the walker to the hop's target node.
    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        *state = event.to;
        Ok(())
    }
}

impl IncrementalModel for GraphRandomWalk {
    /// Returns the total number of edges; slots are global edge indices.
    fn slot_count(&self, _state: &Self::State) -> usize {
        self.slot_source.len()
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        let from = self.slot_source[slot];
        if from == *state {
            self.adjacency[from][slot - self.offsets[from]].1
        } else {
            0.0
        }
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        let from = self.slot_source[slot];
        Hop {
            from,
            to: self.adjacency[from][slot - self.offsets[from]].0,
        }
    }

    fn affected_slots(&self, _state: &Self::State, event: &Self::Event, out: &mut Vec<usize>) {
        for node in [event.from, event.to] {
            out.extend(self.offsets[node]..self.offsets[node + 1]);
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
