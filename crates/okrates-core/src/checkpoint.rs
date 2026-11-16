//! Checkpoint schema version 1: error type, model trait and byte reader.

use std::error::Error;
use std::fmt;

use crate::model::Model;

/// The checkpoint schema version written and accepted by this crate.
pub const SCHEMA_VERSION: u32 = 1;

/// The eight magic bytes that start every checkpoint.
pub(crate) const MAGIC: &[u8; 8] = b"OKRCKPT\0";

/// A model that can be saved to and restored from a checkpoint.
pub trait Checkpointable: Model {
    /// A stable identifier for the model type, stored in every checkpoint.
    const MODEL_ID: &'static str;

    /// Appends the model's parameters to `out`.
    ///
    /// A checkpoint resumes only into a model whose encoded parameters are
    /// byte-for-byte equal to the stored ones.
    fn encode_params(&self, out: &mut Vec<u8>);

    /// Appends the encoded `state` to `out`.
    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>);

    /// Decodes and validates a state produced by [`Checkpointable::encode_state`].
    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>>;
}

/// An error while reading a checkpoint.
#[derive(Debug)]
pub enum CheckpointError {
    /// The input does not start with the checkpoint magic bytes.
    BadMagic,
    /// The schema version is not [`SCHEMA_VERSION`].
    UnsupportedVersion(u32),
    /// The input ended early, or a length does not fit in `usize`.
    Truncated,
    /// Bytes remain after the model state section.
    TrailingBytes,
    /// The stored time is not finite and non-negative.
    InvalidTime(f64),
    /// The stored model id differs from the resuming model's id.
    ModelMismatch,
    /// The stored parameters differ from the resuming model's parameters.
    ParamsMismatch,
    /// The model rejected the stored state.
    InvalidState(Box<dyn Error + Send + Sync>),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "checkpoint magic bytes are wrong"),
            Self::UnsupportedVersion(version) => write!(
                f,
                "unsupported checkpoint schema version {version}, expected {SCHEMA_VERSION}"
            ),
            Self::Truncated => write!(f, "checkpoint is truncated"),
            Self::TrailingBytes => write!(f, "checkpoint has bytes after the model state"),
            Self::InvalidTime(time) => write!(f, "checkpoint time {time} is invalid"),
            Self::ModelMismatch => write!(f, "checkpoint model id differs from the model"),
            Self::ParamsMismatch => write!(f, "checkpoint parameters differ from the model"),
            Self::InvalidState(source) => write!(f, "invalid checkpoint state: {source}"),
        }
    }
}

impl Error for CheckpointError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidState(source) => Some(source.as_ref()),
            _ => None,
        }
    }
}

/// A cursor over checkpoint bytes that reports short input as `Truncated`.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], CheckpointError> {
        if self.bytes.len() < n {
            return Err(CheckpointError::Truncated);
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }

    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N], CheckpointError> {
        let mut out = [0_u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    pub(crate) fn u32(&mut self) -> Result<u32, CheckpointError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, CheckpointError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    pub(crate) fn u128(&mut self) -> Result<u128, CheckpointError> {
        Ok(u128::from_le_bytes(self.array()?))
    }

    /// Reads a `u64` length and returns that many bytes.
    pub(crate) fn section(&mut self) -> Result<&'a [u8], CheckpointError> {
        let len = usize::try_from(self.u64()?).map_err(|_| CheckpointError::Truncated)?;
        self.take(len)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}
