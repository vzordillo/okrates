//! Binary checkpoint encoding (schema version 2) with TwoSum clock state.

use std::error::Error;
use std::fmt;

use crate::clock::Clock;
use crate::model::Model;
use crate::rng::KmcRng;

pub const SCHEMA_VERSION: u32 = 2;

pub(crate) const MAGIC: &[u8; 8] = b"OKRCKPT\0";

/// Binary persistence contract for bitwise simulation checkpoint/restart.
pub trait Checkpointable: Model {
    const MODEL_ID: &'static str;

    /// Encodes model parameters; `resume` requires byte-for-byte equality.
    fn encode_params(&self, out: &mut Vec<u8>);

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>);

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>>;
}

#[derive(Debug)]
pub enum CheckpointError {
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    TrailingBytes,
    InvalidTime(f64),
    InvalidCompensation { time: f64, comp: f64 },
    ModelMismatch,
    ParamsMismatch,
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
            Self::InvalidCompensation { time, comp } => {
                write!(
                    f,
                    "checkpoint clock compensation {comp:e} is invalid for time {time:e}"
                )
            }
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

    pub(crate) fn section(&mut self) -> Result<&'a [u8], CheckpointError> {
        let len = usize::try_from(self.u64()?).map_err(|_| CheckpointError::Truncated)?;
        self.take(len)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

pub(crate) fn encode<M: Checkpointable>(
    steps: u64,
    clock: &Clock,
    rng: &KmcRng,
    model: &M,
    state: &M::State,
) -> Vec<u8> {
    let (seed, stream, word_pos) = rng.export_state();
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
    out.extend_from_slice(&steps.to_le_bytes());
    out.extend_from_slice(&clock.time().to_bits().to_le_bytes());
    out.extend_from_slice(&clock.compensation().to_bits().to_le_bytes());
    out.extend_from_slice(&seed);
    out.extend_from_slice(&stream.to_le_bytes());
    out.extend_from_slice(&word_pos.to_le_bytes());
    let id = M::MODEL_ID.as_bytes();
    out.extend_from_slice(&(id.len() as u32).to_le_bytes());
    out.extend_from_slice(id);
    let mut params = Vec::new();
    model.encode_params(&mut params);
    out.extend_from_slice(&(params.len() as u64).to_le_bytes());
    out.extend_from_slice(&params);
    let mut state_bytes = Vec::new();
    model.encode_state(state, &mut state_bytes);
    out.extend_from_slice(&(state_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&state_bytes);
    out
}

pub(crate) fn decode<M: Checkpointable>(
    model: &M,
    bytes: &[u8],
) -> Result<(u64, Clock, KmcRng, M::State), CheckpointError> {
    let mut reader = Reader::new(bytes);
    if reader.take(MAGIC.len())? != MAGIC {
        return Err(CheckpointError::BadMagic);
    }
    let version = reader.u32()?;
    if version != SCHEMA_VERSION {
        return Err(CheckpointError::UnsupportedVersion(version));
    }
    let steps = reader.u64()?;
    let time = f64::from_bits(reader.u64()?);
    let comp = f64::from_bits(reader.u64()?);
    let seed = reader.array::<32>()?;
    let stream = reader.u64()?;
    let word_pos = reader.u128()?;
    let id_len = usize::try_from(reader.u32()?).map_err(|_| CheckpointError::Truncated)?;
    let id = reader.take(id_len)?;
    let params = reader.section()?;
    let state_bytes = reader.section()?;
    if !reader.is_empty() {
        return Err(CheckpointError::TrailingBytes);
    }
    // Sign-bit check rejects -0.0, which Clock::new cannot produce.
    if !time.is_finite() || time.is_sign_negative() {
        return Err(CheckpointError::InvalidTime(time));
    }
    let clock = Clock::from_parts(time, comp)
        .map_err(|_| CheckpointError::InvalidCompensation { time, comp })?;
    if id != M::MODEL_ID.as_bytes() {
        return Err(CheckpointError::ModelMismatch);
    }
    let mut expected = Vec::new();
    model.encode_params(&mut expected);
    if params != expected {
        return Err(CheckpointError::ParamsMismatch);
    }
    let state = model
        .decode_state(state_bytes)
        .map_err(CheckpointError::InvalidState)?;

    Ok((
        steps,
        clock,
        KmcRng::from_state(seed, stream, word_pos),
        state,
    ))
}
