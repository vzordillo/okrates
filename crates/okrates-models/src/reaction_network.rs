//! Well-mixed stochastic chemical reaction network template using mass-action propensities.
//!
//! Combinatorial hazard functions and stochastic simulation follow Gillespie,
//! *J. Comput. Phys.* **22**, 403–434 (1976), doi:10.1016/0021-9991(76)90041-3,
//! and *J. Phys. Chem.* **81**, 2340–2361 (1977), doi:10.1021/j100540a008.

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt;

use okrates_core::{Checkpointable, IncrementalModel, Model, ReversibleModel, SumTree};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoichiometryRole {
    Reactant,
    Product,
}

impl fmt::Display for StoichiometryRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reactant => f.write_str("reactant"),
            Self::Product => f.write_str("product"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReactionError {
    ZeroSpecies,
    EmptyNetwork,
    TooManyChannels(usize),
    SpeciesOutOfRange {
        channel: usize,
        species: usize,
        species_count: usize,
    },
    ZeroStoichiometry {
        channel: usize,
        species: usize,
        role: StoichiometryRole,
    },
    DuplicateSpecies {
        channel: usize,
        species: usize,
        role: StoichiometryRole,
    },
    InvalidRateConstant {
        channel: usize,
        rate_constant: f64,
    },
    InvalidReverseChannel {
        channel: usize,
        reverse_channel: usize,
    },
    ReversePairMismatch {
        channel: usize,
        reverse_channel: usize,
    },
    ConservationLawWidth {
        law: usize,
        expected: usize,
        found: usize,
    },
    ConservationStoichiometryMismatch {
        law: usize,
        channel: usize,
        net_delta: i128,
    },
    StateSpeciesMismatch {
        expected: usize,
        found: usize,
    },
    StateChannelMismatch {
        expected: usize,
        found: usize,
    },
    ChannelOutOfRange {
        channel: usize,
        channel_count: usize,
    },
    InsufficientReactants {
        channel: usize,
        species: usize,
        required: u32,
        available: u64,
    },
    CountOverflow {
        channel: usize,
        species: usize,
    },
    FiredCountOverflow {
        channel: usize,
    },
    ConservationOverflow {
        law: usize,
    },
    ConservationViolation {
        law: usize,
        initial: i128,
        current: i128,
    },
}

impl fmt::Display for ReactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSpecies => f.write_str("reaction network requires at least one species"),
            Self::EmptyNetwork => f.write_str("reaction network requires at least one channel"),
            Self::TooManyChannels(n) => write!(f, "channel count {n} exceeds sum-tree capacity"),
            Self::SpeciesOutOfRange {
                channel,
                species,
                species_count,
            } => write!(
                f,
                "channel {channel} references species {species} out of {species_count}"
            ),
            Self::ZeroStoichiometry {
                channel,
                species,
                role,
            } => write!(
                f,
                "channel {channel} {role} entry for species {species} must have positive stoichiometry"
            ),
            Self::DuplicateSpecies {
                channel,
                species,
                role,
            } => write!(
                f,
                "channel {channel} lists duplicate {role} species {species}"
            ),
            Self::InvalidRateConstant {
                channel,
                rate_constant,
            } => write!(
                f,
                "channel {channel} has invalid rate constant {rate_constant}"
            ),
            Self::InvalidReverseChannel {
                channel,
                reverse_channel,
            } => write!(
                f,
                "channel {channel} declares invalid reverse channel {reverse_channel}"
            ),
            Self::ReversePairMismatch {
                channel,
                reverse_channel,
            } => write!(
                f,
                "channel {channel} and reverse channel {reverse_channel} do not form an inverse pair"
            ),
            Self::ConservationLawWidth {
                law,
                expected,
                found,
            } => write!(
                f,
                "conservation law {law} has width {found}, expected {expected}"
            ),
            Self::ConservationStoichiometryMismatch {
                law,
                channel,
                net_delta,
            } => write!(
                f,
                "conservation law {law} is not invariant under channel {channel} (delta={net_delta})"
            ),
            Self::StateSpeciesMismatch { expected, found } => {
                write!(f, "state has {found} species counts, expected {expected}")
            }
            Self::StateChannelMismatch { expected, found } => {
                write!(f, "state has {found} channel counters, expected {expected}")
            }
            Self::ChannelOutOfRange {
                channel,
                channel_count,
            } => write!(
                f,
                "reaction channel {channel} out of range for {channel_count} channels"
            ),
            Self::InsufficientReactants {
                channel,
                species,
                required,
                available,
            } => write!(
                f,
                "channel {channel} needs {required} of species {species}, only {available} present"
            ),
            Self::CountOverflow { channel, species } => {
                write!(
                    f,
                    "channel {channel} overflows copy count for species {species}"
                )
            }
            Self::FiredCountOverflow { channel } => {
                write!(f, "channel {channel} firing counter overflowed u64")
            }
            Self::ConservationOverflow { law } => {
                write!(f, "conservation law {law} sum overflowed i128")
            }
            Self::ConservationViolation {
                law,
                initial,
                current,
            } => write!(
                f,
                "conservation law {law} changed from {initial} to {current}"
            ),
        }
    }
}

impl Error for ReactionError {}

/// Single reaction channel with integer stoichiometry `(species_index, coefficient)` and mass-action rate constant `k >= 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct ReactionChannel {
    pub reactants: Vec<(usize, u32)>,
    pub products: Vec<(usize, u32)>,
    pub rate_constant: f64,
    pub reverse_channel: Option<usize>,
}

impl ReactionChannel {
    pub fn new(
        reactants: Vec<(usize, u32)>,
        products: Vec<(usize, u32)>,
        rate_constant: f64,
    ) -> Self {
        Self {
            reactants,
            products,
            rate_constant,
            reverse_channel: None,
        }
    }

    /// Sets the reverse channel index; [`ReactionNetwork::new`] enforces that the pair is reciprocal.
    pub fn with_reverse(mut self, reverse_channel: usize) -> Self {
        self.reverse_channel = Some(reverse_channel);
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CompiledChannel {
    reactants: Vec<(usize, u32)>,
    products: Vec<(usize, u32)>,
    net_delta: Vec<(usize, i64)>,
    rate_constant: f64,
    reverse_channel: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReactionEvent(pub usize);

/// Allocation-free rollback token for [`ReversibleModel::undo`].
///
/// Must be consumed once on the exact state produced by its matching
/// [`ReversibleModel::apply_reversible`] call before any other mutation.
#[derive(Debug, PartialEq, Eq)]
pub struct ReactionUndo {
    channel: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReactionState {
    pub counts: Vec<u64>,
    pub fired_counts: Vec<u64>,
}

impl ReactionState {
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    pub fn fired_counts(&self) -> &[u64] {
        &self.fired_counts
    }

    pub fn total_firings(&self) -> u128 {
        self.fired_counts.iter().copied().map(u128::from).sum()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReactionNetwork {
    species_count: usize,
    channels: Vec<CompiledChannel>,
    affected_channels: Vec<Vec<usize>>,
    conservation_laws: Vec<Vec<i64>>,
}

fn validate_stoich_list(
    channel: usize,
    role: StoichiometryRole,
    entries: &[(usize, u32)],
    species_count: usize,
) -> Result<Vec<(usize, u32)>, ReactionError> {
    let mut map = BTreeMap::new();
    for &(species, coeff) in entries {
        if species >= species_count {
            return Err(ReactionError::SpeciesOutOfRange {
                channel,
                species,
                species_count,
            });
        }
        if coeff == 0 {
            return Err(ReactionError::ZeroStoichiometry {
                channel,
                species,
                role,
            });
        }
        if map.insert(species, coeff).is_some() {
            return Err(ReactionError::DuplicateSpecies {
                channel,
                species,
                role,
            });
        }
    }
    Ok(map.into_iter().collect())
}

fn compile_net_delta(reactants: &[(usize, u32)], products: &[(usize, u32)]) -> Vec<(usize, i64)> {
    let mut delta: BTreeMap<usize, i64> = BTreeMap::new();
    for &(species, coeff) in reactants {
        *delta.entry(species).or_insert(0) -= i64::from(coeff);
    }
    for &(species, coeff) in products {
        *delta.entry(species).or_insert(0) += i64::from(coeff);
    }
    delta.into_iter().filter(|&(_, d)| d != 0).collect()
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Exact combinatorial binomial coefficient `C(n, k)` using `gcd` cancellation to minimize overflow.
fn checked_binom_u128(n: u64, k: u32) -> Option<u128> {
    let k = u64::from(k);
    if k > n {
        return Some(0);
    }
    if k == 0 || k == n {
        return Some(1);
    }
    let k = k.min(n - k);
    let mut acc = 1_u128;
    for i in 1..=k {
        let num = u128::from(n - k + i);
        let den = u128::from(i);
        let g = gcd_u128(acc, den);
        acc = (acc / g).checked_mul(num / (den / g))?;
    }
    Some(acc)
}

const POW2_POS_512: f64 = f64::from_bits((1023_u64 + 512) << 52);
const POW2_NEG_512: f64 = f64::from_bits((1023_u64 - 512) << 52);
const POW2_64: f64 = f64::from_bits((1023_u64 + 64) << 52);
const MANTISSA_MASK: u64 = (1_u64 << 52) - 1;
const EXP_ONE_BITS: u64 = 1023_u64 << 52;
// C(2k, k) >= 2^(2k)/(2k+1); for k >= 1060 this exceeds f64::MAX / f64::MIN_POSITIVE (~2^2098).
const MAX_SCALED_BINOM_K: u64 = 1060;

fn normalize_pos_f64(mut x: f64) -> (f64, i64) {
    let mut exp = 0_i64;
    if x < f64::MIN_POSITIVE {
        x *= POW2_64;
        exp -= 64;
    }
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    exp += e;
    let mant = f64::from_bits((bits & MANTISSA_MASK) | EXP_ONE_BITS);
    (mant, exp)
}

fn scale_exp2(mut mant: f64, mut exp: i64) -> f64 {
    if mant.is_nan() {
        return f64::NAN;
    }
    if mant <= 0.0 {
        return 0.0;
    }
    if mant.is_infinite() {
        return f64::INFINITY;
    }
    let (m, e) = normalize_pos_f64(mant);
    mant = m;
    exp = exp.saturating_add(e);
    if exp >= 1024 {
        return f64::INFINITY;
    }
    if exp <= -2048 {
        return 0.0;
    }
    while exp > 512 {
        mant *= POW2_POS_512;
        exp -= 512;
    }
    while exp < -512 {
        mant *= POW2_NEG_512;
        exp += 512;
    }
    let pow = f64::from_bits(((1023_i64 + exp) as u64) << 52);
    mant * pow
}

fn mul_scaled(lhs: (f64, i64), rhs: (f64, i64)) -> (f64, i64) {
    let (mant, me) = normalize_pos_f64(lhs.0 * rhs.0);
    (mant, lhs.1.saturating_add(rhs.1).saturating_add(me))
}

fn binom_scaled(n: u64, k: u32) -> Option<(f64, i64)> {
    let k = u64::from(k);
    if k > n {
        return None;
    }
    let k = k.min(n - k);
    if k == 0 {
        return Some((1.0, 0));
    }
    if k >= MAX_SCALED_BINOM_K {
        return None;
    }
    let mut acc = (1.0_f64, 0_i64);
    for i in 1..=k {
        let (nm, ne) = normalize_pos_f64((n - k + i) as f64);
        let (dm, de) = normalize_pos_f64(i as f64);
        let (pm, pe) = normalize_pos_f64(acc.0 * (nm / dm));
        acc = (pm, acc.1.saturating_add(ne - de + pe));
    }
    Some(acc)
}

impl ReactionNetwork {
    /// Builds and validates a reaction network, enforcing reciprocal reverse pairs and precomputing dependencies.
    pub fn new(
        species_count: usize,
        channels: Vec<ReactionChannel>,
    ) -> Result<Self, ReactionError> {
        if species_count == 0 {
            return Err(ReactionError::ZeroSpecies);
        }
        if channels.is_empty() {
            return Err(ReactionError::EmptyNetwork);
        }
        SumTree::check_len(channels.len())
            .map_err(|_| ReactionError::TooManyChannels(channels.len()))?;

        let mut compiled = Vec::with_capacity(channels.len());
        let mut by_reactant_species: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

        for (idx, ch) in channels.iter().enumerate() {
            if !ch.rate_constant.is_finite() || ch.rate_constant < 0.0 {
                return Err(ReactionError::InvalidRateConstant {
                    channel: idx,
                    rate_constant: ch.rate_constant,
                });
            }
            let rate_constant = if ch.rate_constant == 0.0 {
                0.0
            } else {
                ch.rate_constant
            };
            let reactants = validate_stoich_list(
                idx,
                StoichiometryRole::Reactant,
                &ch.reactants,
                species_count,
            )?;
            let products =
                validate_stoich_list(idx, StoichiometryRole::Product, &ch.products, species_count)?;
            for &(species, _) in &reactants {
                by_reactant_species.entry(species).or_default().push(idx);
            }
            let net_delta = compile_net_delta(&reactants, &products);
            compiled.push(CompiledChannel {
                reactants,
                products,
                net_delta,
                rate_constant,
                reverse_channel: ch.reverse_channel,
            });
        }

        for (idx, ch) in compiled.iter().enumerate() {
            if let Some(rev) = ch.reverse_channel {
                let Some(partner) = compiled.get(rev) else {
                    return Err(ReactionError::InvalidReverseChannel {
                        channel: idx,
                        reverse_channel: rev,
                    });
                };
                if rev == idx || partner.reverse_channel != Some(idx) {
                    return Err(ReactionError::InvalidReverseChannel {
                        channel: idx,
                        reverse_channel: rev,
                    });
                }
                if ch.reactants != partner.products || ch.products != partner.reactants {
                    return Err(ReactionError::ReversePairMismatch {
                        channel: idx,
                        reverse_channel: rev,
                    });
                }
            }
        }

        let mut affected_channels = Vec::with_capacity(compiled.len());
        let mut seen = HashSet::new();
        for ch in &compiled {
            seen.clear();
            let mut deps = Vec::new();
            for &(species, _) in &ch.net_delta {
                if let Some(consumers) = by_reactant_species.get(&species) {
                    for &dep_channel in consumers {
                        if seen.insert(dep_channel) {
                            deps.push(dep_channel);
                        }
                    }
                }
            }
            deps.sort_unstable();
            affected_channels.push(deps);
        }

        Ok(Self {
            species_count,
            channels: compiled,
            affected_channels,
            conservation_laws: Vec::new(),
        })
    }

    /// Attaches stoichiometric linear conservation vectors `w` and verifies `w · delta_r == 0` for every channel.
    pub fn with_conservation_laws(mut self, laws: Vec<Vec<i64>>) -> Result<Self, ReactionError> {
        for (law_idx, weights) in laws.iter().enumerate() {
            if weights.len() != self.species_count {
                return Err(ReactionError::ConservationLawWidth {
                    law: law_idx,
                    expected: self.species_count,
                    found: weights.len(),
                });
            }
            for (ch_idx, ch) in self.channels.iter().enumerate() {
                let mut sum = 0_i128;
                for &(species, delta) in &ch.net_delta {
                    sum = sum
                        .checked_add(i128::from(weights[species]) * i128::from(delta))
                        .ok_or(ReactionError::ConservationOverflow { law: law_idx })?;
                }
                if sum != 0 {
                    return Err(ReactionError::ConservationStoichiometryMismatch {
                        law: law_idx,
                        channel: ch_idx,
                        net_delta: sum,
                    });
                }
            }
        }
        self.conservation_laws = laws;
        Ok(self)
    }

    pub fn species_count(&self) -> usize {
        self.species_count
    }

    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    pub fn initial_state(&self, counts: Vec<u64>) -> Result<ReactionState, ReactionError> {
        if counts.len() != self.species_count {
            return Err(ReactionError::StateSpeciesMismatch {
                expected: self.species_count,
                found: counts.len(),
            });
        }
        Ok(ReactionState {
            counts,
            fired_counts: vec![0; self.channels.len()],
        })
    }

    pub fn check_conservation(
        &self,
        initial: &ReactionState,
        current: &ReactionState,
    ) -> Result<(), ReactionError> {
        for state in [initial, current] {
            if state.counts.len() != self.species_count {
                return Err(ReactionError::StateSpeciesMismatch {
                    expected: self.species_count,
                    found: state.counts.len(),
                });
            }
            if state.fired_counts.len() != self.channels.len() {
                return Err(ReactionError::StateChannelMismatch {
                    expected: self.channels.len(),
                    found: state.fired_counts.len(),
                });
            }
        }
        let invariant_sum = |law_idx: usize, counts: &[u64]| -> Result<i128, ReactionError> {
            let weights = &self.conservation_laws[law_idx];
            let mut total = 0_i128;
            for (&w, &c) in weights.iter().zip(counts) {
                let term = i128::from(w)
                    .checked_mul(i128::from(c))
                    .ok_or(ReactionError::ConservationOverflow { law: law_idx })?;
                total = total
                    .checked_add(term)
                    .ok_or(ReactionError::ConservationOverflow { law: law_idx })?;
            }
            Ok(total)
        };
        for law_idx in 0..self.conservation_laws.len() {
            let init_val = invariant_sum(law_idx, &initial.counts)?;
            let curr_val = invariant_sum(law_idx, &current.counts)?;
            if init_val != curr_val {
                return Err(ReactionError::ConservationViolation {
                    law: law_idx,
                    initial: init_val,
                    current: curr_val,
                });
            }
        }
        Ok(())
    }

    /// Evaluates the combinatorial mass-action propensity `k_r * prod_i C(n_i, nu_i^-)` for `channel`.
    pub fn propensity(&self, state: &ReactionState, channel: usize) -> f64 {
        let Some(ch) = self.channels.get(channel) else {
            return f64::NAN;
        };
        if state.counts.len() != self.species_count {
            return f64::NAN;
        }
        let mut factor_u128 = 1_u128;
        let mut scaled_factor: Option<(f64, i64)> = None;
        let mut overflowed_inf = false;
        for &(species, req) in &ch.reactants {
            let avail = state.counts[species];
            if avail < u64::from(req) {
                return 0.0;
            }
            if overflowed_inf {
                continue;
            }
            match (scaled_factor, checked_binom_u128(avail, req)) {
                (None, Some(comb)) => {
                    if let Some(next) = factor_u128.checked_mul(comb) {
                        factor_u128 = next;
                    } else {
                        let lhs = normalize_pos_f64(factor_u128 as f64);
                        let rhs = normalize_pos_f64(comb as f64);
                        scaled_factor = Some(mul_scaled(lhs, rhs));
                    }
                }
                (None, None) => match binom_scaled(avail, req) {
                    Some(rhs) => {
                        let lhs = normalize_pos_f64(factor_u128 as f64);
                        scaled_factor = Some(mul_scaled(lhs, rhs));
                    }
                    None => overflowed_inf = true,
                },
                (Some(prev), Some(comb)) => {
                    scaled_factor = Some(mul_scaled(prev, normalize_pos_f64(comb as f64)));
                }
                (Some(prev), None) => match binom_scaled(avail, req) {
                    Some(rhs) => scaled_factor = Some(mul_scaled(prev, rhs)),
                    None => overflowed_inf = true,
                },
            }
        }
        if ch.rate_constant == 0.0 {
            return 0.0;
        }
        if overflowed_inf {
            return f64::INFINITY;
        }
        if let Some(scaled) = scaled_factor {
            let rate = normalize_pos_f64(ch.rate_constant);
            let total = mul_scaled(scaled, rate);
            return scale_exp2(total.0, total.1);
        }
        (factor_u128 as f64) * ch.rate_constant
    }
}

impl Model for ReactionNetwork {
    type State = ReactionState;
    type Event = ReactionEvent;
    type Error = ReactionError;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        for channel in 0..self.channels.len() {
            let rate = self.propensity(state, channel);
            if rate != 0.0 {
                emit(ReactionEvent(channel), rate);
            }
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        let _ = self.apply_reversible(state, event)?;
        Ok(())
    }
}

impl ReversibleModel for ReactionNetwork {
    type Undo = ReactionUndo;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        let channel = event.0;
        let ch = self
            .channels
            .get(channel)
            .ok_or(ReactionError::ChannelOutOfRange {
                channel,
                channel_count: self.channels.len(),
            })?;
        if state.counts.len() != self.species_count {
            return Err(ReactionError::StateSpeciesMismatch {
                expected: self.species_count,
                found: state.counts.len(),
            });
        }
        if state.fired_counts.len() != self.channels.len() {
            return Err(ReactionError::StateChannelMismatch {
                expected: self.channels.len(),
                found: state.fired_counts.len(),
            });
        }

        for &(species, req) in &ch.reactants {
            let available = state.counts[species];
            if available < u64::from(req) {
                return Err(ReactionError::InsufficientReactants {
                    channel,
                    species,
                    required: req,
                    available,
                });
            }
        }
        let next_fired = state.fired_counts[channel]
            .checked_add(1)
            .ok_or(ReactionError::FiredCountOverflow { channel })?;
        for &(species, delta) in &ch.net_delta {
            let checked = if delta >= 0 {
                state.counts[species].checked_add(delta as u64)
            } else {
                state.counts[species].checked_sub(delta.unsigned_abs())
            };
            if checked.is_none() {
                return Err(ReactionError::CountOverflow { channel, species });
            }
        }

        for &(species, delta) in &ch.net_delta {
            if delta >= 0 {
                state.counts[species] += delta as u64;
            } else {
                state.counts[species] -= delta.unsigned_abs();
            }
        }
        state.fired_counts[channel] = next_fired;
        Ok(ReactionUndo { channel })
    }

    fn undo(&self, state: &mut Self::State, undo: Self::Undo) {
        let Some(ch) = self.channels.get(undo.channel) else {
            return;
        };
        if state.counts.len() != self.species_count
            || state.fired_counts.len() != self.channels.len()
        {
            return;
        }
        for &(species, delta) in &ch.net_delta {
            if delta >= 0 {
                state.counts[species] -= delta as u64;
            } else {
                state.counts[species] += delta.unsigned_abs();
            }
        }
        state.fired_counts[undo.channel] -= 1;
    }
}

impl IncrementalModel for ReactionNetwork {
    fn slot_count(&self, _state: &Self::State) -> usize {
        self.channels.len()
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        self.propensity(state, slot)
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        ReactionEvent(slot)
    }

    fn affected_slots(&self, _state: &Self::State, event: &Self::Event, out: &mut Vec<usize>) {
        if let Some(deps) = self.affected_channels.get(event.0) {
            out.extend_from_slice(deps);
        }
    }
}

impl Checkpointable for ReactionNetwork {
    const MODEL_ID: &'static str = "okrates.reaction_network.v1";

    fn encode_params(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.species_count as u64).to_le_bytes());
        out.extend_from_slice(&(self.channels.len() as u64).to_le_bytes());
        for ch in &self.channels {
            out.extend_from_slice(&(ch.reactants.len() as u64).to_le_bytes());
            for &(sp, c) in &ch.reactants {
                out.extend_from_slice(&(sp as u64).to_le_bytes());
                out.extend_from_slice(&c.to_le_bytes());
            }
            out.extend_from_slice(&(ch.products.len() as u64).to_le_bytes());
            for &(sp, c) in &ch.products {
                out.extend_from_slice(&(sp as u64).to_le_bytes());
                out.extend_from_slice(&c.to_le_bytes());
            }
            out.extend_from_slice(&ch.rate_constant.to_bits().to_le_bytes());
            out.extend_from_slice(
                &ch.reverse_channel
                    .map_or(u64::MAX, |r| r as u64)
                    .to_le_bytes(),
            );
        }
        out.extend_from_slice(&(self.conservation_laws.len() as u64).to_le_bytes());
        for law in &self.conservation_laws {
            for &w in law {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        out.extend_from_slice(&(state.counts.len() as u64).to_le_bytes());
        out.extend_from_slice(&(state.fired_counts.len() as u64).to_le_bytes());
        for &c in &state.counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for &f in &state.fired_counts {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        let (chunks, remainder) = bytes.as_chunks::<8>();
        if !remainder.is_empty() {
            return Err(format!(
                "reaction state byte length {} is not a multiple of 8",
                bytes.len()
            )
            .into());
        }
        let total_words = self
            .species_count
            .checked_add(self.channels.len())
            .and_then(|n| n.checked_add(2))
            .ok_or("reaction state word count overflow")?;
        let expected_bytes = total_words
            .checked_mul(8)
            .ok_or("reaction state byte count overflow")?;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "reaction state must be {expected_bytes} bytes, got {}",
                bytes.len()
            )
            .into());
        }
        let mut words = chunks.iter().copied().map(u64::from_le_bytes);
        let counts_len = words
            .next()
            .ok_or("missing reaction counts length header")?;
        let fired_len = words.next().ok_or("missing reaction fired length header")?;
        if counts_len != self.species_count as u64 {
            return Err(format!(
                "reaction state counts length {counts_len} does not match expected {}",
                self.species_count
            )
            .into());
        }
        if fired_len != self.channels.len() as u64 {
            return Err(format!(
                "reaction state fired length {fired_len} does not match expected {}",
                self.channels.len()
            )
            .into());
        }
        let mut counts = Vec::with_capacity(self.species_count);
        for _ in 0..self.species_count {
            counts.push(words.next().ok_or("truncated reaction counts payload")?);
        }
        let mut fired_counts = Vec::with_capacity(self.channels.len());
        for _ in 0..self.channels.len() {
            fired_counts.push(words.next().ok_or("truncated reaction fired payload")?);
        }
        Ok(ReactionState {
            counts,
            fired_counts,
        })
    }
}
