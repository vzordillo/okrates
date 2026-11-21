//! Multi-species surface lattice-gas (adsorption, desorption, diffusion, and reaction) KMC template.
//!
//! Built on [`PeriodicLattice`], sites hold either a vacancy
//! (`None`) or one adsorbate species (`Some(s)`). Elementary steps comprise:
//! - single-site events ([`SingleSiteRule`]): adsorption, desorption, and unimolecular conversion;
//! - two-site nearest-neighbor events ([`PairRule`]): diffusion, associative desorption,
//!   dissociative adsorption, and bimolecular surface reactions.
//!
//! When [`LatticeGasEnergetics`] is configured, rate prefactors are modulated by the symmetric
//! Boltzmann factor `exp(-delta_E_local / (2 * k_B * T))`, where `delta_E_local` is the exact
//! configuration-energy change including on-site energies and nearest-neighbor lateral pair bonds,
//! preserving grand-canonical detailed balance across arbitrary periodic multiplicities.

use std::error::Error;
use std::fmt;

use okrates_core::{Checkpointable, IncrementalModel, Model, ReversibleModel, SumTree};

use crate::lattice_hopping::PeriodicLattice;
use crate::vacancy::BOLTZMANN_EV_PER_K;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LatticeGasError {
    ZeroSpecies,
    NoRules,
    TooManySlots(usize),
    SpeciesOutOfRange {
        species: usize,
        species_count: usize,
    },
    SiteOutOfRange {
        site: usize,
        site_count: usize,
    },
    BranchOutOfRange {
        site: usize,
        branch: usize,
        degree: usize,
    },
    SingleRuleOutOfRange {
        rule: usize,
        rule_count: usize,
    },
    PairRuleOutOfRange {
        rule: usize,
        rule_count: usize,
    },
    SiteTypeOutOfRange {
        site_type: usize,
        type_count: usize,
    },
    PathwayOutOfRange {
        pathway: usize,
        pathway_count: usize,
    },
    NoOpSingleRule {
        rule: usize,
    },
    NoOpPairRule {
        rule: usize,
    },
    InvalidRate {
        rate_hz: f64,
    },
    InvalidPressure {
        pressure_bar: f64,
    },
    InvalidTemperature(f64),
    InvalidSiteEnergies,
    InvalidPairInteractions,
    RulePreconditionMismatch,
    CounterOverflow,
    StateMismatch(&'static str),
}

impl fmt::Display for LatticeGasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSpecies => write!(f, "lattice gas requires at least one adsorbate species"),
            Self::NoRules => write!(
                f,
                "lattice gas requires at least one single-site or pair rule"
            ),
            Self::TooManySlots(slots) => write!(f, "{slots} slots exceed SumTree capacity"),
            Self::SpeciesOutOfRange {
                species,
                species_count,
            } => write!(
                f,
                "species index {species} out of range for {species_count} species"
            ),
            Self::SiteOutOfRange { site, site_count } => {
                write!(f, "site index {site} out of range for {site_count} sites")
            }
            Self::BranchOutOfRange {
                site,
                branch,
                degree,
            } => write!(
                f,
                "branch {branch} at site {site} out of range for degree {degree}"
            ),
            Self::SingleRuleOutOfRange { rule, rule_count } => write!(
                f,
                "single-site rule {rule} out of range for {rule_count} rules"
            ),
            Self::PairRuleOutOfRange { rule, rule_count } => {
                write!(f, "pair rule {rule} out of range for {rule_count} rules")
            }
            Self::SiteTypeOutOfRange {
                site_type,
                type_count,
            } => write!(
                f,
                "site type {site_type} out of range for {type_count} site types"
            ),
            Self::PathwayOutOfRange {
                pathway,
                pathway_count,
            } => write!(
                f,
                "pathway {pathway} out of range for {pathway_count} lattice pathways"
            ),
            Self::NoOpSingleRule { rule } => {
                write!(
                    f,
                    "single-site rule {rule} has identical from and to states"
                )
            }
            Self::NoOpPairRule { rule } => {
                write!(f, "pair rule {rule} has identical from and to states")
            }
            Self::InvalidRate { rate_hz } => {
                write!(f, "rate {rate_hz} Hz must be finite and non-negative")
            }
            Self::InvalidPressure { pressure_bar } => {
                write!(
                    f,
                    "pressure {pressure_bar} bar must be finite and non-negative"
                )
            }
            Self::InvalidTemperature(t) => {
                write!(f, "temperature {t} K must be finite and strictly positive")
            }
            Self::InvalidSiteEnergies => {
                write!(
                    f,
                    "site energies must be finite with length type_count * species_count"
                )
            }
            Self::InvalidPairInteractions => write!(
                f,
                "pair interactions must be finite and symmetric with length species_count^2"
            ),
            Self::RulePreconditionMismatch => {
                write!(f, "site state or type does not match rule precondition")
            }
            Self::CounterOverflow => write!(f, "lattice-gas event counter overflowed u64"),
            Self::StateMismatch(msg) => write!(f, "lattice-gas state invariant violated: {msg}"),
        }
    }
}

impl Error for LatticeGasError {}

fn normalize_rate(rate_hz: f64) -> Result<f64, LatticeGasError> {
    if !rate_hz.is_finite() || rate_hz < 0.0 {
        return Err(LatticeGasError::InvalidRate { rate_hz });
    }
    Ok(if rate_hz == 0.0 { 0.0 } else { rate_hz })
}

fn normalize_pressure_product(
    pressure_bar: f64,
    sticking_rate_per_bar_hz: f64,
) -> Result<f64, LatticeGasError> {
    if !pressure_bar.is_finite() || pressure_bar < 0.0 {
        return Err(LatticeGasError::InvalidPressure { pressure_bar });
    }
    let k = normalize_rate(sticking_rate_per_bar_hz)?;
    normalize_rate(pressure_bar * k)
}

/// Elementary single-site surface process (`from -> to`), optionally restricted to one `site_type`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingleSiteRule {
    pub from: Option<usize>,
    pub to: Option<usize>,
    pub site_type: Option<usize>,
    pub base_rate_hz: f64,
}

impl SingleSiteRule {
    /// Gas adsorption onto a vacant site (`None -> Some(species)`) at `pressure_bar * sticking_rate_per_bar_hz`.
    pub fn adsorption(
        species: usize,
        pressure_bar: f64,
        sticking_rate_per_bar_hz: f64,
    ) -> Result<Self, LatticeGasError> {
        Ok(Self {
            from: None,
            to: Some(species),
            site_type: None,
            base_rate_hz: normalize_pressure_product(pressure_bar, sticking_rate_per_bar_hz)?,
        })
    }

    /// Thermal desorption from a site (`Some(species) -> None`).
    pub fn desorption(species: usize, rate_hz: f64) -> Result<Self, LatticeGasError> {
        Ok(Self {
            from: Some(species),
            to: None,
            site_type: None,
            base_rate_hz: normalize_rate(rate_hz)?,
        })
    }

    /// Unimolecular on-site transformation (`Some(from) -> Some(to)`).
    pub fn conversion(from: usize, to: usize, rate_hz: f64) -> Result<Self, LatticeGasError> {
        if from == to {
            return Err(LatticeGasError::NoOpSingleRule { rule: 0 });
        }
        Ok(Self {
            from: Some(from),
            to: Some(to),
            site_type: None,
            base_rate_hz: normalize_rate(rate_hz)?,
        })
    }

    /// Restricts this rule to a specific lattice `site_type`.
    pub fn with_site_type(mut self, site_type: usize) -> Self {
        self.site_type = Some(site_type);
        self
    }
}

/// Elementary two-site nearest-neighbor process (`(site, neighbor): from -> to`),
/// optionally restricted to one lattice `pathway`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairRule {
    pub from: (Option<usize>, Option<usize>),
    pub to: (Option<usize>, Option<usize>),
    pub pathway: Option<usize>,
    pub base_rate_hz: f64,
}

impl PairRule {
    /// Adsorbate hop to an adjacent empty site (`(Some(species), None) -> (None, Some(species))`).
    pub fn diffusion(species: usize, rate_hz: f64) -> Result<Self, LatticeGasError> {
        Ok(Self {
            from: (Some(species), None),
            to: (None, Some(species)),
            pathway: None,
            base_rate_hz: normalize_rate(rate_hz)?,
        })
    }

    /// Bimolecular associative desorption (`(Some(s1), Some(s2)) -> (None, None)`).
    pub fn associative_desorption(
        s1: usize,
        s2: usize,
        rate_hz: f64,
    ) -> Result<Self, LatticeGasError> {
        Ok(Self {
            from: (Some(s1), Some(s2)),
            to: (None, None),
            pathway: None,
            base_rate_hz: normalize_rate(rate_hz)?,
        })
    }

    /// Dissociative adsorption onto an adjacent pair of vacant sites (`(None, None) -> (Some(s1), Some(s2))`).
    pub fn dissociative_adsorption(
        s1: usize,
        s2: usize,
        pressure_bar: f64,
        sticking_rate_per_bar_hz: f64,
    ) -> Result<Self, LatticeGasError> {
        Ok(Self {
            from: (None, None),
            to: (Some(s1), Some(s2)),
            pathway: None,
            base_rate_hz: normalize_pressure_product(pressure_bar, sticking_rate_per_bar_hz)?,
        })
    }

    /// Arbitrary two-site reaction (`from -> to`).
    pub fn reaction(
        from: (Option<usize>, Option<usize>),
        to: (Option<usize>, Option<usize>),
        rate_hz: f64,
    ) -> Result<Self, LatticeGasError> {
        if from == to {
            return Err(LatticeGasError::NoOpPairRule { rule: 0 });
        }
        Ok(Self {
            from,
            to,
            pathway: None,
            base_rate_hz: normalize_rate(rate_hz)?,
        })
    }

    /// Restricts this pair rule to a specific directed `pathway` on the lattice.
    pub fn with_pathway(mut self, pathway: usize) -> Self {
        self.pathway = Some(pathway);
        self
    }
}

/// Optional on-site energies and nearest-neighbor lateral interactions in electronvolts.
#[derive(Clone, Debug, PartialEq)]
pub struct LatticeGasEnergetics {
    pub temperature_k: f64,
    /// Row-major (`type_count * species_count`): binding/site energy for `(site_type, species)`.
    pub site_energies_ev: Vec<f64>,
    /// Symmetric (`species_count * species_count`): pairwise bond energy `eps(a, b) == eps(b, a)`.
    pub pair_interactions_ev: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct NormalizedEnergetics {
    temperature_k: f64,
    two_kbt_ev: f64,
    species_count: usize,
    site_energies_ev: Vec<f64>,
    pair_interactions_ev: Vec<f64>,
}

impl NormalizedEnergetics {
    #[inline]
    fn site_energy(&self, site_type: usize, species: Option<usize>) -> f64 {
        match species {
            Some(s) => self.site_energies_ev[site_type * self.species_count + s],
            None => 0.0,
        }
    }

    #[inline]
    fn pair_energy(&self, a: Option<usize>, b: Option<usize>) -> f64 {
        match (a, b) {
            (Some(sa), Some(sb)) => self.pair_interactions_ev[sa * self.species_count + sb],
            _ => 0.0,
        }
    }

    #[inline]
    fn modulate_rate(&self, base_rate_hz: f64, delta_e_ev: f64) -> f64 {
        if base_rate_hz == 0.0 || delta_e_ev == 0.0 {
            return base_rate_hz;
        }
        let exponent = -delta_e_ev / self.two_kbt_ev;
        let factor = exponent.exp();
        if factor >= f64::MIN_POSITIVE && factor.is_finite() {
            let scaled = base_rate_hz * factor;
            if scaled.is_finite() {
                return scaled;
            }
        }
        (base_rate_hz.ln() + exponent).exp()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LatticeGasEvent {
    SingleSite {
        site: usize,
        rule: usize,
    },
    Pair {
        site: usize,
        branch: usize,
        rule: usize,
    },
}

/// Single-use undo token returned by [`ReversibleModel::apply_reversible`](LatticeGas::apply_reversible).
///
/// Do not reuse or apply to a state other than the immediately preceding transitioned state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticeGasUndo {
    site: usize,
    target: Option<usize>,
    rule: usize,
    prev_from_occ: Option<usize>,
    prev_to_occ: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticeGasState {
    geometry_hash: u64,
    occupants: Vec<Option<usize>>,
    species_counts: Vec<usize>,
    single_rule_counts: Vec<u64>,
    pair_rule_counts: Vec<u64>,
}

impl LatticeGasState {
    pub fn site_count(&self) -> usize {
        self.occupants.len()
    }

    pub fn species_count(&self) -> usize {
        self.species_counts.len()
    }

    pub fn occupant(&self, site: usize) -> Option<Option<usize>> {
        self.occupants.get(site).copied()
    }

    pub fn occupants(&self) -> &[Option<usize>] {
        &self.occupants
    }

    pub fn species_count_on_surface(&self, species: usize) -> Option<usize> {
        self.species_counts.get(species).copied()
    }

    pub fn species_counts(&self) -> &[usize] {
        &self.species_counts
    }

    pub fn empty_site_count(&self) -> usize {
        let occupied: usize = self.species_counts.iter().copied().sum();
        self.occupants.len().saturating_sub(occupied)
    }

    pub fn coverage(&self, species: usize) -> Option<f64> {
        let count = self.species_counts.get(species)?;
        if self.occupants.is_empty() {
            return None;
        }
        Some(*count as f64 / self.occupants.len() as f64)
    }

    pub fn empty_fraction(&self) -> f64 {
        if self.occupants.is_empty() {
            return 0.0;
        }
        self.empty_site_count() as f64 / self.occupants.len() as f64
    }

    pub fn single_rule_count(&self, rule: usize) -> Option<u64> {
        self.single_rule_counts.get(rule).copied()
    }

    pub fn pair_rule_count(&self, rule: usize) -> Option<u64> {
        self.pair_rule_counts.get(rule).copied()
    }

    pub fn single_rule_counts(&self) -> &[u64] {
        &self.single_rule_counts
    }

    pub fn pair_rule_counts(&self) -> &[u64] {
        &self.pair_rule_counts
    }

    pub fn total_events(&self) -> u128 {
        let single_sum: u128 = self.single_rule_counts.iter().map(|&c| u128::from(c)).sum();
        let pair_sum: u128 = self.pair_rule_counts.iter().map(|&c| u128::from(c)).sum();
        single_sum + pair_sum
    }

    /// Computes turnover frequency per lattice site (`Hz`) over `elapsed_s` for the selected rule sets.
    ///
    /// Returns `None` if `elapsed_s` is non-positive or non-finite, the state has zero sites, or any rule
    /// index is out of bounds.
    pub fn turnover_frequency_per_site_hz(
        &self,
        elapsed_s: f64,
        single_rules: &[usize],
        pair_rules: &[usize],
    ) -> Option<f64> {
        if !elapsed_s.is_finite() || elapsed_s <= 0.0 || self.occupants.is_empty() {
            return None;
        }
        let mut total = 0_u128;
        for &r in single_rules {
            let count = self.single_rule_counts.get(r)?;
            total = total.checked_add(u128::from(*count))?;
        }
        for &r in pair_rules {
            let count = self.pair_rule_counts.get(r)?;
            total = total.checked_add(u128::from(*count))?;
        }
        Some((total as f64) / ((self.occupants.len() as f64) * elapsed_s))
    }
}

/// Multi-species lattice-gas KMC model supporting adsorption, desorption, diffusion,
/// lateral interactions, and surface catalytic reactions on an arbitrary [`PeriodicLattice`].
#[derive(Clone, Debug, PartialEq)]
pub struct LatticeGas {
    lattice: PeriodicLattice,
    species_count: usize,
    single_rules: Vec<SingleSiteRule>,
    pair_rules: Vec<PairRule>,
    energetics: Option<NormalizedEnergetics>,
    has_pair_interactions: bool,
    site_stride: usize,
}

fn validate_optional_species(
    species: Option<usize>,
    species_count: usize,
) -> Result<(), LatticeGasError> {
    if let Some(s) = species
        && s >= species_count
    {
        return Err(LatticeGasError::SpeciesOutOfRange {
            species: s,
            species_count,
        });
    }
    Ok(())
}

impl LatticeGas {
    pub fn new(
        lattice: PeriodicLattice,
        species_count: usize,
        mut single_rules: Vec<SingleSiteRule>,
        mut pair_rules: Vec<PairRule>,
        energetics: Option<LatticeGasEnergetics>,
    ) -> Result<Self, LatticeGasError> {
        if species_count == 0 {
            return Err(LatticeGasError::ZeroSpecies);
        }
        if usize::try_from(u64::MAX).is_ok_and(|max_u64| species_count >= max_u64) {
            return Err(LatticeGasError::SpeciesOutOfRange {
                species: species_count,
                species_count,
            });
        }
        if single_rules.is_empty() && pair_rules.is_empty() {
            return Err(LatticeGasError::NoRules);
        }

        let type_count = lattice.type_count();
        let pathway_count = lattice.pathway_count();

        for (idx, rule) in single_rules.iter_mut().enumerate() {
            validate_optional_species(rule.from, species_count)?;
            validate_optional_species(rule.to, species_count)?;
            if rule.from == rule.to {
                return Err(LatticeGasError::NoOpSingleRule { rule: idx });
            }
            if let Some(st) = rule.site_type
                && st >= type_count
            {
                return Err(LatticeGasError::SiteTypeOutOfRange {
                    site_type: st,
                    type_count,
                });
            }
            rule.base_rate_hz = normalize_rate(rule.base_rate_hz)?;
        }

        for (idx, rule) in pair_rules.iter_mut().enumerate() {
            validate_optional_species(rule.from.0, species_count)?;
            validate_optional_species(rule.from.1, species_count)?;
            validate_optional_species(rule.to.0, species_count)?;
            validate_optional_species(rule.to.1, species_count)?;
            if rule.from == rule.to {
                return Err(LatticeGasError::NoOpPairRule { rule: idx });
            }
            if let Some(pw) = rule.pathway
                && pw >= pathway_count
            {
                return Err(LatticeGasError::PathwayOutOfRange {
                    pathway: pw,
                    pathway_count,
                });
            }
            rule.base_rate_hz = normalize_rate(rule.base_rate_hz)?;
        }

        let (norm_ener, has_pair_interactions) = match energetics {
            None => (None, false),
            Some(e) => {
                if !e.temperature_k.is_finite() || e.temperature_k <= 0.0 {
                    return Err(LatticeGasError::InvalidTemperature(e.temperature_k));
                }
                let two_kbt_ev = 2.0 * BOLTZMANN_EV_PER_K * e.temperature_k;
                if !two_kbt_ev.is_finite() || two_kbt_ev <= 0.0 {
                    return Err(LatticeGasError::InvalidTemperature(e.temperature_k));
                }
                let expected_site_len = type_count
                    .checked_mul(species_count)
                    .ok_or(LatticeGasError::InvalidSiteEnergies)?;
                if e.site_energies_ev.len() != expected_site_len
                    || e.site_energies_ev.iter().any(|v| !v.is_finite())
                {
                    return Err(LatticeGasError::InvalidSiteEnergies);
                }
                let expected_pair_len = species_count
                    .checked_mul(species_count)
                    .ok_or(LatticeGasError::InvalidPairInteractions)?;
                if e.pair_interactions_ev.len() != expected_pair_len
                    || e.pair_interactions_ev.iter().any(|v| !v.is_finite())
                {
                    return Err(LatticeGasError::InvalidPairInteractions);
                }
                for a in 0..species_count {
                    for b in (a + 1)..species_count {
                        if e.pair_interactions_ev[a * species_count + b]
                            != e.pair_interactions_ev[b * species_count + a]
                        {
                            return Err(LatticeGasError::InvalidPairInteractions);
                        }
                    }
                }
                let site_energies_ev: Vec<f64> = e
                    .site_energies_ev
                    .into_iter()
                    .map(|v| if v == 0.0 { 0.0 } else { v })
                    .collect();
                let pair_interactions_ev: Vec<f64> = e
                    .pair_interactions_ev
                    .into_iter()
                    .map(|v| if v == 0.0 { 0.0 } else { v })
                    .collect();
                let has_pairs = pair_interactions_ev.iter().any(|&v| v != 0.0);
                (
                    Some(NormalizedEnergetics {
                        temperature_k: e.temperature_k,
                        two_kbt_ev,
                        species_count,
                        site_energies_ev,
                        pair_interactions_ev,
                    }),
                    has_pairs,
                )
            }
        };

        let site_stride = lattice
            .max_degree()
            .checked_mul(pair_rules.len())
            .and_then(|p| p.checked_add(single_rules.len()))
            .ok_or(LatticeGasError::TooManySlots(usize::MAX))?;
        let total_slots = lattice
            .site_count()
            .checked_mul(site_stride)
            .ok_or(LatticeGasError::TooManySlots(usize::MAX))?;
        SumTree::check_len(total_slots).map_err(|_| LatticeGasError::TooManySlots(total_slots))?;

        Ok(Self {
            lattice,
            species_count,
            single_rules,
            pair_rules,
            energetics: norm_ener,
            has_pair_interactions,
            site_stride,
        })
    }

    pub fn lattice(&self) -> &PeriodicLattice {
        &self.lattice
    }

    pub fn species_count(&self) -> usize {
        self.species_count
    }

    pub fn single_rules(&self) -> &[SingleSiteRule] {
        &self.single_rules
    }

    pub fn pair_rules(&self) -> &[PairRule] {
        &self.pair_rules
    }

    pub fn temperature_k(&self) -> Option<f64> {
        self.energetics.as_ref().map(|e| e.temperature_k)
    }

    pub fn has_pair_interactions(&self) -> bool {
        self.has_pair_interactions
    }

    pub fn empty_state(&self) -> LatticeGasState {
        LatticeGasState {
            geometry_hash: self.lattice.geometry_hash(),
            occupants: vec![None; self.lattice.site_count()],
            species_counts: vec![0; self.species_count],
            single_rule_counts: vec![0; self.single_rules.len()],
            pair_rule_counts: vec![0; self.pair_rules.len()],
        }
    }

    pub fn state_from_occupants(
        &self,
        occupants: &[Option<usize>],
    ) -> Result<LatticeGasState, LatticeGasError> {
        let site_count = self.lattice.site_count();
        if occupants.len() != site_count {
            return Err(LatticeGasError::StateMismatch(
                "occupants length disagrees with lattice site_count",
            ));
        }
        let mut species_counts = vec![0; self.species_count];
        for &occ in occupants {
            if let Some(s) = occ {
                let Some(slot) = species_counts.get_mut(s) else {
                    return Err(LatticeGasError::SpeciesOutOfRange {
                        species: s,
                        species_count: self.species_count,
                    });
                };
                *slot += 1;
            }
        }
        Ok(LatticeGasState {
            geometry_hash: self.lattice.geometry_hash(),
            occupants: occupants.to_vec(),
            species_counts,
            single_rule_counts: vec![0; self.single_rules.len()],
            pair_rule_counts: vec![0; self.pair_rules.len()],
        })
    }

    #[inline]
    fn check_fast_compatibility(&self, state: &LatticeGasState) -> Result<(), LatticeGasError> {
        if state.geometry_hash != self.lattice.geometry_hash() {
            return Err(LatticeGasError::StateMismatch(
                "state lattice geometry disagrees with model lattice",
            ));
        }
        if state.occupants.len() != self.lattice.site_count() {
            return Err(LatticeGasError::StateMismatch(
                "occupants length disagrees with lattice site_count",
            ));
        }
        if state.species_counts.len() != self.species_count {
            return Err(LatticeGasError::StateMismatch(
                "species_counts length disagrees with model species_count",
            ));
        }
        if state.single_rule_counts.len() != self.single_rules.len()
            || state.pair_rule_counts.len() != self.pair_rules.len()
        {
            return Err(LatticeGasError::StateMismatch(
                "rule firing counters disagree with model rule counts",
            ));
        }
        Ok(())
    }

    pub fn check_invariants(&self, state: &LatticeGasState) -> Result<(), LatticeGasError> {
        self.check_fast_compatibility(state)?;
        let mut recomputed = vec![0_usize; self.species_count];
        for &occ in &state.occupants {
            if let Some(s) = occ {
                let Some(cnt) = recomputed.get_mut(s) else {
                    return Err(LatticeGasError::SpeciesOutOfRange {
                        species: s,
                        species_count: self.species_count,
                    });
                };
                *cnt += 1;
            }
        }
        if recomputed != state.species_counts {
            return Err(LatticeGasError::StateMismatch(
                "species_counts disagree with occupants histogram",
            ));
        }
        Ok(())
    }

    /// Evaluates the full configuration energy `E(sigma)` in eV (zero if no energetics are set).
    pub fn total_energy_ev(&self, state: &LatticeGasState) -> Result<f64, LatticeGasError> {
        self.check_invariants(state)?;
        let Some(ener) = &self.energetics else {
            return Ok(0.0);
        };
        let mut total = 0.0_f64;
        for (site, &occ) in state.occupants.iter().enumerate() {
            let st = self.lattice.site_type(site).unwrap_or(0);
            total += ener.site_energy(st, occ);
            if self.has_pair_interactions {
                let mut pair_acc = 0.0_f64;
                for hop in self.lattice.hops(site) {
                    pair_acc += ener.pair_energy(occ, state.occupants[hop.target]);
                }
                total += 0.5 * pair_acc;
            }
        }
        Ok(total)
    }

    #[inline]
    fn local_delta_e_single(
        &self,
        ener: &NormalizedEnergetics,
        state: &LatticeGasState,
        site: usize,
        from: Option<usize>,
        to: Option<usize>,
    ) -> f64 {
        let st = self.lattice.site_type(site).unwrap_or(0);
        let mut delta = ener.site_energy(st, to) - ener.site_energy(st, from);
        if self.has_pair_interactions {
            for hop in self.lattice.hops(site) {
                let nb_occ = state.occupants[hop.target];
                delta += ener.pair_energy(to, nb_occ) - ener.pair_energy(from, nb_occ);
            }
        }
        delta
    }

    #[inline]
    fn local_delta_e_pair(
        &self,
        ener: &NormalizedEnergetics,
        state: &LatticeGasState,
        site: usize,
        target: usize,
        from: (Option<usize>, Option<usize>),
        to: (Option<usize>, Option<usize>),
    ) -> f64 {
        let t_site = self.lattice.site_type(site).unwrap_or(0);
        let t_target = self.lattice.site_type(target).unwrap_or(0);
        let mut delta = (ener.site_energy(t_site, to.0) - ener.site_energy(t_site, from.0))
            + (ener.site_energy(t_target, to.1) - ener.site_energy(t_target, from.1));
        if self.has_pair_interactions {
            let mut direct_bonds = 0_usize;
            for hop in self.lattice.hops(site) {
                if hop.target == target {
                    direct_bonds += 1;
                } else {
                    let nb_occ = state.occupants[hop.target];
                    delta += ener.pair_energy(to.0, nb_occ) - ener.pair_energy(from.0, nb_occ);
                }
            }
            for hop in self.lattice.hops(target) {
                if hop.target != site {
                    let nb_occ = state.occupants[hop.target];
                    delta += ener.pair_energy(to.1, nb_occ) - ener.pair_energy(from.1, nb_occ);
                }
            }
            delta += (direct_bonds as f64)
                * (ener.pair_energy(to.0, to.1) - ener.pair_energy(from.0, from.1));
        }
        delta
    }

    #[inline]
    fn single_rate_at(&self, state: &LatticeGasState, site: usize, rule_idx: usize) -> f64 {
        let Some(rule) = self.single_rules.get(rule_idx) else {
            return f64::NAN;
        };
        if let Some(required_type) = rule.site_type
            && self.lattice.site_type(site) != Some(required_type)
        {
            return 0.0;
        }
        if state.occupants[site] != rule.from {
            return 0.0;
        }
        if rule.base_rate_hz == 0.0 {
            return 0.0;
        }
        let Some(ener) = &self.energetics else {
            return rule.base_rate_hz;
        };
        let delta_e = self.local_delta_e_single(ener, state, site, rule.from, rule.to);
        ener.modulate_rate(rule.base_rate_hz, delta_e)
    }

    #[inline]
    fn pair_rate_at(
        &self,
        state: &LatticeGasState,
        site: usize,
        branch: usize,
        rule_idx: usize,
    ) -> f64 {
        let Some(hop) = self.lattice.hops(site).get(branch) else {
            return 0.0;
        };
        let Some(rule) = self.pair_rules.get(rule_idx) else {
            return f64::NAN;
        };
        if let Some(required_pw) = rule.pathway
            && hop.pathway != required_pw
        {
            return 0.0;
        }
        if (state.occupants[site], state.occupants[hop.target]) != rule.from {
            return 0.0;
        }
        if rule.base_rate_hz == 0.0 {
            return 0.0;
        }
        let Some(ener) = &self.energetics else {
            return rule.base_rate_hz;
        };
        let delta_e = self.local_delta_e_pair(ener, state, site, hop.target, rule.from, rule.to);
        ener.modulate_rate(rule.base_rate_hz, delta_e)
    }

    pub fn evaluate_single_rate(
        &self,
        state: &LatticeGasState,
        site: usize,
        rule: usize,
    ) -> Option<f64> {
        self.check_fast_compatibility(state).ok()?;
        if site >= self.lattice.site_count() || rule >= self.single_rules.len() {
            return None;
        }
        Some(self.single_rate_at(state, site, rule))
    }

    pub fn evaluate_pair_rate(
        &self,
        state: &LatticeGasState,
        site: usize,
        branch: usize,
        rule: usize,
    ) -> Option<f64> {
        self.check_fast_compatibility(state).ok()?;
        if site >= self.lattice.site_count()
            || branch >= self.lattice.degree(site)
            || rule >= self.pair_rules.len()
        {
            return None;
        }
        Some(self.pair_rate_at(state, site, branch, rule))
    }

    /// Computes exact non-interacting competitive Langmuir equilibrium coverages `theta_s` at `site_type`.
    ///
    /// Returns `None` outside the pure Langmuir domain: when `site_type` is out of range, lateral pair
    /// interactions are non-zero, any non-adsorption/desorption conversion or pair-reaction rule exists,
    /// multi-type lattices combine pair diffusion across types, or any adsorbing species has zero total
    /// desorption rate.
    pub fn langmuir_equilibrium_coverage(&self, site_type: usize) -> Option<Vec<f64>> {
        if site_type >= self.lattice.type_count() || self.has_pair_interactions {
            return None;
        }
        for r in &self.pair_rules {
            match (r.from, r.to) {
                ((Some(s1), None), (None, Some(s2))) if s1 == s2 => {}
                _ => return None,
            }
        }
        if !self.pair_rules.is_empty() && self.lattice.type_count() > 1 {
            return None;
        }
        let mut ads_sum = vec![0.0_f64; self.species_count];
        let mut des_sum = vec![0.0_f64; self.species_count];
        for r in &self.single_rules {
            if r.site_type.is_some_and(|st| st != site_type) {
                continue;
            }
            match (r.from, r.to) {
                (None, Some(s)) => {
                    let eff = match &self.energetics {
                        None => r.base_rate_hz,
                        Some(ener) => {
                            let de = ener.site_energy(site_type, Some(s));
                            ener.modulate_rate(r.base_rate_hz, de)
                        }
                    };
                    ads_sum[s] += eff;
                }
                (Some(s), None) => {
                    let eff = match &self.energetics {
                        None => r.base_rate_hz,
                        Some(ener) => {
                            let de = -ener.site_energy(site_type, Some(s));
                            ener.modulate_rate(r.base_rate_hz, de)
                        }
                    };
                    des_sum[s] += eff;
                }
                _ => return None,
            }
        }

        let mut ks = vec![0.0_f64; self.species_count];
        let mut denom = 1.0_f64;
        for s in 0..self.species_count {
            if ads_sum[s] > 0.0 {
                if des_sum[s] <= 0.0 || !des_sum[s].is_finite() {
                    return None;
                }
                let k = ads_sum[s] / des_sum[s];
                if !k.is_finite() {
                    return None;
                }
                ks[s] = k;
                denom += k;
            }
        }
        if !denom.is_finite() || denom <= 0.0 {
            return None;
        }
        Some(ks.into_iter().map(|k| k / denom).collect())
    }

    #[inline]
    fn push_site_and_incoming_slots(&self, v: usize, out: &mut Vec<usize>) {
        let base = v * self.site_stride;
        out.extend(base..(base + self.site_stride));
        if !self.pair_rules.is_empty() {
            let single_len = self.single_rules.len();
            let pair_len = self.pair_rules.len();
            for hop in self.lattice.hops(v) {
                let nb_base =
                    hop.target * self.site_stride + single_len + hop.reverse_branch * pair_len;
                out.extend(nb_base..(nb_base + pair_len));
            }
        }
    }
}

fn adjust_species_count(
    species_counts: &mut [usize],
    from: Option<usize>,
    to: Option<usize>,
) -> Result<(), LatticeGasError> {
    if from == to {
        return Ok(());
    }
    if let Some(f) = from {
        let Some(cnt) = species_counts.get_mut(f) else {
            return Err(LatticeGasError::StateMismatch(
                "occupant species out of range",
            ));
        };
        *cnt = cnt
            .checked_sub(1)
            .ok_or(LatticeGasError::StateMismatch("species counter underflow"))?;
    }
    if let Some(t) = to {
        let Some(cnt) = species_counts.get_mut(t) else {
            return Err(LatticeGasError::StateMismatch(
                "target species out of range",
            ));
        };
        *cnt = cnt
            .checked_add(1)
            .ok_or(LatticeGasError::StateMismatch("species counter overflow"))?;
    }
    Ok(())
}

impl Model for LatticeGas {
    type State = LatticeGasState;
    type Event = LatticeGasEvent;
    type Error = LatticeGasError;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        if self.check_fast_compatibility(state).is_err()
            || state
                .occupants
                .iter()
                .any(|occ| occ.is_some_and(|s| s >= self.species_count))
        {
            emit(LatticeGasEvent::SingleSite { site: 0, rule: 0 }, f64::NAN);
            return;
        }
        let site_count = self.lattice.site_count();
        let single_len = self.single_rules.len();
        let pair_len = self.pair_rules.len();
        for site in 0..site_count {
            for rule in 0..single_len {
                let rate = self.single_rate_at(state, site, rule);
                if rate != 0.0 {
                    emit(LatticeGasEvent::SingleSite { site, rule }, rate);
                }
            }
            if pair_len > 0 {
                let deg = self.lattice.degree(site);
                for branch in 0..deg {
                    for rule in 0..pair_len {
                        let rate = self.pair_rate_at(state, site, branch, rule);
                        if rate != 0.0 {
                            emit(LatticeGasEvent::Pair { site, branch, rule }, rate);
                        }
                    }
                }
            }
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        let _ = self.apply_reversible(state, event)?;
        Ok(())
    }
}

impl ReversibleModel for LatticeGas {
    type Undo = LatticeGasUndo;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        self.check_fast_compatibility(state)?;
        let site_count = self.lattice.site_count();
        match *event {
            LatticeGasEvent::SingleSite { site, rule } => {
                if site >= site_count {
                    return Err(LatticeGasError::SiteOutOfRange { site, site_count });
                }
                let Some(rule_def) = self.single_rules.get(rule) else {
                    return Err(LatticeGasError::SingleRuleOutOfRange {
                        rule,
                        rule_count: self.single_rules.len(),
                    });
                };
                if let Some(req_st) = rule_def.site_type
                    && self.lattice.site_type(site) != Some(req_st)
                {
                    return Err(LatticeGasError::RulePreconditionMismatch);
                }
                let prev_from_occ = state.occupants[site];
                if prev_from_occ != rule_def.from {
                    return Err(LatticeGasError::RulePreconditionMismatch);
                }
                let next_count = state.single_rule_counts[rule]
                    .checked_add(1)
                    .ok_or(LatticeGasError::CounterOverflow)?;
                adjust_species_count(&mut state.species_counts, rule_def.from, rule_def.to)?;
                state.occupants[site] = rule_def.to;
                state.single_rule_counts[rule] = next_count;
                Ok(LatticeGasUndo {
                    site,
                    target: None,
                    rule,
                    prev_from_occ,
                    prev_to_occ: None,
                })
            }
            LatticeGasEvent::Pair { site, branch, rule } => {
                if site >= site_count {
                    return Err(LatticeGasError::SiteOutOfRange { site, site_count });
                }
                let hops = self.lattice.hops(site);
                let Some(hop) = hops.get(branch) else {
                    return Err(LatticeGasError::BranchOutOfRange {
                        site,
                        branch,
                        degree: hops.len(),
                    });
                };
                let Some(rule_def) = self.pair_rules.get(rule) else {
                    return Err(LatticeGasError::PairRuleOutOfRange {
                        rule,
                        rule_count: self.pair_rules.len(),
                    });
                };
                if let Some(req_pw) = rule_def.pathway
                    && hop.pathway != req_pw
                {
                    return Err(LatticeGasError::RulePreconditionMismatch);
                }
                let target = hop.target;
                let prev_from_occ = state.occupants[site];
                let prev_to_occ = state.occupants[target];
                if (prev_from_occ, prev_to_occ) != rule_def.from {
                    return Err(LatticeGasError::RulePreconditionMismatch);
                }
                let next_count = state.pair_rule_counts[rule]
                    .checked_add(1)
                    .ok_or(LatticeGasError::CounterOverflow)?;
                adjust_species_count(&mut state.species_counts, rule_def.from.0, rule_def.to.0)?;
                if let Err(err) =
                    adjust_species_count(&mut state.species_counts, rule_def.from.1, rule_def.to.1)
                {
                    // Restore first species count change on hypothetical invariant corruption.
                    let _ = adjust_species_count(
                        &mut state.species_counts,
                        rule_def.to.0,
                        rule_def.from.0,
                    );
                    return Err(err);
                }
                state.occupants[site] = rule_def.to.0;
                state.occupants[target] = rule_def.to.1;
                state.pair_rule_counts[rule] = next_count;
                Ok(LatticeGasUndo {
                    site,
                    target: Some(target),
                    rule,
                    prev_from_occ,
                    prev_to_occ,
                })
            }
        }
    }

    fn undo(&self, state: &mut Self::State, undo: Self::Undo) {
        if self.check_fast_compatibility(state).is_err() || undo.site >= state.occupants.len() {
            return;
        }
        let cur_from = state.occupants[undo.site];
        if let Some(target) = undo.target {
            if target >= state.occupants.len() || undo.rule >= state.pair_rule_counts.len() {
                return;
            }
            let cur_to = state.occupants[target];
            let _ = adjust_species_count(&mut state.species_counts, cur_from, undo.prev_from_occ);
            let _ = adjust_species_count(&mut state.species_counts, cur_to, undo.prev_to_occ);
            state.occupants[undo.site] = undo.prev_from_occ;
            state.occupants[target] = undo.prev_to_occ;
            state.pair_rule_counts[undo.rule] = state.pair_rule_counts[undo.rule].saturating_sub(1);
        } else {
            if undo.rule >= state.single_rule_counts.len() {
                return;
            }
            let _ = adjust_species_count(&mut state.species_counts, cur_from, undo.prev_from_occ);
            state.occupants[undo.site] = undo.prev_from_occ;
            state.single_rule_counts[undo.rule] =
                state.single_rule_counts[undo.rule].saturating_sub(1);
        }
    }
}

impl IncrementalModel for LatticeGas {
    fn slot_count(&self, state: &Self::State) -> usize {
        if self.check_fast_compatibility(state).is_err() {
            return 1;
        }
        self.lattice.site_count() * self.site_stride
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        if self.check_fast_compatibility(state).is_err() || self.site_stride == 0 {
            return f64::NAN;
        }
        let site = slot / self.site_stride;
        let local = slot % self.site_stride;
        if site >= self.lattice.site_count() {
            return f64::NAN;
        }
        let occ = state.occupants[site];
        if occ.is_some_and(|s| s >= self.species_count) {
            return f64::NAN;
        }
        let single_len = self.single_rules.len();
        if local < single_len {
            return self.single_rate_at(state, site, local);
        }
        let pair_len = self.pair_rules.len();
        if pair_len == 0 {
            return f64::NAN;
        }
        let pair_slot = local - single_len;
        let branch = pair_slot / pair_len;
        let rule = pair_slot % pair_len;
        self.pair_rate_at(state, site, branch, rule)
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        if self.site_stride == 0 {
            return LatticeGasEvent::SingleSite {
                site: usize::MAX,
                rule: usize::MAX,
            };
        }
        let site = slot / self.site_stride;
        let local = slot % self.site_stride;
        let single_len = self.single_rules.len();
        if local < single_len {
            return LatticeGasEvent::SingleSite { site, rule: local };
        }
        let pair_len = self.pair_rules.len();
        if pair_len == 0 {
            return LatticeGasEvent::SingleSite {
                site: usize::MAX,
                rule: usize::MAX,
            };
        }
        let pair_slot = local - single_len;
        LatticeGasEvent::Pair {
            site,
            branch: pair_slot / pair_len,
            rule: pair_slot % pair_len,
        }
    }

    fn affected_slots(&self, state: &Self::State, event: &Self::Event, out: &mut Vec<usize>) {
        if self.site_stride == 0 {
            return;
        }
        let site_count = self.lattice.site_count();
        let (s1, s2_opt) = match *event {
            LatticeGasEvent::SingleSite { site, .. } if site < site_count => (site, None),
            LatticeGasEvent::Pair { site, branch, .. } if site < site_count => {
                if let Some(hop) = self.lattice.hops(site).get(branch) {
                    (site, Some(hop.target))
                } else {
                    out.extend(0..self.slot_count(state));
                    return;
                }
            }
            _ => {
                out.extend(0..self.slot_count(state));
                return;
            }
        };

        if !self.has_pair_interactions {
            self.push_site_and_incoming_slots(s1, out);
            if let Some(s2) = s2_opt {
                self.push_site_and_incoming_slots(s2, out);
            }
        } else {
            self.push_site_and_incoming_slots(s1, out);
            for hop in self.lattice.hops(s1) {
                self.push_site_and_incoming_slots(hop.target, out);
            }
            if let Some(s2) = s2_opt {
                self.push_site_and_incoming_slots(s2, out);
                for hop in self.lattice.hops(s2) {
                    self.push_site_and_incoming_slots(hop.target, out);
                }
            }
        }

        out.sort_unstable();
        out.dedup();
    }
}

fn encode_opt_usize(v: Option<usize>, out: &mut Vec<u8>) {
    out.extend_from_slice(&v.map_or(u64::MAX, |x| x as u64).to_le_bytes());
}

impl Checkpointable for LatticeGas {
    const MODEL_ID: &'static str = "okrates.lattice_gas.v1";

    fn encode_params(&self, out: &mut Vec<u8>) {
        self.lattice.encode_geometry(out);
        out.extend_from_slice(&(self.species_count as u64).to_le_bytes());
        out.extend_from_slice(&(self.single_rules.len() as u64).to_le_bytes());
        for r in &self.single_rules {
            encode_opt_usize(r.from, out);
            encode_opt_usize(r.to, out);
            encode_opt_usize(r.site_type, out);
            out.extend_from_slice(&r.base_rate_hz.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&(self.pair_rules.len() as u64).to_le_bytes());
        for r in &self.pair_rules {
            encode_opt_usize(r.from.0, out);
            encode_opt_usize(r.from.1, out);
            encode_opt_usize(r.to.0, out);
            encode_opt_usize(r.to.1, out);
            encode_opt_usize(r.pathway, out);
            out.extend_from_slice(&r.base_rate_hz.to_bits().to_le_bytes());
        }
        match &self.energetics {
            None => out.push(0),
            Some(e) => {
                out.push(1);
                out.extend_from_slice(&e.temperature_k.to_bits().to_le_bytes());
                for &v in &e.site_energies_ev {
                    out.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                for &v in &e.pair_interactions_ev {
                    out.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
        }
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        out.extend_from_slice(&state.geometry_hash.to_le_bytes());
        out.extend_from_slice(&(state.occupants.len() as u64).to_le_bytes());
        out.extend_from_slice(&(state.species_counts.len() as u64).to_le_bytes());
        out.extend_from_slice(&(state.single_rule_counts.len() as u64).to_le_bytes());
        out.extend_from_slice(&(state.pair_rule_counts.len() as u64).to_le_bytes());
        for &occ in &state.occupants {
            encode_opt_usize(occ, out);
        }
        for &cnt in &state.single_rule_counts {
            out.extend_from_slice(&cnt.to_le_bytes());
        }
        for &cnt in &state.pair_rule_counts {
            out.extend_from_slice(&cnt.to_le_bytes());
        }
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        let (chunks, remainder) = bytes.as_chunks::<8>();
        if !remainder.is_empty() {
            return Err(format!(
                "lattice-gas state byte length {} is not a multiple of 8",
                bytes.len()
            )
            .into());
        }
        let mut words = chunks.iter().copied();
        let geom_hash = words
            .next()
            .map(u64::from_le_bytes)
            .ok_or("missing lattice-gas geometry hash word")?;
        if geom_hash != self.lattice.geometry_hash() {
            return Err("lattice geometry hash differs from model lattice".into());
        }
        let site_count = self.lattice.site_count();
        let single_len = self.single_rules.len();
        let pair_len = self.pair_rules.len();
        let raw_sites = words
            .next()
            .map(u64::from_le_bytes)
            .ok_or("missing site count word")?;
        let raw_species = words
            .next()
            .map(u64::from_le_bytes)
            .ok_or("missing species count word")?;
        let raw_single = words
            .next()
            .map(u64::from_le_bytes)
            .ok_or("missing single rule count word")?;
        let raw_pair = words
            .next()
            .map(u64::from_le_bytes)
            .ok_or("missing pair rule count word")?;
        if raw_sites != site_count as u64
            || raw_species != self.species_count as u64
            || raw_single != single_len as u64
            || raw_pair != pair_len as u64
        {
            return Err("lattice-gas state header disagrees with model dimensions".into());
        }
        let expected_words = 5_usize
            .checked_add(site_count)
            .and_then(|w| w.checked_add(single_len))
            .and_then(|w| w.checked_add(pair_len))
            .ok_or("lattice-gas state size overflow")?;
        if chunks.len() != expected_words {
            return Err(format!(
                "lattice-gas state must be {} bytes, got {}",
                expected_words * 8,
                bytes.len()
            )
            .into());
        }

        let mut occupants = Vec::with_capacity(site_count);
        for _ in 0..site_count {
            let raw = u64::from_le_bytes(words.next().ok_or("truncated occupant word")?);
            if raw == u64::MAX {
                occupants.push(None);
            } else {
                let s = usize::try_from(raw)
                    .ok()
                    .filter(|&v| v < self.species_count)
                    .ok_or(LatticeGasError::SpeciesOutOfRange {
                        species: usize::try_from(raw).unwrap_or(usize::MAX),
                        species_count: self.species_count,
                    })?;
                occupants.push(Some(s));
            }
        }
        let mut single_rule_counts = Vec::with_capacity(single_len);
        for _ in 0..single_len {
            single_rule_counts.push(u64::from_le_bytes(
                words.next().ok_or("truncated single-rule count word")?,
            ));
        }
        let mut pair_rule_counts = Vec::with_capacity(pair_len);
        for _ in 0..pair_len {
            pair_rule_counts.push(u64::from_le_bytes(
                words.next().ok_or("truncated pair-rule count word")?,
            ));
        }

        let mut state = self.state_from_occupants(&occupants)?;
        state.single_rule_counts = single_rule_counts;
        state.pair_rule_counts = pair_rule_counts;
        self.check_invariants(&state)?;
        Ok(state)
    }
}
