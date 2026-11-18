//! Fe–Cr radiation-induced segregation model: lattice atoms, ⟨110⟩ dumbbells and vacancies.
//!
//! Part 1 holds parameters, state, geometry, energy, barriers and rates. The lattice
//! bonds and the vacancy saddle interactions come from [`Martinez2012`]. Dumbbell
//! parameters follow Senninger et al., Acta Mater. 103, 1 (2016), as refitted in R1
//! ([`DumbbellParams::senninger2016_r1`]). Positions are in half-lattice units.
//!
//! All rates use local energy differences, never differences of total energies, so
//! detailed balance holds to rounding. The barrier is `max(raw, max(0, ΔE))`; the
//! floor acts on the saddle level shared by the forward and the reverse jump.

use std::cell::RefCell;
use std::collections::{BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::sync::LazyLock;

use okrates_core::{Checkpointable, IncrementalModel, KmcRng, Model};
use rand_chacha::ChaCha12Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};

use crate::bcc::{BccLattice, DISPLACEMENTS, LatticeError, SECOND_DISPLACEMENTS};
use crate::fecr::{Energetics, FeCrError, Species};
use crate::fecr_martinez::Martinez2012;
use crate::vacancy::BOLTZMANN_EV_PER_K;

/// The six ⟨110⟩ dumbbell axes (first non-zero component +1), in half-lattice units.
pub const AXES: [[i64; 3]; 6] = [
    [1, 1, 0],
    [1, -1, 0],
    [1, 0, 1],
    [1, 0, -1],
    [0, 1, 1],
    [0, 1, -1],
];

/// Offsets of shells 1–4 (|o|² in {3, 4, 8, 11}); an I–V pair within these recombines.
pub static RC_OFFSETS: LazyLock<Vec<[i64; 3]>> = LazyLock::new(|| {
    let mut out = Vec::new();
    for x in -3..=3_i64 {
        for y in -3..=3_i64 {
            for z in -3..=3_i64 {
                let same_parity = (x - y) % 2 == 0 && (y - z) % 2 == 0;
                if same_parity && matches!(x * x + y * y + z * z, 3 | 4 | 8 | 11) {
                    out.push([x, y, z]);
                }
            }
        }
    }
    out
});

const NO_RECORD: u32 = u32::MAX;

fn dot(a: [i64; 3], b: [i64; 3]) -> i64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Parameters of the dumbbell energetics, in eV (and s⁻¹, k_B for the rates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DumbbellParams {
    /// Dumbbell formation terms `[FeFe, FeCr, CrCr]`.
    pub g: [f64; 3],
    /// Interaction with a neighbour Fe or Cr: `e[D][t][X]`, D = FeFe, mixed, CrCr;
    /// t = C (CFe for mixed), T, CCr; X = Fe, Cr. Entries `e[0][2]` and `e[2][2]` are 0.
    pub e: [[[f64; 2]; 3]; 3],
    /// Saddle interactions `saddle[jumper][neighbour]`, Fe = 0, Cr = 1.
    pub saddle: [[f64; 2]; 2],
    /// Attempt frequency in s⁻¹.
    pub nu0: f64,
    /// Migration entropy in k_B.
    pub ds_mig_kb: f64,
}

impl DumbbellParams {
    /// The R1 minimax fit to Senninger et al. 2016, copied from `tests/data/fecr_ris_r1.txt`.
    pub fn senninger2016_r1() -> Self {
        Self {
            g: [-8.880000000000827, -8.737909004145983, -6.719999999999268],
            e: [
                [
                    [0.0, -0.002796825340728226],
                    [0.0, 0.04948139886618801],
                    [0.0, 0.0],
                ],
                [
                    [0.0, -0.1012139813746966],
                    [0.0, 0.047380904503169854],
                    [0.0, 0.24878601862530336],
                ],
                [
                    [-0.2637297804701352, 0.0],
                    [-0.05242727878889055, 0.0],
                    [0.0, 0.0],
                ],
            ],
            saddle: [
                [-2.040227272727425, -1.9351747398188035],
                [-2.0362158982455605, -1.4629351311350782],
            ],
            nu0: 1e13,
            ds_mig_kb: 2.1,
        }
    }

    fn validate(&self) -> Result<(), FeCrRisError> {
        let finite = self.g.iter().all(|v| v.is_finite())
            && self.e.iter().flatten().flatten().all(|v| v.is_finite())
            && self.saddle.iter().flatten().all(|v| v.is_finite())
            && self.ds_mig_kb.is_finite();
        if !finite {
            return Err(FeCrRisError::Parameter("dumbbell energies must be finite"));
        }
        if self.e[0][2] != [0.0; 2] || self.e[2][2] != [0.0; 2] {
            return Err(FeCrRisError::Parameter(
                "e[FeFe][CCr] and e[CrCr][CCr] must be zero",
            ));
        }
        if !self.nu0.is_finite() || self.nu0 <= 0.0 {
            return Err(FeCrRisError::Parameter("nu0 must be finite and positive"));
        }
        Ok(())
    }
}

/// Parameters of a [`FeCrRis`] model.
#[derive(Clone, Debug, PartialEq)]
pub struct FeCrRisParams {
    /// Conventional cells per axis (each at least 6).
    pub cells: [usize; 3],
    /// x positions (half units, in `[0, 2·cells[0])`) of the sink planes.
    pub sink_planes: Vec<i64>,
    /// Temperature in K (finite, at least 0).
    pub temperature_k: f64,
    /// Frenkel-pair creation rate per site in s⁻¹ (finite, at least 0).
    pub k0_dpa_per_s: f64,
    /// Number of defect records P (at least 1).
    pub capacity: usize,
    /// Dumbbell energetics.
    pub dumbbell: DumbbellParams,
}

/// The content of a lattice site in a [`FeCrRisState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    /// An iron atom.
    Fe,
    /// A chromium atom.
    Cr,
    /// A vacancy.
    Vacancy,
    /// A dumbbell (two atoms on one site).
    Dumbbell,
}

/// A defect record slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Record {
    /// An unused slot.
    Empty,
    /// A vacancy.
    Vacancy {
        /// Its site.
        site: usize,
    },
    /// A dumbbell.
    Dumbbell {
        /// Its site.
        site: usize,
        /// Index into [`AXES`].
        axis: usize,
        /// Species at the +u end.
        plus: Species,
        /// Species at the −u end.
        minus: Species,
    },
}

/// A defect to place in a state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Defect {
    /// A vacancy.
    Vacancy {
        /// Its site.
        site: usize,
    },
    /// A dumbbell.
    Dumbbell {
        /// Its site.
        site: usize,
        /// Index into [`AXES`].
        axis: usize,
        /// Species at the +u end (Fe or Cr).
        plus: Species,
        /// Species at the −u end (Fe or Cr).
        minus: Species,
    },
}

impl Defect {
    fn site(self) -> usize {
        match self {
            Self::Vacancy { site } | Self::Dumbbell { site, .. } => site,
        }
    }
}

impl From<Defect> for Record {
    fn from(d: Defect) -> Self {
        match d {
            Defect::Vacancy { site } => Self::Vacancy { site },
            Defect::Dumbbell {
                site,
                axis,
                plus,
                minus,
            } => Self::Dumbbell {
                site,
                axis,
                plus,
                minus,
            },
        }
    }
}

/// An error from building or checking a [`FeCrRis`] model or state.
#[derive(Clone, Debug, PartialEq)]
pub enum FeCrRisError {
    /// The lattice could not be built.
    Lattice(LatticeError),
    /// The lattice energetics could not be built.
    Energetics(FeCrError),
    /// A cell count is below 6.
    CellsTooSmall {
        /// The rejected cell counts.
        cells: [usize; 3],
    },
    /// A parameter is invalid.
    Parameter(&'static str),
    /// A sink plane lies outside `[0, 2·cells[0])`.
    SinkPlane(i64),
    /// The occupancy length differs from the site count.
    SiteCount {
        /// Expected site count.
        expected: usize,
        /// Length found.
        found: usize,
    },
    /// The occupancy contains a vacancy.
    VacancyInOccupancy {
        /// The site.
        site: usize,
    },
    /// More Cr atoms were requested than there are sites.
    TooManyAtoms,
    /// More defects were given than records.
    TooManyDefects {
        /// Record capacity P.
        capacity: usize,
        /// Number of defects given.
        defects: usize,
    },
    /// Two defects share a site.
    DuplicateDefect {
        /// The site.
        site: usize,
    },
    /// A defect site is outside the lattice.
    DefectOutOfRange {
        /// The site.
        site: usize,
    },
    /// A defect lies on a sink plane.
    DefectOnSink {
        /// The site.
        site: usize,
    },
    /// A dumbbell has an axis of 6 or more, or a vacancy end.
    InvalidDefect {
        /// The site.
        site: usize,
    },
    /// A vacancy and a dumbbell are within the recombination radius.
    WithinRecombinationRadius {
        /// The vacancy site.
        vacancy: usize,
        /// The dumbbell site.
        dumbbell: usize,
    },
    /// The state is internally inconsistent.
    Inconsistent(&'static str),
    /// Fewer than two empty records remain for a Frenkel pair.
    PoolExhausted {
        /// Record capacity P.
        capacity: usize,
        /// Records in use.
        defects: usize,
    },
    /// No acceptable Frenkel chain was found.
    CreationFailed,
    /// The event names no possible jump: bad index, wrong record kind, empty record or blocked target.
    NoSuchJump {
        /// Record index.
        rec: usize,
        /// Jump or direction index.
        jump: usize,
    },
}

impl fmt::Display for FeCrRisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lattice(e) => write!(f, "lattice: {e}"),
            Self::Energetics(e) => write!(f, "energetics: {e}"),
            Self::CellsTooSmall { cells } => {
                write!(f, "each cell count must be at least 6, got {cells:?}")
            }
            Self::Parameter(m) => write!(f, "invalid parameter: {m}"),
            Self::SinkPlane(x) => write!(f, "sink plane x = {x} is outside the box"),
            Self::SiteCount { expected, found } => {
                write!(f, "expected {expected} sites, found {found}")
            }
            Self::VacancyInOccupancy { site } => {
                write!(f, "occupancy has a vacancy on site {site}")
            }
            Self::TooManyAtoms => write!(f, "more Cr atoms than sites"),
            Self::TooManyDefects { capacity, defects } => {
                write!(f, "{defects} defects exceed the capacity {capacity}")
            }
            Self::DuplicateDefect { site } => write!(f, "two defects on site {site}"),
            Self::DefectOutOfRange { site } => write!(f, "defect site {site} is out of range"),
            Self::DefectOnSink { site } => write!(f, "defect on sink plane at site {site}"),
            Self::InvalidDefect { site } => write!(f, "invalid defect on site {site}"),
            Self::WithinRecombinationRadius { vacancy, dumbbell } => write!(
                f,
                "vacancy {vacancy} and dumbbell {dumbbell} are within the recombination radius"
            ),
            Self::Inconsistent(m) => write!(f, "inconsistent state: {m}"),
            Self::PoolExhausted { capacity, defects } => write!(
                f,
                "defect pool exhausted: capacity {capacity}, {defects} in use"
            ),
            Self::CreationFailed => write!(f, "no acceptable Frenkel chain after 1000 attempts"),
            Self::NoSuchJump { rec, jump } => {
                write!(f, "record {rec} has no possible jump {jump}")
            }
        }
    }
}

impl Error for FeCrRisError {}

/// Builds the creation stream: ChaCha12 seeded by `run_seed` (LE, zero padded), on `stream`.
fn creation_rng(run_seed: u64, stream: u64) -> ChaCha12Rng {
    let mut bytes = [0_u8; 32];
    bytes[..8].copy_from_slice(&run_seed.to_le_bytes());
    let mut rng = ChaCha12Rng::from_seed(bytes);
    rng.set_stream(stream);
    rng
}

/// A uniform value strictly inside (0, 1), as in `okrates_core::KmcRng::open01`.
fn open01(rng: &mut ChaCha12Rng) -> f64 {
    ((rng.next_u64() >> 12) as f64 + 0.5) * f64::EPSILON
}

const N_CACHE_SIZE: usize = 1024;

/// The dynamic state: site contents, defect records, counters and the creation stream.
#[derive(Clone, Debug)]
pub struct FeCrRisState {
    cells: Vec<Cell>,
    records: Vec<Record>,
    site_record: Vec<u32>,
    reservoir: VecDeque<Species>,
    borrowed_fe: u64,
    created: u64,
    recombined: u64,
    absorbed_v: u64,
    absorbed_i: u64,
    creation_attempts: u64,
    floored_dumbbell_jumps: u64,
    floored_vacancy_jumps: u64,
    creation_rng: ChaCha12Rng,
    peak_defects: u64,
    last_changed: Vec<usize>,
    last_records: Vec<usize>,
    /// Memo of `n(site)` for the current cells; cleared whenever `apply` runs.
    n_cache: RefCell<Vec<(usize, u32)>>,
}

impl FeCrRisState {
    /// Site contents.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }
    /// The P defect records.
    pub fn records(&self) -> &[Record] {
        &self.records
    }
    /// The record index of the defect on `site`, if any.
    pub fn record_at(&self, site: usize) -> Option<usize> {
        match self.site_record[site] {
            NO_RECORD => None,
            r => Some(r as usize),
        }
    }
    /// Atoms absorbed by sinks and not yet returned, oldest first.
    pub fn reservoir(&self) -> &VecDeque<Species> {
        &self.reservoir
    }
    /// Fe atoms created from nothing because the reservoir was empty.
    pub fn borrowed_fe(&self) -> u64 {
        self.borrowed_fe
    }
    /// Frenkel pairs created.
    pub fn created(&self) -> u64 {
        self.created
    }
    /// I–V recombinations.
    pub fn recombined(&self) -> u64 {
        self.recombined
    }
    /// Vacancies absorbed at sinks.
    pub fn absorbed_v(&self) -> u64 {
        self.absorbed_v
    }
    /// Dumbbells absorbed at sinks.
    pub fn absorbed_i(&self) -> u64 {
        self.absorbed_i
    }
    /// Creation draws, including accepted ones.
    pub fn creation_attempts(&self) -> u64 {
        self.creation_attempts
    }
    /// Dumbbell jumps whose barrier was raised by the floor.
    pub fn floored_dumbbell_jumps(&self) -> u64 {
        self.floored_dumbbell_jumps
    }
    /// Vacancy jumps whose barrier was raised by the floor.
    pub fn floored_vacancy_jumps(&self) -> u64 {
        self.floored_vacancy_jumps
    }
    /// The largest number of defect records in use so far.
    pub fn peak_defects(&self) -> u64 {
        self.peak_defects
    }
    /// The word position of the creation stream.
    pub fn creation_word_pos(&self) -> u128 {
        self.creation_rng.get_word_pos()
    }
    /// Sites changed by the last apply (empty until an apply has run).
    pub fn last_changed(&self) -> &[usize] {
        &self.last_changed
    }
    /// Records created, moved or freed by the last apply (empty until an apply has run).
    pub fn last_records(&self) -> &[usize] {
        &self.last_records
    }
    /// Atom counts `[lattice Fe, lattice Cr, dumbbell Fe, dumbbell Cr, reservoir Fe, reservoir Cr]`.
    pub fn atom_counts(&self) -> [u64; 6] {
        let mut c = [0_u64; 6];
        for cell in &self.cells {
            match cell {
                Cell::Fe => c[0] += 1,
                Cell::Cr => c[1] += 1,
                _ => {}
            }
        }
        for r in &self.records {
            if let Record::Dumbbell { plus, minus, .. } = r {
                for sp in [plus, minus] {
                    c[2 + sp.index()] += 1;
                }
            }
        }
        for sp in &self.reservoir {
            c[4 + sp.index()] += 1;
        }
        c
    }
    /// The numbers of vacancies and dumbbells.
    pub fn defect_counts(&self) -> (usize, usize) {
        let v = self
            .records
            .iter()
            .filter(|r| matches!(r, Record::Vacancy { .. }))
            .count();
        let d = self
            .records
            .iter()
            .filter(|r| matches!(r, Record::Dumbbell { .. }))
            .count();
        (v, d)
    }

    fn content(&self, site: usize) -> Content {
        match self.cells[site] {
            Cell::Fe => Content::Fe,
            Cell::Cr => Content::Cr,
            Cell::Vacancy => Content::Vacancy,
            Cell::Dumbbell => match self.records[self.site_record[site] as usize] {
                Record::Dumbbell {
                    axis, plus, minus, ..
                } => Content::Dumbbell { axis, plus, minus },
                _ => unreachable!("dumbbell cell without a dumbbell record"),
            },
        }
    }

    /// Cr weight of the content of `site`, without building a [`Content`].
    fn cr_weight_at(&self, site: usize) -> u32 {
        match self.cells[site] {
            Cell::Cr => 1,
            Cell::Dumbbell => match self.records[self.site_record[site] as usize] {
                Record::Dumbbell { plus, minus, .. } => {
                    u32::from(plus == Species::Cr) + u32::from(minus == Species::Cr)
                }
                _ => unreachable!("dumbbell cell without a dumbbell record"),
            },
            _ => 0,
        }
    }

    /// Σ Cr weights over {site} ∪ 1NN ∪ 2NN.
    fn n(&self, lattice: &BccLattice, site: usize) -> u32 {
        let mut n = self.cr_weight_at(site);
        for &t in lattice.neighbors(site) {
            n += self.cr_weight_at(t);
        }
        for &t in lattice.second_neighbors(site) {
            n += self.cr_weight_at(t);
        }
        n
    }

    /// [`Self::n`] memoised until the next `apply`.
    fn n_cached(&self, lattice: &BccLattice, site: usize) -> u32 {
        let mut cache = self.n_cache.borrow_mut();
        if cache.is_empty() {
            cache.resize(N_CACHE_SIZE, (usize::MAX, 0));
        }
        let entry = &mut cache[site % N_CACHE_SIZE];
        if entry.0 != site {
            *entry = (site, self.n(lattice, site));
        }
        entry.1
    }

    fn clear_n_cache(&mut self) {
        self.n_cache.get_mut().fill((usize::MAX, 0));
    }

    fn open_records(&self) -> u64 {
        self.records
            .iter()
            .filter(|r| !matches!(r, Record::Empty))
            .count() as u64
    }
}

/// A site's content as seen by the energy routines; `Empty` is a saddle end.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Content {
    Fe,
    Cr,
    Vacancy,
    Dumbbell {
        axis: usize,
        plus: Species,
        minus: Species,
    },
    Empty,
}

impl Content {
    fn of(species: Species) -> Self {
        match species {
            Species::Fe => Self::Fe,
            Species::Cr => Self::Cr,
            Species::Vacancy => Self::Vacancy,
        }
    }

    /// The species for bond purposes; dumbbells and empty sites have no bonds.
    fn bond_species(self) -> Option<Species> {
        match self {
            Self::Fe => Some(Species::Fe),
            Self::Cr => Some(Species::Cr),
            Self::Vacancy => Some(Species::Vacancy),
            Self::Dumbbell { .. } | Self::Empty => None,
        }
    }

    fn cr_weight(self) -> u32 {
        match self {
            Self::Cr => 1,
            Self::Dumbbell { plus, minus, .. } => {
                u32::from(plus == Species::Cr) + u32::from(minus == Species::Cr)
            }
            _ => 0,
        }
    }
}

fn is_mixed(a: Content, b: Content) -> bool {
    matches!(
        (a, b),
        (Content::Fe, Content::Cr) | (Content::Cr, Content::Fe)
    )
}

/// The state with some sites overridden.
struct View<'a> {
    state: &'a FeCrRisState,
    overrides: &'a [(usize, Content)],
}

impl View<'_> {
    fn get(&self, site: usize) -> Content {
        for &(s, c) in self.overrides {
            if s == site {
                return c;
            }
        }
        self.state.content(site)
    }

    /// Σ Cr weights over {site} ∪ 1NN ∪ 2NN.
    fn n(&self, lattice: &BccLattice, site: usize) -> u32 {
        let mut n = self.get(site).cr_weight();
        for &t in lattice.neighbors(site) {
            n += self.get(t).cr_weight();
        }
        for &t in lattice.second_neighbors(site) {
            n += self.get(t).cr_weight();
        }
        n
    }
}

/// Dumbbell type D: 0 FeFe, 1 mixed, 2 CrCr.
fn dumbbell_type(plus: Species, minus: Species) -> usize {
    match (plus, minus) {
        (Species::Fe, Species::Fe) => 0,
        (Species::Cr, Species::Cr) => 2,
        _ => 1,
    }
}

/// Site type t for the neighbour offset `d` of a dumbbell: 0 C (CFe), 1 T, 2 CCr.
fn site_type(axis: usize, plus: Species, minus: Species, d: [i64; 3]) -> usize {
    let along = dot(d, AXES[axis]);
    if along == 0 {
        return 1;
    }
    if plus == minus {
        return 0;
    }
    let facing = if along > 0 { plus } else { minus };
    if facing == Species::Fe { 0 } else { 2 }
}

/// A jump, resolved: the sites, species and the new defect.
#[derive(Clone, Copy, Debug)]
enum Move {
    Dumbbell {
        /// Old dumbbell site.
        o: usize,
        /// Target site.
        k: usize,
        /// New dumbbell on `k`.
        axis: usize,
        plus: Species,
        minus: Species,
        /// The jumper.
        jumper: Species,
        /// The atom left on `o`.
        rest: Species,
        /// The former atom on `k`.
        atom: Species,
    },
    Vacancy {
        q: usize,
        p: usize,
        atom: Species,
    },
}

impl Move {
    fn ends(&self) -> (usize, usize) {
        match *self {
            Self::Dumbbell { o, k, .. } => (o, k),
            Self::Vacancy { q, p, .. } => (q, p),
        }
    }

    fn final_overrides(&self) -> [(usize, Content); 2] {
        match *self {
            Self::Dumbbell {
                o,
                k,
                axis,
                plus,
                minus,
                rest,
                ..
            } => [
                (o, Content::of(rest)),
                (k, Content::Dumbbell { axis, plus, minus }),
            ],
            Self::Vacancy { q, p, atom } => [(q, Content::of(atom)), (p, Content::Vacancy)],
        }
    }

    fn saddle_overrides(&self) -> [(usize, Content); 2] {
        let (a, b) = self.ends();
        [(a, Content::Empty), (b, Content::Empty)]
    }
}

/// The Fe–Cr RIS model.
#[derive(Clone, Debug)]
pub struct FeCrRis {
    params: FeCrRisParams,
    lattice: BccLattice,
    energetics: Martinez2012,
    kt: f64,
    /// Saddle interactions refitted at T (r2-spec §2.5); `params` keeps the fitted values.
    saddle: [[f64; 2]; 2],
}

impl FeCrRis {
    /// Builds the model, validating the parameters.
    pub fn new(params: FeCrRisParams) -> Result<Self, FeCrRisError> {
        if params.cells.iter().any(|&c| c < 6) {
            return Err(FeCrRisError::CellsTooSmall {
                cells: params.cells,
            });
        }
        if !params.temperature_k.is_finite() || params.temperature_k < 0.0 {
            return Err(FeCrRisError::Parameter(
                "temperature must be finite and at least 0",
            ));
        }
        if !params.k0_dpa_per_s.is_finite() || params.k0_dpa_per_s < 0.0 {
            return Err(FeCrRisError::Parameter("K0 must be finite and at least 0"));
        }
        if params.capacity == 0 {
            return Err(FeCrRisError::Parameter("capacity must be at least 1"));
        }
        let width = 2 * params.cells[0] as i64;
        if let Some(&x) = params.sink_planes.iter().find(|&&x| x < 0 || x >= width) {
            return Err(FeCrRisError::SinkPlane(x));
        }
        params.dumbbell.validate()?;
        let lattice = BccLattice::new(params.cells).map_err(FeCrRisError::Lattice)?;
        let energetics =
            Martinez2012::new(params.temperature_k).map_err(FeCrRisError::Energetics)?;
        let kt = BOLTZMANN_EV_PER_K * params.temperature_k;
        let saddle = if params.temperature_k > 0.0 {
            Self::refit_saddle(&params)?
        } else {
            params.dumbbell.saddle
        };
        Ok(Self {
            params,
            lattice,
            energetics,
            kt,
            saddle,
        })
    }

    /// A 6×6×6 model without sinks at `temperature_k` with saddle table `saddle`.
    fn probe(
        params: &FeCrRisParams,
        temperature_k: f64,
        saddle: [[f64; 2]; 2],
    ) -> Result<Self, FeCrRisError> {
        Ok(Self {
            params: FeCrRisParams {
                cells: [6, 6, 6],
                sink_planes: Vec::new(),
                capacity: 1,
                temperature_k,
                ..params.clone()
            },
            lattice: BccLattice::new([6, 6, 6]).map_err(FeCrRisError::Lattice)?,
            energetics: Martinez2012::new(temperature_k).map_err(FeCrRisError::Energetics)?,
            kt: BOLTZMANN_EV_PER_K * temperature_k,
            saddle,
        })
    }

    /// Raw barriers of the four defining pure-Fe jumps of r2-spec §2.5, in the order
    /// FeFe dumbbell / Fe, mixed / Cr, mixed / Fe, CrCr dumbbell / Cr.
    fn defining_barriers(&self) -> Result<[f64; 4], FeCrRisError> {
        use Species::{Cr, Fe};
        let mut out = [0.0; 4];
        for (i, (plus, minus, mover)) in [(Fe, Fe, Fe), (Cr, Fe, Cr), (Cr, Fe, Fe), (Cr, Cr, Cr)]
            .into_iter()
            .enumerate()
        {
            let occ = vec![Fe; self.lattice.site_count()];
            let defect = Defect::Dumbbell {
                site: 0,
                axis: 0,
                plus,
                minus,
            };
            let s = self.state_with_defects(occ, &[defect], 0)?;
            let mv = (0..8)
                .filter_map(|j| self.jump_move(&s, 0, j))
                .find(|mv| matches!(*mv, Move::Dumbbell { jumper, .. } if jumper == mover))
                .expect("a pure-Fe dumbbell has jumps for both of its atoms");
            out[i] = self.raw_of(&s, &mv);
        }
        Ok(out)
    }

    /// Refits the four saddle interactions at T so that the defining barriers keep their
    /// 0 K values (r2-spec §2.5): FeFe and CrFe from jumps 1 and 2 (8 terms each), then
    /// FeCr and CrCr from jumps 3 and 4 (1 term each).
    fn refit_saddle(params: &FeCrRisParams) -> Result<[[f64; 2]; 2], FeCrRisError> {
        let fitted = params.dumbbell.saddle;
        let target = Self::probe(params, 0.0, fitted)?.defining_barriers()?;
        let mut probe = Self::probe(params, params.temperature_k, fitted)?;
        let b = probe.defining_barriers()?;
        probe.saddle[0][0] -= (b[0] - target[0]) / 8.0;
        probe.saddle[1][0] -= (b[1] - target[1]) / 8.0;
        let b = probe.defining_barriers()?;
        probe.saddle[0][1] -= b[2] - target[2];
        probe.saddle[1][1] -= b[3] - target[3];
        if !probe.saddle.iter().flatten().all(|v| v.is_finite()) {
            return Err(FeCrRisError::Parameter(
                "the refitted dumbbell saddle interactions are not finite",
            ));
        }
        Ok(probe.saddle)
    }

    /// The saddle interactions `[jumper][neighbour]` in use at the model temperature.
    pub fn dumbbell_saddle(&self) -> [[f64; 2]; 2] {
        self.saddle
    }

    /// The parameters.
    pub fn params(&self) -> &FeCrRisParams {
        &self.params
    }

    /// The lattice.
    pub fn lattice(&self) -> &BccLattice {
        &self.lattice
    }

    /// Adds `shift` (eV) to all four dumbbell saddle interactions.
    ///
    /// Exists only to force the barrier floor in tests.
    #[doc(hidden)]
    pub fn with_dumbbell_saddle_shift(mut self, shift: f64) -> Self {
        for table in [&mut self.params.dumbbell.saddle, &mut self.saddle] {
            for g in table.iter_mut().flatten() {
                *g += shift;
            }
        }
        self
    }

    fn on_sink(&self, site: usize) -> bool {
        self.params
            .sink_planes
            .contains(&self.lattice.position(site)[0])
    }

    fn new_state(
        &self,
        cells: Vec<Cell>,
        records: Vec<Record>,
        site_record: Vec<u32>,
        run_seed: u64,
    ) -> FeCrRisState {
        let mut state = FeCrRisState {
            cells,
            records,
            site_record,
            reservoir: VecDeque::new(),
            borrowed_fe: 0,
            created: 0,
            recombined: 0,
            absorbed_v: 0,
            absorbed_i: 0,
            creation_attempts: 0,
            floored_dumbbell_jumps: 0,
            floored_vacancy_jumps: 0,
            creation_rng: creation_rng(run_seed, 1),
            peak_defects: 0,
            last_changed: Vec::new(),
            last_records: Vec::new(),
            n_cache: RefCell::new(Vec::new()),
        };
        state.peak_defects = state.open_records();
        state
    }

    /// A reproducible state of Fe with `cr_count` Cr atoms on shuffled sites and no defects.
    ///
    /// `seed` drives the shuffle; `run_seed` seeds the creation stream.
    pub fn random_state(
        &self,
        cr_count: usize,
        seed: u64,
        run_seed: u64,
    ) -> Result<FeCrRisState, FeCrRisError> {
        let n = self.lattice.site_count();
        if cr_count > n {
            return Err(FeCrRisError::TooManyAtoms);
        }
        let mut rng = KmcRng::from_seed(seed);
        let mut order: Vec<usize> = (0..n).collect();
        for i in 0..cr_count {
            let j = i + ((rng.open01() * (n - i) as f64) as usize).min(n - i - 1);
            order.swap(i, j);
        }
        let mut occ = vec![Species::Fe; n];
        for &site in &order[..cr_count] {
            occ[site] = Species::Cr;
        }
        self.state_with_defects(occ, &[], run_seed)
    }

    /// A state from an Fe/Cr occupancy, with `defects` overwriting their sites.
    ///
    /// Records 0.. take the defects in order. Rejects vacancies in `occ`, too many,
    /// duplicate, out-of-range or sink-plane defects, invalid dumbbells, and any
    /// vacancy–dumbbell pair within the recombination radius.
    pub fn state_with_defects(
        &self,
        occ: Vec<Species>,
        defects: &[Defect],
        run_seed: u64,
    ) -> Result<FeCrRisState, FeCrRisError> {
        let n = self.lattice.site_count();
        if occ.len() != n {
            return Err(FeCrRisError::SiteCount {
                expected: n,
                found: occ.len(),
            });
        }
        if defects.len() > self.params.capacity {
            return Err(FeCrRisError::TooManyDefects {
                capacity: self.params.capacity,
                defects: defects.len(),
            });
        }
        let mut cells = Vec::with_capacity(n);
        for (site, sp) in occ.iter().enumerate() {
            cells.push(match sp {
                Species::Fe => Cell::Fe,
                Species::Cr => Cell::Cr,
                Species::Vacancy => return Err(FeCrRisError::VacancyInOccupancy { site }),
            });
        }
        let mut records = vec![Record::Empty; self.params.capacity];
        let mut site_record = vec![NO_RECORD; n];
        for (i, &d) in defects.iter().enumerate() {
            let site = d.site();
            if site >= n {
                return Err(FeCrRisError::DefectOutOfRange { site });
            }
            if site_record[site] != NO_RECORD {
                return Err(FeCrRisError::DuplicateDefect { site });
            }
            if self.on_sink(site) {
                return Err(FeCrRisError::DefectOnSink { site });
            }
            cells[site] = match d {
                Defect::Vacancy { .. } => Cell::Vacancy,
                Defect::Dumbbell {
                    axis, plus, minus, ..
                } => {
                    if axis >= 6 || plus == Species::Vacancy || minus == Species::Vacancy {
                        return Err(FeCrRisError::InvalidDefect { site });
                    }
                    Cell::Dumbbell
                }
            };
            records[i] = d.into();
            site_record[site] = i as u32;
        }
        let state = self.new_state(cells, records, site_record, run_seed);
        if let Some((vacancy, dumbbell)) = self.recombination_pair(&state) {
            return Err(FeCrRisError::WithinRecombinationRadius { vacancy, dumbbell });
        }
        Ok(state)
    }

    /// The first vacancy–dumbbell pair (vacancy site, dumbbell site) within [`RC_OFFSETS`].
    fn recombination_pair(&self, s: &FeCrRisState) -> Option<(usize, usize)> {
        for r in &s.records {
            if let Record::Dumbbell { site, .. } = *r {
                let pos = self.lattice.position(site);
                for o in RC_OFFSETS.iter() {
                    let other = self
                        .lattice
                        .site_at([pos[0] + o[0], pos[1] + o[1], pos[2] + o[2]])
                        .expect("same-parity offsets stay on the lattice");
                    if s.cells[other] == Cell::Vacancy {
                        return Some((other, site));
                    }
                }
            }
        }
        None
    }

    /// Checks sizes, the record/cell correspondence, sink planes and the recombination invariant.
    pub fn check_invariants(&self, s: &FeCrRisState) -> Result<(), FeCrRisError> {
        let n = self.lattice.site_count();
        if s.cells.len() != n || s.site_record.len() != n {
            return Err(FeCrRisError::SiteCount {
                expected: n,
                found: s.cells.len(),
            });
        }
        if s.records.len() != self.params.capacity {
            return Err(FeCrRisError::Inconsistent("record count differs from P"));
        }
        let defect_cells = s
            .cells
            .iter()
            .filter(|c| matches!(c, Cell::Vacancy | Cell::Dumbbell))
            .count();
        if defect_cells as u64 != s.open_records() {
            return Err(FeCrRisError::Inconsistent(
                "cells and records differ in number",
            ));
        }
        for (i, r) in s.records.iter().enumerate() {
            let (site, cell) = match *r {
                Record::Empty => continue,
                Record::Vacancy { site } => (site, Cell::Vacancy),
                Record::Dumbbell {
                    site,
                    axis,
                    plus,
                    minus,
                } => {
                    if axis >= 6 || plus == Species::Vacancy || minus == Species::Vacancy {
                        return Err(FeCrRisError::InvalidDefect { site });
                    }
                    (site, Cell::Dumbbell)
                }
            };
            if site >= n || s.cells[site] != cell || s.site_record[site] != i as u32 {
                return Err(FeCrRisError::Inconsistent("record does not match its site"));
            }
            if self.on_sink(site) {
                return Err(FeCrRisError::DefectOnSink { site });
            }
        }
        if s.site_record.iter().filter(|&&r| r != NO_RECORD).count() as u64 != s.open_records() {
            return Err(FeCrRisError::Inconsistent(
                "site map and records differ in number",
            ));
        }
        if let Some((vacancy, dumbbell)) = self.recombination_pair(s) {
            return Err(FeCrRisError::WithinRecombinationRadius { vacancy, dumbbell });
        }
        Ok(())
    }

    fn bond(&self, a: Content, b: Content, n_sum: u32) -> f64 {
        self.energetics
            .bond_energy(a.bond_species(), b.bond_species(), n_sum)
    }

    /// g[D] + Σ e over the Fe/Cr 1NN sites of a dumbbell.
    fn dumbbell_term(
        &self,
        view: &View,
        site: usize,
        axis: usize,
        plus: Species,
        minus: Species,
    ) -> f64 {
        let p = &self.params.dumbbell;
        let d_type = dumbbell_type(plus, minus);
        let mut term = p.g[d_type];
        for (i, d) in DISPLACEMENTS.iter().enumerate() {
            let x = match view.get(self.lattice.neighbor(site, i)) {
                Content::Fe => 0,
                Content::Cr => 1,
                _ => continue,
            };
            term += p.e[d_type][site_type(axis, plus, minus, *d)][x];
        }
        term
    }

    /// The energy of the whole state in eV, computed from scratch (for tests).
    pub fn total_energy(&self, s: &FeCrRisState) -> f64 {
        let view = View {
            state: s,
            overrides: &[],
        };
        let n_sites = self.lattice.site_count();
        let n: Vec<u32> = (0..n_sites).map(|i| view.n(&self.lattice, i)).collect();
        let mut sum = 0.0;
        for site in 0..n_sites {
            let a = view.get(site);
            for (shell, weight) in [
                (&self.lattice.neighbors(site)[..], 1.0),
                (&self.lattice.second_neighbors(site)[..], 0.5),
            ] {
                for &t in shell {
                    if t > site {
                        sum += weight * self.bond(a, view.get(t), n[site] + n[t]);
                    }
                }
            }
        }
        for r in &s.records {
            if let Record::Dumbbell {
                site,
                axis,
                plus,
                minus,
            } = *r
            {
                sum += self.dumbbell_term(&view, site, axis, plus, minus);
            }
        }
        sum
    }

    /// E_after − E_before in eV for placing `overrides` on the state, from local terms only.
    fn local_difference(&self, s: &FeCrRisState, overrides: &[(usize, Content)]) -> f64 {
        let lattice = &self.lattice;
        let before = View {
            state: s,
            overrides: &[],
        };
        let after = View {
            state: s,
            overrides,
        };
        // The sites whose bonds or n(x) can change, sorted, with the change of n(x) caused by
        // the overrides: n(x) after the change differs from n(x) before it only by the Cr
        // weights of the overridden sites in the neighbourhood of x (exact integers).
        let mut tagged: Vec<(usize, i32)> = Vec::with_capacity(15 * overrides.len());
        for &(c, new) in overrides {
            let dw = new.cr_weight() as i32 - s.cr_weight_at(c) as i32;
            tagged.push((c, dw));
            tagged.extend(lattice.neighbors(c).iter().map(|&t| (t, dw)));
            tagged.extend(lattice.second_neighbors(c).iter().map(|&t| (t, dw)));
        }
        tagged.sort_unstable_by_key(|&(site, _)| site);
        let mut region: Vec<usize> = Vec::with_capacity(tagged.len());
        let mut deltas: Vec<i32> = Vec::with_capacity(tagged.len());
        for &(site, dw) in &tagged {
            if region.last() == Some(&site) {
                *deltas.last_mut().expect("deltas parallel region") += dw;
            } else {
                region.push(site);
                deltas.push(dw);
            }
        }
        let changed = |site: usize| overrides.iter().any(|&(c, _)| c == site);

        let mut sum = 0.0;
        for (site_index, &site) in region.iter().enumerate() {
            let (s_before, s_after) = (before.get(site), after.get(site));
            let s_changed = changed(site);
            let (mut n_before, mut n_after) = (None, None);
            let site_delta = deltas[site_index];
            for (shell, weight) in [
                (&lattice.neighbors(site)[..], 1.0),
                (&lattice.second_neighbors(site)[..], 0.5),
            ] {
                for &t in shell {
                    let t_changed = changed(t);
                    if !s_changed
                        && !t_changed
                        && !matches!(
                            (s.cells[site], s.cells[t]),
                            (Cell::Fe, Cell::Cr) | (Cell::Cr, Cell::Fe)
                        )
                    {
                        continue;
                    }
                    let t_position = region.binary_search(&t);
                    if t < site && t_position.is_ok() {
                        continue;
                    }
                    let t_delta = t_position.map_or(0, |i| deltas[i]);
                    // An unchanged Fe–Cr bond whose local Cr count is unchanged has an
                    // exactly zero difference.
                    if !s_changed && !t_changed && site_delta + t_delta == 0 {
                        continue;
                    }
                    let (t_before, t_after) = (before.get(t), after.get(t));
                    let n_sum_before = if is_mixed(s_before, t_before) {
                        *n_before.get_or_insert_with(|| s.n_cached(lattice, site))
                            + s.n_cached(lattice, t)
                    } else {
                        0
                    };
                    let n_sum_after = if is_mixed(s_after, t_after) {
                        let ns = *n_after.get_or_insert_with(|| {
                            (s.n_cached(lattice, site) as i32 + site_delta) as u32
                        });
                        ns + (s.n_cached(lattice, t) as i32 + t_delta) as u32
                    } else {
                        0
                    };
                    sum += weight
                        * (self.bond(s_after, t_after, n_sum_after)
                            - self.bond(s_before, t_before, n_sum_before));
                }
            }
        }

        let mut holders: Vec<usize> = Vec::with_capacity(9 * overrides.len());
        for &(c, _) in overrides {
            holders.push(c);
            holders.extend_from_slice(lattice.neighbors(c));
        }
        holders.sort_unstable();
        holders.dedup();
        for &x in &holders {
            let term = |view: &View| match view.get(x) {
                Content::Dumbbell { axis, plus, minus } => {
                    self.dumbbell_term(view, x, axis, plus, minus)
                }
                _ => 0.0,
            };
            sum += term(&after) - term(&before);
        }
        sum
    }

    /// The 6 sites in 1NN(a) ∩ 2NN(b) or 2NN(a) ∩ 1NN(b), excluding a and b.
    fn saddle_sites(&self, a: usize, b: usize) -> Vec<usize> {
        let lat = &self.lattice;
        let mut out = Vec::with_capacity(6);
        out.extend(
            lat.neighbors(a)
                .iter()
                .filter(|k| lat.second_neighbors(b).contains(k)),
        );
        out.extend(
            lat.second_neighbors(a)
                .iter()
                .filter(|k| lat.neighbors(b).contains(k)),
        );
        out.retain(|&k| k != a && k != b);
        out
    }

    /// Resolves jump `jump` (0..8) of record `rec`; `None` for an empty record or a blocked target.
    fn jump_move(&self, s: &FeCrRisState, rec: usize, jump: usize) -> Option<Move> {
        assert!(jump < 8, "jump index {jump} out of range");
        let lattice_species = |site: usize| match s.cells[site] {
            Cell::Fe => Some(Species::Fe),
            Cell::Cr => Some(Species::Cr),
            _ => None,
        };
        match s.records[rec] {
            Record::Empty => None,
            Record::Vacancy { site: q } => {
                let p = self.lattice.neighbor(q, jump);
                Some(Move::Vacancy {
                    q,
                    p,
                    atom: lattice_species(p)?,
                })
            }
            Record::Dumbbell {
                site: o,
                axis,
                plus,
                minus,
            } => {
                let u = AXES[axis];
                let (target_index, d) = DISPLACEMENTS
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| dot(**d, u).abs() == 2)
                    .nth(jump / 2)
                    .expect("a dumbbell has 4 C sites");
                let k = self.lattice.neighbor(o, target_index);
                let atom = lattice_species(k)?;
                let (jumper, rest) = if dot(*d, u) > 0 {
                    (plus, minus)
                } else {
                    (minus, plus)
                };
                let back = [-d[0], -d[1], -d[2]];
                let new_axis = (0..6)
                    .filter(|&b| dot(back, AXES[b]).abs() == 2 && dot(u, AXES[b]).abs() == 1)
                    .nth(jump % 2)
                    .expect("a dumbbell jump has 2 new axes");
                let (new_plus, new_minus) = if dot(back, AXES[new_axis]) > 0 {
                    (jumper, atom)
                } else {
                    (atom, jumper)
                };
                Some(Move::Dumbbell {
                    o,
                    k,
                    axis: new_axis,
                    plus: new_plus,
                    minus: new_minus,
                    jumper,
                    rest,
                    atom,
                })
            }
        }
    }

    /// The defect after jump `jump` of record `rec` and the species left on the old site
    /// (dumbbell: the remaining atom R on O) or moved onto it (vacancy: the atom that moved).
    ///
    /// `None` if the record is empty or the target holds a defect. Reactions are not applied.
    pub fn jump_result(
        &self,
        s: &FeCrRisState,
        rec: usize,
        jump: usize,
    ) -> Option<(Defect, Species)> {
        match self.jump_move(s, rec, jump)? {
            Move::Dumbbell {
                k,
                axis,
                plus,
                minus,
                rest,
                ..
            } => Some((
                Defect::Dumbbell {
                    site: k,
                    axis,
                    plus,
                    minus,
                },
                rest,
            )),
            Move::Vacancy { p, atom, .. } => Some((Defect::Vacancy { site: p }, atom)),
        }
    }

    fn raw_of(&self, s: &FeCrRisState, mv: &Move) -> f64 {
        let (a, b) = mv.ends();
        let mut sum = self.local_difference(s, &mv.saddle_overrides());
        let lattice_index = |site: usize| match s.cells[site] {
            Cell::Fe => Some(0),
            Cell::Cr => Some(1),
            _ => None,
        };
        let (table, jumper) = match *mv {
            Move::Dumbbell {
                jumper, rest, atom, ..
            } => {
                let t = &self.saddle;
                sum += t[jumper.index()][rest.index()] + t[jumper.index()][atom.index()];
                (*t, jumper)
            }
            Move::Vacancy { atom, .. } => (self.energetics.saddle_interactions(), atom),
        };
        for site in self.saddle_sites(a, b) {
            if let Some(x) = lattice_index(site) {
                sum += table[jumper.index()][x];
            }
        }
        sum
    }

    fn change_of(&self, s: &FeCrRisState, mv: &Move) -> f64 {
        self.local_difference(s, &mv.final_overrides())
    }

    /// E_saddle − E_initial in eV without the floor; NaN if there is no such jump.
    pub fn raw_barrier(&self, s: &FeCrRisState, rec: usize, jump: usize) -> f64 {
        self.jump_move(s, rec, jump)
            .map_or(f64::NAN, |mv| self.raw_of(s, &mv))
    }

    /// E_final − E_initial in eV; NaN if there is no such jump.
    pub fn energy_change(&self, s: &FeCrRisState, rec: usize, jump: usize) -> f64 {
        self.jump_move(s, rec, jump)
            .map_or(f64::NAN, |mv| self.change_of(s, &mv))
    }

    /// The barrier `max(raw, max(0, ΔE))` in eV; NaN if there is no such jump.
    pub fn barrier(&self, s: &FeCrRisState, rec: usize, jump: usize) -> f64 {
        self.jump_move(s, rec, jump)
            .map_or(f64::NAN, |mv| self.barrier_of(s, &mv))
    }

    fn barrier_of(&self, s: &FeCrRisState, mv: &Move) -> f64 {
        let raw = self.raw_of(s, mv);
        let de = self.change_of(s, mv);
        raw.max(de.max(0.0))
    }

    /// The jump rate in s⁻¹; 0 for an empty record or a blocked jump.
    pub fn rate(&self, s: &FeCrRisState, rec: usize, jump: usize) -> f64 {
        let Some(mv) = self.jump_move(s, rec, jump) else {
            return 0.0;
        };
        let barrier = self.barrier_of(s, &mv);
        let boltzmann = if self.kt == 0.0 {
            f64::from(u8::from(barrier <= 0.0))
        } else {
            (-barrier / self.kt).exp()
        };
        match mv {
            Move::Dumbbell { .. } => {
                let p = &self.params.dumbbell;
                p.nu0 * p.ds_mig_kb.exp() * boltzmann
            }
            Move::Vacancy { atom, .. } => {
                let nu = self
                    .energetics
                    .attempt_frequency(atom)
                    .expect("Fe and Cr have attempt frequencies");
                nu * boltzmann
            }
        }
    }
}

/// An event of the [`FeCrRis`] model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeCrRisEvent {
    /// Vacancy record `rec` jumps in `DISPLACEMENTS` direction `dir`.
    VacancyJump {
        /// Record index.
        rec: usize,
        /// Direction index 0..8.
        dir: usize,
    },
    /// Dumbbell record `rec` makes jump `jump` (0..8).
    DumbbellJump {
        /// Record index.
        rec: usize,
        /// Jump index 0..8.
        jump: usize,
    },
    /// A Frenkel pair is created.
    Create,
}

fn cell_of(species: Species) -> Cell {
    match species {
        Species::Cr => Cell::Cr,
        _ => Cell::Fe,
    }
}

impl FeCrRis {
    fn site_species(s: &FeCrRisState, site: usize) -> Species {
        match s.cells[site] {
            Cell::Cr => Species::Cr,
            _ => Species::Fe,
        }
    }

    fn free_record(s: &mut FeCrRisState, rec: usize) {
        if let Record::Vacancy { site } | Record::Dumbbell { site, .. } = s.records[rec] {
            s.site_record[site] = NO_RECORD;
        }
        s.records[rec] = Record::Empty;
        if !s.last_records.contains(&rec) {
            s.last_records.push(rec);
        }
    }

    /// Recombines record `rec` with its nearest opposite defect within [`RC_OFFSETS`], if any.
    fn recombine(&self, s: &mut FeCrRisState, rec: usize) -> bool {
        let (site, is_dumbbell) = match s.records[rec] {
            Record::Empty => return false,
            Record::Vacancy { site } => (site, false),
            Record::Dumbbell { site, .. } => (site, true),
        };
        let pos = self.lattice.position(site);
        let mut best: Option<(i64, usize, [i64; 3])> = None;
        for o in RC_OFFSETS.iter() {
            let other = self
                .lattice
                .site_at([pos[0] + o[0], pos[1] + o[1], pos[2] + o[2]])
                .expect("same-parity offsets stay on the lattice");
            let Some(r) = s.record_at(other) else {
                continue;
            };
            let opposite = matches!(
                (s.records[r], is_dumbbell),
                (Record::Vacancy { .. }, true) | (Record::Dumbbell { .. }, false)
            );
            if !opposite {
                continue;
            }
            let key = (dot(*o, *o), r);
            if best.is_none_or(|(r2, br, _)| key < (r2, br)) {
                best = Some((key.0, r, *o));
            }
        }
        let Some((_, partner, o)) = best else {
            return false;
        };
        let (i_rec, v_rec, delta) = if is_dumbbell {
            (rec, partner, o)
        } else {
            (partner, rec, [-o[0], -o[1], -o[2]])
        };
        let Record::Dumbbell {
            site: i_site,
            axis,
            plus,
            minus,
        } = s.records[i_rec]
        else {
            unreachable!("recombination partner kinds were checked")
        };
        let Record::Vacancy { site: v_site } = s.records[v_rec] else {
            unreachable!("recombination partner kinds were checked")
        };
        let (fills, stays) = if dot(AXES[axis], delta) >= 0 {
            (plus, minus)
        } else {
            (minus, plus)
        };
        s.cells[v_site] = cell_of(fills);
        s.cells[i_site] = cell_of(stays);
        s.last_changed.extend([v_site, i_site]);
        Self::free_record(s, i_rec);
        Self::free_record(s, v_rec);
        s.recombined += 1;
        true
    }

    fn create(&self, s: &mut FeCrRisState) -> Result<(), FeCrRisError> {
        let empty: Vec<usize> = (0..s.records.len())
            .filter(|&r| matches!(s.records[r], Record::Empty))
            .collect();
        if empty.len() < 2 {
            return Err(FeCrRisError::PoolExhausted {
                capacity: self.params.capacity,
                defects: s.records.len() - empty.len(),
            });
        }
        let n = self.lattice.site_count();
        let draw = |rng: &mut ChaCha12Rng, m: usize| ((open01(rng) * m as f64) as usize).min(m - 1);
        let mut found = None;
        for _ in 0..1000 {
            let site = draw(&mut s.creation_rng, n);
            let dir = draw(&mut s.creation_rng, 8);
            let orient = draw(&mut s.creation_rng, 12);
            s.creation_attempts += 1;
            let (p0, t) = (self.lattice.position(site), DISPLACEMENTS[dir]);
            let chain: Vec<usize> = (0..=10_i64)
                .map(|i| {
                    self.lattice
                        .site_at([p0[0] + i * t[0], p0[1] + i * t[1], p0[2] + i * t[2]])
                        .expect("a chain along a 111 direction stays on the lattice")
                })
                .collect();
            if chain
                .iter()
                .all(|&c| matches!(s.cells[c], Cell::Fe | Cell::Cr) && !self.on_sink(c))
            {
                found = Some((chain, orient));
                break;
            }
        }
        let Some((chain, orient)) = found else {
            return Err(FeCrRisError::CreationFailed);
        };
        let atoms: Vec<Species> = chain.iter().map(|&c| Self::site_species(s, c)).collect();
        s.cells[chain[0]] = Cell::Vacancy;
        for i in 1..=9 {
            s.cells[chain[i]] = cell_of(atoms[i - 1]);
        }
        s.cells[chain[10]] = Cell::Dumbbell;
        let (plus, minus) = if orient % 2 == 0 {
            (atoms[9], atoms[10])
        } else {
            (atoms[10], atoms[9])
        };
        let (v_rec, d_rec) = (empty[0], empty[1]);
        s.records[v_rec] = Record::Vacancy { site: chain[0] };
        s.records[d_rec] = Record::Dumbbell {
            site: chain[10],
            axis: orient / 2,
            plus,
            minus,
        };
        s.site_record[chain[0]] = v_rec as u32;
        s.site_record[chain[10]] = d_rec as u32;
        s.created += 1;
        s.last_changed.extend(chain);
        s.last_records.extend([v_rec, d_rec]);
        self.recombine(s, v_rec);
        self.recombine(s, d_rec);
        Ok(())
    }

    fn jump(
        &self,
        s: &mut FeCrRisState,
        rec: usize,
        jump: usize,
        vacancy: bool,
    ) -> Result<(), FeCrRisError> {
        let bad = FeCrRisError::NoSuchJump { rec, jump };
        if rec >= s.records.len() || jump >= 8 {
            return Err(bad);
        }
        if matches!(s.records[rec], Record::Vacancy { .. }) != vacancy {
            return Err(bad);
        }
        let Some(mv) = self.jump_move(s, rec, jump) else {
            return Err(bad);
        };
        let floored = self.raw_of(s, &mv) < self.change_of(s, &mv).max(0.0);
        s.last_changed.clear();
        s.last_records.clear();
        s.last_records.push(rec);
        let (a, b) = mv.ends();
        s.last_changed.extend([a, b]);
        s.site_record[a] = NO_RECORD;
        s.site_record[b] = rec as u32;
        match mv {
            Move::Vacancy { q, p, atom } => {
                s.floored_vacancy_jumps += u64::from(floored);
                s.cells[q] = cell_of(atom);
                s.cells[p] = Cell::Vacancy;
                s.records[rec] = Record::Vacancy { site: p };
                if !self.recombine(s, rec) && self.on_sink(p) {
                    s.cells[p] = cell_of(s.reservoir.pop_front().unwrap_or_else(|| {
                        s.borrowed_fe += 1;
                        Species::Fe
                    }));
                    Self::free_record(s, rec);
                    s.absorbed_v += 1;
                }
            }
            Move::Dumbbell {
                o,
                k,
                axis,
                plus,
                minus,
                jumper,
                rest,
                atom,
            } => {
                s.floored_dumbbell_jumps += u64::from(floored);
                s.cells[o] = cell_of(rest);
                s.cells[k] = Cell::Dumbbell;
                s.records[rec] = Record::Dumbbell {
                    site: k,
                    axis,
                    plus,
                    minus,
                };
                if !self.recombine(s, rec) && self.on_sink(k) {
                    s.cells[k] = cell_of(jumper);
                    s.reservoir.push_back(atom);
                    Self::free_record(s, rec);
                    s.absorbed_i += 1;
                }
            }
        }
        Ok(())
    }
}

impl Model for FeCrRis {
    type State = FeCrRisState;
    type Event = FeCrRisEvent;
    type Error = FeCrRisError;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        for (rec, r) in state.records.iter().enumerate() {
            for jump in 0..8 {
                let event = match r {
                    Record::Empty => break,
                    Record::Vacancy { .. } => FeCrRisEvent::VacancyJump { rec, dir: jump },
                    Record::Dumbbell { .. } => FeCrRisEvent::DumbbellJump { rec, jump },
                };
                let rate = self.rate(state, rec, jump);
                if rate > 0.0 {
                    emit(event, rate);
                }
            }
        }
        if self.params.k0_dpa_per_s > 0.0 {
            emit(FeCrRisEvent::Create, self.creation_rate());
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        let result = match *event {
            FeCrRisEvent::VacancyJump { rec, dir } => self.jump(state, rec, dir, true),
            FeCrRisEvent::DumbbellJump { rec, jump } => self.jump(state, rec, jump, false),
            FeCrRisEvent::Create => {
                state.last_changed.clear();
                state.last_records.clear();
                self.create(state)
            }
        };
        state.clear_n_cache();
        result?;
        state.peak_defects = state.peak_defects.max(state.open_records());
        Ok(())
    }
}

/// Offsets whose sites can change a record's rates: `({0} ∪ 1NN) ⊕ H ⊕ H ⊕ H`, `H = {0} ∪ 1NN ∪ 2NN`.
///
/// A jump end is the record's site or a 1NN of it; three hops then reach the bond, the
/// bond's local Cr count and the Cr counts of the bonds that change.
pub(crate) static INFLUENCE_OFFSETS: LazyLock<Vec<[i64; 3]>> = LazyLock::new(|| {
    let hop: Vec<[i64; 3]> = std::iter::once([0; 3])
        .chain(DISPLACEMENTS)
        .chain(SECOND_DISPLACEMENTS)
        .collect();
    let mut set: BTreeSet<[i64; 3]> = std::iter::once([0; 3]).chain(DISPLACEMENTS).collect();
    for _ in 0..3 {
        set = set
            .iter()
            .flat_map(|a| {
                hop.iter()
                    .map(move |h| [a[0] + h[0], a[1] + h[1], a[2] + h[2]])
            })
            .collect();
    }
    set.into_iter().collect()
});

impl IncrementalModel for FeCrRis {
    fn slot_count(&self, state: &Self::State) -> usize {
        8 * state.records.len() + 1
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        let creation = 8 * state.records.len();
        if slot < creation {
            self.rate(state, slot / 8, slot % 8)
        } else {
            self.creation_rate()
        }
    }

    fn slot_event(&self, state: &Self::State, slot: usize) -> Self::Event {
        let (rec, j) = (slot / 8, slot % 8);
        if rec >= state.records.len() {
            return FeCrRisEvent::Create;
        }
        match state.records[rec] {
            Record::Vacancy { .. } => FeCrRisEvent::VacancyJump { rec, dir: j },
            _ => FeCrRisEvent::DumbbellJump { rec, jump: j },
        }
    }

    fn affected_slots(&self, state: &Self::State, _event: &Self::Event, out: &mut Vec<usize>) {
        let mut records: Vec<usize> = state.last_records.clone();
        let [lx, ly, lz] = self.lattice.cells();
        let m = [2 * lx as i64, 2 * ly as i64, 2 * lz as i64];
        let wrap = |x: i64, axis: usize| {
            if x < 0 {
                x + m[axis]
            } else if x >= m[axis] {
                x - m[axis]
            } else {
                x
            }
        };
        for &c in &state.last_changed {
            let p = self.lattice.position(c);
            for o in INFLUENCE_OFFSETS.iter() {
                // Offsets are at most 7 in each component and cells are at least 6, so a
                // single wrap suffices. The offsets keep the sublattice parity.
                let w: [i64; 3] = std::array::from_fn(|axis| wrap(p[axis] - o[axis], axis));
                let sub = w[0] & 1;
                let site = ((((sub as usize) * lx + ((w[0] - sub) / 2) as usize) * ly
                    + ((w[1] - sub) / 2) as usize)
                    * lz)
                    + ((w[2] - sub) / 2) as usize;
                if let Some(r) = state.record_at(site) {
                    records.push(r);
                }
            }
        }
        records.sort_unstable();
        records.dedup();
        for r in records {
            out.extend(8 * r..8 * r + 8);
        }
    }
}

/// Appends `v` as little-endian bytes of its bit pattern.
fn put_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// A cursor over checkpoint bytes that fails instead of panicking.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.0.len() < n {
            return Err(format!(
                "state ends early: {n} bytes wanted, {} left",
                self.0.len()
            ));
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u128(&mut self) -> Result<u128, String> {
        Ok(u128::from_le_bytes(self.take(16)?.try_into().unwrap()))
    }
}

fn species_byte(sp: Species) -> u8 {
    sp.index() as u8
}

fn species_of(byte: u8) -> Result<Species, String> {
    match byte {
        0 => Ok(Species::Fe),
        1 => Ok(Species::Cr),
        b => Err(format!("species byte {b} is not Fe (0) or Cr (1)")),
    }
}

impl Checkpointable for FeCrRis {
    const MODEL_ID: &'static str = "okrates.fecr_ris";

    fn encode_params(&self, out: &mut Vec<u8>) {
        let p = &self.params;
        for c in p.cells {
            put_u64(out, c as u64);
        }
        put_u64(out, p.sink_planes.len() as u64);
        for &x in &p.sink_planes {
            out.extend_from_slice(&x.to_le_bytes());
        }
        put_f64(out, p.temperature_k);
        put_f64(out, p.k0_dpa_per_s);
        put_u64(out, p.capacity as u64);
        let d = &p.dumbbell;
        put_f64(out, d.nu0);
        put_f64(out, d.ds_mig_kb);
        for v in d.g {
            put_f64(out, v);
        }
        for v in d.e.iter().flatten().flatten() {
            put_f64(out, *v);
        }
        for v in d.saddle.iter().flatten() {
            put_f64(out, *v);
        }
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        out.extend(state.cells.iter().map(|c| *c as u8));
        for r in &state.records {
            let (kind, site, axis, plus, minus) = match *r {
                Record::Empty => (0_u8, 0_u32, 0_u8, 0_u8, 0_u8),
                Record::Vacancy { site } => (1, site as u32, 0, 0, 0),
                Record::Dumbbell {
                    site,
                    axis,
                    plus,
                    minus,
                } => (
                    2,
                    site as u32,
                    axis as u8,
                    species_byte(plus),
                    species_byte(minus),
                ),
            };
            out.push(kind);
            out.extend_from_slice(&site.to_le_bytes());
            out.extend_from_slice(&[axis, plus, minus]);
        }
        put_u64(out, state.reservoir.len() as u64);
        out.extend(state.reservoir.iter().map(|&sp| species_byte(sp)));
        put_u64(out, state.borrowed_fe);
        for v in [
            state.created,
            state.recombined,
            state.absorbed_v,
            state.absorbed_i,
            state.creation_attempts,
            state.floored_dumbbell_jumps,
            state.floored_vacancy_jumps,
        ] {
            put_u64(out, v);
        }
        out.extend_from_slice(&state.creation_rng.get_seed());
        put_u64(out, state.creation_rng.get_stream());
        out.extend_from_slice(&state.creation_rng.get_word_pos().to_le_bytes());
        put_u64(out, state.peak_defects);
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        Ok(self.decode(bytes)?)
    }
}

impl FeCrRis {
    /// The creation rate `K0·N` in s⁻¹ (0 when `K0` is 0).
    fn creation_rate(&self) -> f64 {
        self.params.k0_dpa_per_s * self.lattice.site_count() as f64
    }

    fn decode(&self, bytes: &[u8]) -> Result<FeCrRisState, String> {
        let n = self.lattice.site_count();
        let p = self.params.capacity;
        let mut rd = Reader(bytes);
        let mut cells = Vec::with_capacity(n);
        for &b in rd.take(n)? {
            cells.push(match b {
                0 => Cell::Fe,
                1 => Cell::Cr,
                2 => Cell::Vacancy,
                3 => Cell::Dumbbell,
                b => return Err(format!("cell byte {b} is not a content tag")),
            });
        }
        let mut records = Vec::with_capacity(p);
        let mut site_record = vec![NO_RECORD; n];
        for i in 0..p {
            let kind = rd.u8()?;
            let site = rd.u32()? as usize;
            let (axis, plus, minus) = (rd.u8()?, rd.u8()?, rd.u8()?);
            let record = match kind {
                0 => {
                    if (site, axis, plus, minus) != (0, 0, 0, 0) {
                        return Err(format!("empty record {i} has non-zero fields"));
                    }
                    Record::Empty
                }
                1 => {
                    if (axis, plus, minus) != (0, 0, 0) {
                        return Err(format!("vacancy record {i} has non-zero fields"));
                    }
                    Record::Vacancy { site }
                }
                2 => Record::Dumbbell {
                    site,
                    axis: axis as usize,
                    plus: species_of(plus)?,
                    minus: species_of(minus)?,
                },
                k => return Err(format!("record {i} has kind tag {k}")),
            };
            if let Record::Vacancy { site } | Record::Dumbbell { site, .. } = record {
                if site >= n {
                    return Err(format!("record {i} site {site} is out of range"));
                }
                if site_record[site] != NO_RECORD {
                    return Err(format!("two records on site {site}"));
                }
                let expected = match record {
                    Record::Vacancy { .. } => Cell::Vacancy,
                    _ => Cell::Dumbbell,
                };
                if cells[site] != expected {
                    return Err(format!("record {i} does not match the cell on site {site}"));
                }
                site_record[site] = i as u32;
            }
            records.push(record);
        }
        let reservoir_len = rd.u64()?;
        let reservoir_len = usize::try_from(reservoir_len)
            .ok()
            .filter(|&l| l <= rd.0.len())
            .ok_or_else(|| {
                format!("reservoir length {reservoir_len} exceeds the remaining bytes")
            })?;
        let reservoir = rd
            .take(reservoir_len)?
            .iter()
            .map(|&b| species_of(b))
            .collect::<Result<VecDeque<_>, _>>()?;
        let borrowed_fe = rd.u64()?;
        let mut counters = [0_u64; 7];
        for c in &mut counters {
            *c = rd.u64()?;
        }
        let seed: [u8; 32] = rd.take(32)?.try_into().unwrap();
        let stream = rd.u64()?;
        if stream != 1 {
            return Err(format!("creation stream is {stream}, expected 1"));
        }
        let word_pos = rd.u128()?;
        let peak_defects = rd.u64()?;
        if !rd.0.is_empty() {
            return Err(format!("{} trailing bytes after the state", rd.0.len()));
        }
        let mut creation_rng = ChaCha12Rng::from_seed(seed);
        creation_rng.set_stream(stream);
        creation_rng.set_word_pos(word_pos);
        if creation_rng.get_word_pos() != word_pos {
            return Err(format!("word position {word_pos} is out of range"));
        }
        let state = FeCrRisState {
            cells,
            records,
            site_record,
            reservoir,
            borrowed_fe,
            created: counters[0],
            recombined: counters[1],
            absorbed_v: counters[2],
            absorbed_i: counters[3],
            creation_attempts: counters[4],
            floored_dumbbell_jumps: counters[5],
            floored_vacancy_jumps: counters[6],
            creation_rng,
            peak_defects,
            last_changed: Vec::new(),
            last_records: Vec::new(),
            n_cache: RefCell::new(Vec::new()),
        };
        self.check_invariants(&state).map_err(|e| e.to_string())?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::{INFLUENCE_OFFSETS, RC_OFFSETS, creation_rng, open01};
    use okrates_core::KmcRng;

    #[test]
    fn creation_stream_zero_matches_kmc_rng_and_stream_one_differs() {
        for seed in [1_u64, 42] {
            let mut reference = KmcRng::from_seed(seed);
            let mut zero = creation_rng(seed, 0);
            for i in 0..100 {
                assert_eq!(
                    open01(&mut zero).to_bits(),
                    reference.open01().to_bits(),
                    "seed {seed} draw {i}"
                );
            }
            let first_zero = open01(&mut creation_rng(seed, 0));
            let first_one = open01(&mut creation_rng(seed, 1));
            assert_ne!(first_zero.to_bits(), first_one.to_bits());
        }
    }

    #[test]
    fn rc_offsets_count() {
        assert_eq!(RC_OFFSETS.len(), 50);
    }

    #[test]
    fn influence_offsets_count() {
        assert_eq!(INFLUENCE_OFFSETS.len(), 363);
    }
}
