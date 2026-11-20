//! Fe–Cr radiation-induced segregation model: lattice atoms, ⟨110⟩ dumbbells, and vacancies.
//!
//! Lattice bonds and vacancy saddle interactions come from [`Martinez2012`]; dumbbell
//! parameters follow O. Senninger, F. Soisson, E. Martínez, M. Nastar, C.-C. Fu, and
//! Y. Bréchet, *Acta Mater.* **103**, 1–11 (2016), doi:10.1016/j.actamat.2015.09.058
//! ([`DumbbellParams::senninger2016_r1`]), with ⟨110⟩ dumbbell migration via Johnson's 60°
//! rotation–translation jump (Johnson, *Phys. Rev.* **134**, A1329–A1336 (1964),
//! doi:10.1103/PhysRev.134.A1329; Fu, Willaime, and Ordejón, *Phys. Rev. Lett.* **92**,
//! 175503 (2004), doi:10.1103/PhysRevLett.92.175503). Positions are in half-lattice units (`a/2`).
//!
//! Thermal vacancy and dumbbell jumps use local energy differences obeying detailed balance to rounding;
//! per Senninger et al. Eq. (3), each of the four compression-site dumbbell jump branches splits its
//! transition rate equally (`1/2`) between the two symmetry-equivalent 60° final orientations (`z_eff = 4`).
//! Frenkel-pair creation, I–V recombination, and sink absorption are driven or irreversible.
//! When a jump lands on a sink site within [`RC_OFFSETS`] of an opposite defect, recombination takes
//! precedence over sink absorption.
//! Jump barriers apply a shared saddle-level floor `max(raw, max(0, ΔE))`.

use std::cell::Cell as StdCell;
use std::collections::{BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use okrates_core::{Checkpointable, IncrementalModel, KmcRng, Model, ReversibleModel};
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

#[inline]
const fn dot(a: [i64; 3], b: [i64; 3]) -> i64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

const fn contains_offset(list: &[[i64; 3]], target: [i64; 3]) -> bool {
    let mut i = 0;
    while i < list.len() {
        let item = list[i];
        if item[0] == target[0] && item[1] == target[1] && item[2] == target[2] {
            return true;
        }
        i += 1;
    }
    false
}

#[derive(Clone, Copy, Debug)]
struct DumbbellJumpSpec {
    target_dir: u8,
    jumper_is_plus: bool,
    new_axis: u8,
    jumper_is_new_plus: bool,
}

const fn build_dumbbell_jumps() -> [[DumbbellJumpSpec; 8]; 6] {
    let empty = DumbbellJumpSpec {
        target_dir: 0,
        jumper_is_plus: false,
        new_axis: 0,
        jumper_is_new_plus: false,
    };
    let mut table = [[empty; 8]; 6];
    let mut axis = 0;
    while axis < 6 {
        let u = AXES[axis];
        let mut jump = 0;
        while jump < 8 {
            let target_want = jump / 2;
            let mut seen_target = 0;
            let mut target_index = 0;
            let mut target_dir_vec = [0_i64; 3];
            let mut found_target = false;
            let mut i = 0;
            while i < DISPLACEMENTS.len() {
                let d = DISPLACEMENTS[i];
                if dot(d, u).abs() == 2 {
                    if seen_target == target_want {
                        target_index = i;
                        target_dir_vec = d;
                        found_target = true;
                        break;
                    }
                    seen_target += 1;
                }
                i += 1;
            }
            if !found_target {
                panic!("a dumbbell has 4 C sites");
            }

            let back = [-target_dir_vec[0], -target_dir_vec[1], -target_dir_vec[2]];
            let axis_want = jump % 2;
            let mut seen_axis = 0;
            let mut new_axis = 0;
            let mut found_axis = false;
            let mut b = 0;
            while b < AXES.len() {
                if dot(back, AXES[b]).abs() == 2 && dot(u, AXES[b]).abs() == 1 {
                    if seen_axis == axis_want {
                        new_axis = b;
                        found_axis = true;
                        break;
                    }
                    seen_axis += 1;
                }
                b += 1;
            }
            if !found_axis {
                panic!("a dumbbell jump has 2 new axes");
            }

            table[axis][jump] = DumbbellJumpSpec {
                target_dir: target_index as u8,
                jumper_is_plus: dot(target_dir_vec, u) > 0,
                new_axis: new_axis as u8,
                jumper_is_new_plus: dot(back, AXES[new_axis]) > 0,
            };
            jump += 1;
        }
        axis += 1;
    }
    table
}

/// Precomputed Johnson 60° jump targets and orientations for `[axis][jump]`.
const DUMBBELL_JUMPS: [[DumbbellJumpSpec; 8]; 6] = build_dumbbell_jumps();

#[inline]
const fn dumbbell_kind(plus: Species, minus: Species) -> usize {
    if matches!(
        (plus, minus),
        (Species::Fe, Species::Fe) | (Species::Cr, Species::Cr)
    ) {
        0
    } else if matches!(plus, Species::Fe) {
        1
    } else {
        2
    }
}

const fn build_dumbbell_site_types() -> [[[u8; 8]; 3]; 6] {
    let pairs = [
        (Species::Fe, Species::Fe),
        (Species::Fe, Species::Cr),
        (Species::Cr, Species::Fe),
    ];
    let mut table = [[[0_u8; 8]; 3]; 6];
    let mut axis = 0;
    while axis < 6 {
        let mut kind = 0;
        while kind < 3 {
            let (plus, minus) = pairs[kind];
            let mut dir = 0;
            while dir < 8 {
                table[axis][kind][dir] = site_type(axis, plus, minus, DISPLACEMENTS[dir]) as u8;
                dir += 1;
            }
            kind += 1;
        }
        axis += 1;
    }
    table
}

/// Precomputed 1NN site type `t` (`0=C/CFe`, `1=T`, `2=CCr`) for `[axis][kind][dir]`.
const DUMBBELL_SITE_TYPES: [[[u8; 8]; 3]; 6] = build_dumbbell_site_types();

const fn build_saddle_site_indices() -> [([u8; 3], [u8; 3]); 8] {
    let mut table = [([0_u8; 3], [0_u8; 3]); 8];
    let mut dir = 0;
    while dir < 8 {
        let d = DISPLACEMENTS[dir];
        let mut nb = [0_u8; 3];
        let mut n_nb = 0;
        let mut i = 0;
        while i < DISPLACEMENTS.len() {
            let u = DISPLACEMENTS[i];
            let diff = [u[0] - d[0], u[1] - d[1], u[2] - d[2]];
            if contains_offset(&SECOND_DISPLACEMENTS, diff) {
                if n_nb >= 3 {
                    panic!("each 1NN jump has 3 1NN-of-a saddle sites");
                }
                nb[n_nb] = i as u8;
                n_nb += 1;
            }
            i += 1;
        }
        if n_nb != 3 {
            panic!("each 1NN jump has 3 1NN-of-a saddle sites");
        }

        let mut sn = [0_u8; 3];
        let mut n_sn = 0;
        let mut j = 0;
        while j < SECOND_DISPLACEMENTS.len() {
            let s = SECOND_DISPLACEMENTS[j];
            let diff = [s[0] - d[0], s[1] - d[1], s[2] - d[2]];
            if contains_offset(&DISPLACEMENTS, diff) {
                if n_sn >= 3 {
                    panic!("each 1NN jump has 3 2NN-of-a saddle sites");
                }
                sn[n_sn] = j as u8;
                n_sn += 1;
            }
            j += 1;
        }
        if n_sn != 3 {
            panic!("each 1NN jump has 3 2NN-of-a saddle sites");
        }
        table[dir] = (nb, sn);
        dir += 1;
    }
    table
}

/// For jump direction `dir` (`a -> b`), the 3 `1NN(a) ∩ 2NN(b)` and 3 `2NN(a) ∩ 1NN(b)` indices.
const SADDLE_SITE_INDICES: [([u8; 3], [u8; 3]); 8] = build_saddle_site_indices();

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
    /// The R1 minimax fit to Senninger et al., *Acta Mater.* **103**, 1–11 (2016),
    /// doi:10.1016/j.actamat.2015.09.058, copied from `tests/data/fecr_ris_r1.txt`.
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

    /// Attempt prefactor `nu0 * exp(ds_mig_kb)`, evaluating in log space when `exp(ds_mig_kb)` overflows.
    fn prefactor(&self) -> f64 {
        let exp_ds = self.ds_mig_kb.exp();
        if exp_ds.is_finite() {
            self.nu0 * exp_ds
        } else {
            (self.nu0.ln() + self.ds_mig_kb).exp()
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
        if !self.prefactor().is_finite() {
            return Err(FeCrRisError::Parameter(
                "dumbbell rate prefactor must be finite",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeCrRisParams {
    /// Conventional cells per axis (each at least 6).
    pub cells: [usize; 3],
    /// Sink plane `x` positions in half-lattice units (`[0, 2·cells[0])`).
    pub sink_planes: Vec<i64>,
    /// Temperature in K (finite, at least 0).
    pub temperature_k: f64,
    /// Frenkel-pair creation rate per lattice site in s⁻¹ (finite, >= 0).
    ///
    /// Follows Senninger et al. (2016), Eq. (7)–(8) & Sec. 3.3, defining `dpa` directly as one
    /// Frenkel pair per site (electron irradiation), without an NRT cascade factor
    /// (cf. Norgett, Robinson, and Torrens, *Nucl. Eng. Des.* **33**, 50–54 (1975),
    /// doi:10.1016/0029-5493(75)90035-7).
    pub k0_dpa_per_s: f64,
    /// Defect record capacity `P` (at least 1, and at least 2 when `k0_dpa_per_s > 0`).
    ///
    /// Empty slots carry rate `0` under the ADR-0007 fixed-capacity defect pool contract; when
    /// `k0_dpa_per_s > 0`, Frenkel-pair creation retains its full physical hazard `K0 * N` rather
    /// than suppressing creation silently, returning [`FeCrRisError::PoolExhausted`] if fewer
    /// than two records are free.
    pub capacity: usize,
    pub dumbbell: DumbbellParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Fe,
    Cr,
    Vacancy,
    Dumbbell,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Record {
    Empty,
    Vacancy {
        site: usize,
    },
    Dumbbell {
        site: usize,
        /// Index into [`AXES`].
        axis: usize,
        /// Species at the `+u` end.
        plus: Species,
        /// Species at the `−u` end.
        minus: Species,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Defect {
    Vacancy {
        site: usize,
    },
    Dumbbell {
        site: usize,
        /// Index into [`AXES`].
        axis: usize,
        /// Species at the `+u` end (`Fe` or `Cr`).
        plus: Species,
        /// Species at the `−u` end (`Fe` or `Cr`).
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

#[derive(Clone, Debug, PartialEq)]
pub enum FeCrRisError {
    Lattice(LatticeError),
    Energetics(FeCrError),
    CellsTooSmall {
        cells: [usize; 3],
    },
    Parameter(&'static str),
    SinkPlane(i64),
    SiteCount {
        expected: usize,
        found: usize,
    },
    VacancyInOccupancy {
        site: usize,
    },
    TooManyAtoms,
    TooManyDefects {
        capacity: usize,
        defects: usize,
    },
    DuplicateDefect {
        site: usize,
    },
    DefectOutOfRange {
        site: usize,
    },
    DefectOnSink {
        site: usize,
    },
    InvalidDefect {
        site: usize,
    },
    WithinRecombinationRadius {
        vacancy: usize,
        dumbbell: usize,
    },
    Inconsistent(&'static str),
    /// Creation was selected with fewer than two free defect records (ADR-0007).
    PoolExhausted {
        capacity: usize,
        defects: usize,
    },
    CreationFailed,
    NoSuchJump {
        rec: usize,
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

fn creation_rng(run_seed: u64, stream: u64) -> ChaCha12Rng {
    let mut bytes = [0_u8; 32];
    bytes[..8].copy_from_slice(&run_seed.to_le_bytes());
    let mut rng = ChaCha12Rng::from_seed(bytes);
    rng.set_stream(stream);
    rng
}

/// Uniform draw strictly inside `(0, 1)`, matching [`KmcRng::open01`].
fn open01(rng: &mut ChaCha12Rng) -> f64 {
    ((rng.next_u64() >> 12) as f64 + 0.5) * f64::EPSILON
}

fn live_counts(records: &[Record]) -> (usize, usize) {
    let mut vacancies = 0_usize;
    let mut dumbbells = 0_usize;
    for r in records {
        match r {
            Record::Empty => {}
            Record::Vacancy { .. } => vacancies += 1,
            Record::Dumbbell { .. } => dumbbells += 1,
        }
    }
    (vacancies, dumbbells)
}

static NEXT_MODEL_UID: AtomicU64 = AtomicU64::new(1);

fn next_model_uid() -> u64 {
    NEXT_MODEL_UID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug)]
struct DumbbellBranchCache {
    model_uid: u64,
    version: u64,
    rec: usize,
    branch: u8,
    raw: f64,
    bond_de: f64,
    term_before_o: f64,
}

impl DumbbellBranchCache {
    const fn empty() -> Self {
        Self {
            model_uid: 0,
            version: 0,
            rec: 0,
            branch: 0,
            raw: 0.0,
            bond_de: 0.0,
            term_before_o: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RecordFloorCache {
    model_uid: u64,
    version: u64,
    rec: usize,
    valid_mask: u8,
    floored_mask: u8,
}

impl RecordFloorCache {
    const fn empty() -> Self {
        Self {
            model_uid: 0,
            version: 0,
            rec: 0,
            valid_mask: 0,
            floored_mask: 0,
        }
    }
}

type DerivedTables = (Vec<u8>, Vec<usize>, Vec<u32>);

#[derive(Clone, Debug)]
pub struct FeCrRisState {
    cells: Vec<Cell>,
    records: Vec<Record>,
    site_record: Vec<u32>,
    cr_n: Vec<u8>,
    active_records: Vec<usize>,
    active_pos: Vec<u32>,
    live_vacancies: usize,
    live_dumbbells: usize,
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
    version: u64,
    branch_cache: StdCell<DumbbellBranchCache>,
    floor_cache: StdCell<RecordFloorCache>,
}

impl FeCrRisState {
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }
    pub fn records(&self) -> &[Record] {
        &self.records
    }
    /// Indices of currently non-empty defect records in `records()`.
    pub fn active_records(&self) -> &[usize] {
        &self.active_records
    }
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
    /// Fe atoms supplied when a vacancy reached a sink with an empty reservoir.
    pub fn borrowed_fe(&self) -> u64 {
        self.borrowed_fe
    }
    pub fn created(&self) -> u64 {
        self.created
    }
    pub fn recombined(&self) -> u64 {
        self.recombined
    }
    pub fn absorbed_v(&self) -> u64 {
        self.absorbed_v
    }
    pub fn absorbed_i(&self) -> u64 {
        self.absorbed_i
    }
    /// Creation chain draws, including accepted ones.
    pub fn creation_attempts(&self) -> u64 {
        self.creation_attempts
    }
    /// Number of executed dumbbell jumps whose raw barrier fell below `max(0, dE)`.
    ///
    /// Recomputes `raw < max(0, dE)` on cache miss when [`Model::apply`] is called without a prior [`FeCrRis::rate`] query.
    pub fn floored_dumbbell_jumps(&self) -> u64 {
        self.floored_dumbbell_jumps
    }
    /// Number of executed vacancy jumps whose raw barrier fell below `max(0, dE)`.
    ///
    /// Recomputes `raw < max(0, dE)` on cache miss when [`Model::apply`] is called without a prior [`FeCrRis::rate`] query.
    pub fn floored_vacancy_jumps(&self) -> u64 {
        self.floored_vacancy_jumps
    }
    pub fn peak_defects(&self) -> u64 {
        self.peak_defects
    }
    pub fn creation_word_pos(&self) -> u128 {
        self.creation_rng.get_word_pos()
    }
    pub fn last_changed(&self) -> &[usize] {
        &self.last_changed
    }
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
    /// `(vacancies, dumbbells)` currently active in `records`.
    pub fn defect_counts(&self) -> (usize, usize) {
        (self.live_vacancies, self.live_dumbbells)
    }

    #[inline]
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
    #[inline]
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

    #[inline]
    fn add_cr_weight(&mut self, lattice: &BccLattice, site: usize, delta: i8) {
        if delta == 0 {
            return;
        }
        let d = i16::from(delta);
        self.cr_n[site] = (i16::from(self.cr_n[site]) + d) as u8;
        for &t in lattice.neighbors(site) {
            self.cr_n[t] = (i16::from(self.cr_n[t]) + d) as u8;
        }
        for &t in lattice.second_neighbors(site) {
            self.cr_n[t] = (i16::from(self.cr_n[t]) + d) as u8;
        }
    }

    #[inline]
    fn activate_record(&mut self, rec: usize) {
        debug_assert_eq!(self.active_pos[rec], NO_RECORD);
        self.active_pos[rec] = self.active_records.len() as u32;
        self.active_records.push(rec);
    }

    #[inline]
    fn deactivate_record(&mut self, rec: usize) {
        let pos = self.active_pos[rec];
        debug_assert_ne!(pos, NO_RECORD);
        let idx = pos as usize;
        if idx < self.active_records.len() {
            self.active_records.swap_remove(idx);
            if idx < self.active_records.len() {
                let moved = self.active_records[idx];
                self.active_pos[moved] = idx as u32;
            }
        }
        self.active_pos[rec] = NO_RECORD;
    }

    fn derive_tables(
        lattice: &BccLattice,
        cells: &[Cell],
        records: &[Record],
    ) -> Result<DerivedTables, FeCrRisError> {
        let n = lattice.site_count();
        let mut cr_n = Vec::new();
        cr_n.try_reserve_exact(n)
            .map_err(|_| FeCrRisError::Parameter("state allocation exceeds memory limit"))?;
        cr_n.resize(n, 0);
        let mut active_pos = Vec::new();
        active_pos
            .try_reserve_exact(records.len())
            .map_err(|_| FeCrRisError::Parameter("state allocation exceeds memory limit"))?;
        active_pos.resize(records.len(), NO_RECORD);

        let mut add = |site: usize, w: u8| {
            if w != 0 && site < n {
                cr_n[site] += w;
                for &t in lattice.neighbors(site) {
                    cr_n[t] += w;
                }
                for &t in lattice.second_neighbors(site) {
                    cr_n[t] += w;
                }
            }
        };
        for (site, &cell) in cells.iter().enumerate() {
            if cell == Cell::Cr {
                add(site, 1);
            }
        }
        let mut active_records = Vec::new();
        active_records
            .try_reserve_exact(records.len())
            .map_err(|_| FeCrRisError::Parameter("state allocation exceeds memory limit"))?;
        for (rec, &r) in records.iter().enumerate() {
            if !matches!(r, Record::Empty) {
                active_pos[rec] = active_records.len() as u32;
                active_records.push(rec);
            }
            if let Record::Dumbbell {
                site, plus, minus, ..
            } = r
            {
                let w = u8::from(plus == Species::Cr) + u8::from(minus == Species::Cr);
                add(site, w);
            }
        }
        Ok((cr_n, active_records, active_pos))
    }

    #[inline]
    fn bump_version(&mut self) {
        self.version = self.version.wrapping_add(1);
        if self.version == 0 {
            self.version = 1;
        }
    }

    fn open_records(&self) -> u64 {
        (self.live_vacancies + self.live_dumbbells) as u64
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

#[inline]
fn lookup_flags(map_key: &[u32; 64], map_fl: &[u8; 64], site: usize) -> u8 {
    let key = site as u32;
    let mut slot = ((key.wrapping_mul(0x9e37_79b9)) >> 26) as usize;
    loop {
        let k = map_key[slot];
        if k == key {
            return map_fl[slot];
        }
        if k == NO_RECORD {
            return 0;
        }
        slot = (slot + 1) & 63;
    }
}

#[inline]
fn delta_of(fl: u8, dw0: i32, dw1: i32) -> i32 {
    (if (fl & 0x13) != 0 { dw0 } else { 0 }) + (if (fl & 0x2c) != 0 { dw1 } else { 0 })
}

fn build_region(lattice: &BccLattice, c0: usize, c1: usize) -> ([usize; 30], [u8; 30], usize) {
    let mut items = [0_u64; 30];
    let mut n = 0_usize;
    let mut push = |site: usize, bit: u8| {
        items[n] = ((site as u64) << 8) | u64::from(bit);
        n += 1;
    };
    push(c0, 0x10);
    for &t in lattice.neighbors(c0) {
        push(t, 0x01);
    }
    for &t in lattice.second_neighbors(c0) {
        push(t, 0x02);
    }
    push(c1, 0x20);
    for &t in lattice.neighbors(c1) {
        push(t, 0x04);
    }
    for &t in lattice.second_neighbors(c1) {
        push(t, 0x08);
    }
    debug_assert_eq!(n, 30);

    for i in 1..30 {
        let x = items[i];
        let mut j = i;
        while j > 0 && items[j - 1] > x {
            items[j] = items[j - 1];
            j -= 1;
        }
        items[j] = x;
    }

    let mut region = [0_usize; 30];
    let mut flags = [0_u8; 30];
    let mut len = 0_usize;
    for &x in &items {
        let site = (x >> 8) as usize;
        let fl = (x & 0xff) as u8;
        if len > 0 && region[len - 1] == site {
            flags[len - 1] |= fl;
        } else {
            region[len] = site;
            flags[len] = fl;
            len += 1;
        }
    }
    (region, flags, len)
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
const fn site_type(axis: usize, plus: Species, minus: Species, d: [i64; 3]) -> usize {
    let along = dot(d, AXES[axis]);
    if along == 0 {
        return 1;
    }
    if matches!(
        (plus, minus),
        (Species::Fe, Species::Fe)
            | (Species::Cr, Species::Cr)
            | (Species::Vacancy, Species::Vacancy)
    ) {
        return 0;
    }
    let facing = if along > 0 { plus } else { minus };
    if matches!(facing, Species::Fe) { 0 } else { 2 }
}

/// A jump, resolved: the sites, species and the new defect.
#[derive(Clone, Copy, Debug)]
enum Move {
    Dumbbell {
        /// Old dumbbell site.
        o: usize,
        /// Target site.
        k: usize,
        /// Direction index into `DISPLACEMENTS` (`o -> k`).
        dir: u8,
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
        dir: u8,
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

    fn dir(&self) -> usize {
        match *self {
            Self::Dumbbell { dir, .. } | Self::Vacancy { dir, .. } => dir as usize,
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
            Self::Vacancy { q, p, atom, .. } => [(q, Content::of(atom)), (p, Content::Vacancy)],
        }
    }

    fn saddle_overrides(&self) -> [(usize, Content); 2] {
        let (a, b) = self.ends();
        [(a, Content::Empty), (b, Content::Empty)]
    }
}

fn sink_table(lattice: &BccLattice, sink_planes: &[i64]) -> Vec<bool> {
    let mut sink_x = vec![false; 2 * lattice.cells()[0]];
    for &x in sink_planes {
        if let Ok(idx) = usize::try_from(x)
            && let Some(slot) = sink_x.get_mut(idx)
        {
            *slot = true;
        }
    }
    sink_x
}

/// The Fe–Cr RIS model.
#[derive(Clone, Debug)]
pub struct FeCrRis {
    model_uid: u64,
    params: FeCrRisParams,
    lattice: BccLattice,
    energetics: Martinez2012,
    kt: f64,
    /// Saddle interactions refitted at T (r2-spec §2.5); `params` keeps the fitted values.
    saddle: [[f64; 2]; 2],
    sink_x: Vec<bool>,
    half_sites: usize,
    plane_stride: usize,
    dumbbell_prefactor: f64,
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
        if params.k0_dpa_per_s > 0.0 && params.capacity < 2 {
            return Err(FeCrRisError::Parameter(
                "capacity must be at least 2 when K0 > 0",
            ));
        }
        if params.capacity >= NO_RECORD as usize {
            return Err(FeCrRisError::Parameter(
                "capacity exceeds record index limit",
            ));
        }
        let slot_count = 8_usize
            .checked_mul(params.capacity)
            .and_then(|n| n.checked_add(1))
            .ok_or(FeCrRisError::Parameter("capacity overflows slot count"))?;
        okrates_core::SumTree::check_len(slot_count)
            .map_err(|_| FeCrRisError::Parameter("capacity exceeds sum-tree limit"))?;
        params
            .capacity
            .checked_mul(std::mem::size_of::<Record>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(FeCrRisError::Parameter(
                "capacity exceeds record allocation limit",
            ))?;
        let lattice = BccLattice::new(params.cells).map_err(FeCrRisError::Lattice)?;
        if u32::try_from(lattice.site_count()).is_err() {
            return Err(FeCrRisError::Lattice(LatticeError::TooLarge {
                cells: params.cells,
            }));
        }
        let width = i64::try_from(params.cells[0])
            .ok()
            .and_then(|c| c.checked_mul(2))
            .ok_or(FeCrRisError::Lattice(LatticeError::TooLarge {
                cells: params.cells,
            }))?;
        if let Some(&x) = params.sink_planes.iter().find(|&&x| x < 0 || x >= width) {
            return Err(FeCrRisError::SinkPlane(x));
        }
        params.dumbbell.validate()?;
        if !(params.k0_dpa_per_s * lattice.site_count() as f64).is_finite() {
            return Err(FeCrRisError::Parameter("creation rate must be finite"));
        }
        let energetics =
            Martinez2012::new(params.temperature_k).map_err(FeCrRisError::Energetics)?;
        let kt = BOLTZMANN_EV_PER_K * params.temperature_k;
        let saddle = if params.temperature_k > 0.0 {
            Self::refit_saddle(&params)?
        } else {
            params.dumbbell.saddle
        };
        let dumbbell_prefactor = params.dumbbell.prefactor();
        let sink_x = sink_table(&lattice, &params.sink_planes);
        let [lx, ly, lz] = lattice.cells();
        let plane_stride = ly * lz;
        let half_sites = lx * plane_stride;
        Ok(Self {
            model_uid: next_model_uid(),
            params,
            lattice,
            energetics,
            kt,
            saddle,
            sink_x,
            half_sites,
            plane_stride,
            dumbbell_prefactor,
        })
    }

    /// A 6×6×6 model without sinks at `temperature_k` with saddle table `saddle`.
    fn probe(
        params: &FeCrRisParams,
        temperature_k: f64,
        saddle: [[f64; 2]; 2],
    ) -> Result<Self, FeCrRisError> {
        let lattice = BccLattice::new([6, 6, 6]).map_err(FeCrRisError::Lattice)?;
        let [lx, ly, lz] = lattice.cells();
        let plane_stride = ly * lz;
        let half_sites = lx * plane_stride;
        let dumbbell_prefactor = params.dumbbell.prefactor();
        Ok(Self {
            model_uid: next_model_uid(),
            params: FeCrRisParams {
                cells: [6, 6, 6],
                sink_planes: Vec::new(),
                capacity: 1,
                temperature_k,
                ..params.clone()
            },
            sink_x: sink_table(&lattice, &[]),
            lattice,
            energetics: Martinez2012::new(temperature_k).map_err(FeCrRisError::Energetics)?,
            kt: BOLTZMANN_EV_PER_K * temperature_k,
            saddle,
            half_sites,
            plane_stride,
            dumbbell_prefactor,
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
            let (j, mv) = (0..8)
                .filter_map(|j| self.jump_move(&s, 0, j).map(|mv| (j, mv)))
                .find(|(_, mv)| matches!(*mv, Move::Dumbbell { jumper, .. } if jumper == mover))
                .expect("a pure-Fe dumbbell has jumps for both of its atoms");
            out[i] = self.raw_and_change(&s, 0, j, &mv).0;
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

    pub fn lattice(&self) -> &BccLattice {
        &self.lattice
    }

    /// Adds `shift` (eV) to all four dumbbell saddle interactions.
    ///
    /// Exists only to force the barrier floor in tests.
    #[doc(hidden)]
    pub fn with_dumbbell_saddle_shift(mut self, shift: f64) -> Self {
        self.model_uid = next_model_uid();
        for table in [&mut self.params.dumbbell.saddle, &mut self.saddle] {
            for g in table.iter_mut().flatten() {
                *g += shift;
            }
        }
        self
    }

    #[inline]
    fn on_sink(&self, site: usize) -> bool {
        let (sub, within) = if site < self.half_sites {
            (0, site)
        } else {
            (1, site - self.half_sites)
        };
        let x = 2 * (within / self.plane_stride) + sub;
        self.sink_x[x]
    }

    #[inline]
    fn wrapped_site_at(&self, w: [i64; 3]) -> usize {
        let [lx, ly, lz] = self.lattice.cells();
        let sub = (w[0] & 1) as usize;
        let i = (w[0] as usize) >> 1;
        let j = (w[1] as usize) >> 1;
        let k = (w[2] as usize) >> 1;
        (((sub * lx + i) * ly + j) * lz) + k
    }

    fn new_state(
        &self,
        cells: Vec<Cell>,
        records: Vec<Record>,
        site_record: Vec<u32>,
        run_seed: u64,
    ) -> Result<FeCrRisState, FeCrRisError> {
        let (live_vacancies, live_dumbbells) = live_counts(&records);
        let (cr_n, active_records, active_pos) =
            FeCrRisState::derive_tables(&self.lattice, &cells, &records)?;
        let mut state = FeCrRisState {
            cells,
            records,
            site_record,
            cr_n,
            active_records,
            active_pos,
            live_vacancies,
            live_dumbbells,
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
            version: 1,
            branch_cache: StdCell::new(DumbbellBranchCache::empty()),
            floor_cache: StdCell::new(RecordFloorCache::empty()),
        };
        state.peak_defects = state.open_records();
        Ok(state)
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
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(n)
            .map_err(|_| FeCrRisError::Parameter("state allocation exceeds memory limit"))?;
        for (site, sp) in occ.iter().enumerate() {
            cells.push(match sp {
                Species::Fe => Cell::Fe,
                Species::Cr => Cell::Cr,
                Species::Vacancy => return Err(FeCrRisError::VacancyInOccupancy { site }),
            });
        }
        let mut records = Vec::new();
        records
            .try_reserve_exact(self.params.capacity)
            .map_err(|_| FeCrRisError::Parameter("state allocation exceeds memory limit"))?;
        records.resize(self.params.capacity, Record::Empty);
        let mut site_record = Vec::new();
        site_record
            .try_reserve_exact(n)
            .map_err(|_| FeCrRisError::Parameter("state allocation exceeds memory limit"))?;
        site_record.resize(n, NO_RECORD);
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
        let state = self.new_state(cells, records, site_record, run_seed)?;
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

    /// Checks sizes, the record/cell correspondence, derived caches, sink planes and the recombination invariant.
    pub fn check_invariants(&self, s: &FeCrRisState) -> Result<(), FeCrRisError> {
        let n = self.lattice.site_count();
        if s.cells.len() != n || s.site_record.len() != n || s.cr_n.len() != n {
            return Err(FeCrRisError::SiteCount {
                expected: n,
                found: s.cells.len(),
            });
        }
        if s.records.len() != self.params.capacity {
            return Err(FeCrRisError::Inconsistent("record count differs from P"));
        }
        if s.active_pos.len() != self.params.capacity {
            return Err(FeCrRisError::Inconsistent(
                "active_pos count differs from P",
            ));
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
        if s.creation_attempts < s.created {
            return Err(FeCrRisError::Inconsistent(
                "creation_attempts is less than created",
            ));
        }
        let (mut v_count, mut d_count) = (0_usize, 0_usize);
        for (i, r) in s.records.iter().enumerate() {
            let (site, cell) = match *r {
                Record::Empty => {
                    if s.active_pos[i] != NO_RECORD {
                        return Err(FeCrRisError::Inconsistent("empty record listed as active"));
                    }
                    continue;
                }
                Record::Vacancy { site } => {
                    v_count += 1;
                    (site, Cell::Vacancy)
                }
                Record::Dumbbell {
                    site,
                    axis,
                    plus,
                    minus,
                } => {
                    d_count += 1;
                    if axis >= 6 || plus == Species::Vacancy || minus == Species::Vacancy {
                        return Err(FeCrRisError::InvalidDefect { site });
                    }
                    (site, Cell::Dumbbell)
                }
            };
            let pos = s.active_pos[i] as usize;
            if pos >= s.active_records.len() || s.active_records[pos] != i {
                return Err(FeCrRisError::Inconsistent(
                    "active_records does not match active_pos",
                ));
            }
            if site >= n || s.cells[site] != cell || s.site_record[site] != i as u32 {
                return Err(FeCrRisError::Inconsistent("record does not match its site"));
            }
            if self.on_sink(site) {
                return Err(FeCrRisError::DefectOnSink { site });
            }
        }
        if s.active_records.len() != v_count + d_count {
            return Err(FeCrRisError::Inconsistent(
                "active_records length does not match live defect count",
            ));
        }
        if (s.live_vacancies, s.live_dumbbells) != (v_count, d_count) {
            return Err(FeCrRisError::Inconsistent(
                "live defect counters do not match records",
            ));
        }
        if s.site_record.iter().filter(|&&r| r != NO_RECORD).count() as u64 != s.open_records() {
            return Err(FeCrRisError::Inconsistent(
                "site map and records differ in number",
            ));
        }
        for site in 0..n {
            if u32::from(s.cr_n[site]) != s.n(&self.lattice, site) {
                return Err(FeCrRisError::Inconsistent(
                    "cr_n cache disagrees with local Cr count",
                ));
            }
        }
        if let Some((vacancy, dumbbell)) = self.recombination_pair(s) {
            return Err(FeCrRisError::WithinRecombinationRadius { vacancy, dumbbell });
        }
        Ok(())
    }

    #[inline]
    fn bond(&self, a: Content, b: Content, n_sum: u32) -> f64 {
        self.energetics
            .bond_energy(a.bond_species(), b.bond_species(), n_sum)
    }

    /// g[D] + Σ e over the Fe/Cr 1NN sites of a dumbbell.
    #[inline]
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
        let st = &DUMBBELL_SITE_TYPES[axis][dumbbell_kind(plus, minus)];
        let mut term = p.g[d_type];
        for (i, &nb) in self.lattice.neighbors(site).iter().enumerate() {
            let x = match view.get(nb) {
                Content::Fe => 0,
                Content::Cr => 1,
                _ => continue,
            };
            term += p.e[d_type][st[i] as usize][x];
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

    /// Computes `(raw_bond_delta, final_bond_delta)` for `mv` in one sorted pass over the local region.
    fn bond_raw_and_change(&self, s: &FeCrRisState, mv: &Move) -> (f64, f64) {
        let lattice = &self.lattice;
        let [(c0, o0), (c1, o1)] = mv.final_overrides();
        debug_assert_ne!(c0, c1);
        debug_assert!(lattice.neighbors(c0).contains(&c1));

        let w0 = s.cr_weight_at(c0) as i32;
        let w1 = s.cr_weight_at(c1) as i32;
        let dw_raw0 = -w0;
        let dw_raw1 = -w1;
        let dw_chg0 = o0.cr_weight() as i32 - w0;
        let dw_chg1 = o1.cr_weight() as i32 - w1;

        let (region, flags, r_len) = build_region(lattice, c0, c1);
        let mut sum_raw = 0.0;
        let mut sum_chg = 0.0;

        if dw_raw0 == 0 && dw_raw1 == 0 && dw_chg0 == 0 && dw_chg1 == 0 {
            for i in 0..r_len {
                let site = region[i];
                let fl = flags[i];
                if (fl & 0x30) != 0 {
                    debug_assert_ne!(fl & 0x30, 0x30);
                    let s_before = s.content(site);
                    let s_after = if (fl & 0x10) != 0 { o0 } else { o1 };
                    let n_s = u32::from(s.cr_n[site]);
                    for (shell, weight) in [
                        (&lattice.neighbors(site)[..], 1.0),
                        (&lattice.second_neighbors(site)[..], 0.5),
                    ] {
                        for &t in shell {
                            if t < site {
                                continue;
                            }
                            let t_before = s.content(t);
                            let t_after = if t == c0 {
                                o0
                            } else if t == c1 {
                                o1
                            } else {
                                t_before
                            };
                            let n_before = if is_mixed(s_before, t_before) {
                                n_s + u32::from(s.cr_n[t])
                            } else {
                                0
                            };
                            let b_before = self.bond(s_before, t_before, n_before);
                            let n_after = if is_mixed(s_after, t_after) {
                                n_s + u32::from(s.cr_n[t])
                            } else {
                                0
                            };
                            sum_raw += weight * (0.0 - b_before);
                            sum_chg += weight * (self.bond(s_after, t_after, n_after) - b_before);
                        }
                    }
                } else {
                    debug_assert!((fl & 0x05) != 0x05 && (fl & 0x0a) != 0x0a);
                    let s_content = s.content(site);
                    let n_s = u32::from(s.cr_n[site]);
                    for (c, o_c, cond, weight) in [
                        (c0, o0, fl & 0x01 != 0, 1.0),
                        (c1, o1, fl & 0x04 != 0, 1.0),
                        (c0, o0, fl & 0x02 != 0, 0.5),
                        (c1, o1, fl & 0x08 != 0, 0.5),
                    ] {
                        if cond && c > site {
                            let t_before = s.content(c);
                            let n_before = if is_mixed(s_content, t_before) {
                                n_s + u32::from(s.cr_n[c])
                            } else {
                                0
                            };
                            let b_before = self.bond(s_content, t_before, n_before);
                            let n_after = if is_mixed(s_content, o_c) {
                                n_s + u32::from(s.cr_n[c])
                            } else {
                                0
                            };
                            sum_raw += weight * (0.0 - b_before);
                            sum_chg += weight * (self.bond(s_content, o_c, n_after) - b_before);
                        }
                    }
                }
            }
            return (sum_raw, sum_chg);
        }

        let mut map_key = [NO_RECORD; 64];
        let mut map_fl = [0_u8; 64];
        for i in 0..r_len {
            let key = region[i] as u32;
            let mut slot = ((key.wrapping_mul(0x9e37_79b9)) >> 26) as usize;
            while map_key[slot] != NO_RECORD {
                slot = (slot + 1) & 63;
            }
            map_key[slot] = key;
            map_fl[slot] = flags[i];
        }
        let fl_c0 = lookup_flags(&map_key, &map_fl, c0);
        let fl_c1 = lookup_flags(&map_key, &map_fl, c1);

        for i in 0..r_len {
            let site = region[i];
            let fl = flags[i];
            if (fl & 0x30) != 0 {
                debug_assert_ne!(fl & 0x30, 0x30);
                let s_before = s.content(site);
                let s_after = if (fl & 0x10) != 0 { o0 } else { o1 };
                let site_delta_chg = delta_of(fl, dw_chg0, dw_chg1);
                let n_s = u32::from(s.cr_n[site]);
                for (shell, weight) in [
                    (&lattice.neighbors(site)[..], 1.0),
                    (&lattice.second_neighbors(site)[..], 0.5),
                ] {
                    for &t in shell {
                        if t < site {
                            continue;
                        }
                        let t_before = s.content(t);
                        let t_after = if t == c0 {
                            o0
                        } else if t == c1 {
                            o1
                        } else {
                            t_before
                        };
                        let n_before = if is_mixed(s_before, t_before) {
                            n_s + u32::from(s.cr_n[t])
                        } else {
                            0
                        };
                        let b_before = self.bond(s_before, t_before, n_before);
                        sum_raw += weight * (0.0 - b_before);
                        let n_after = if is_mixed(s_after, t_after) {
                            let t_fl = lookup_flags(&map_key, &map_fl, t);
                            debug_assert_ne!(t_fl, 0);
                            (n_s as i32 + site_delta_chg) as u32
                                + (i32::from(s.cr_n[t]) + delta_of(t_fl, dw_chg0, dw_chg1)) as u32
                        } else {
                            0
                        };
                        sum_chg += weight * (self.bond(s_after, t_after, n_after) - b_before);
                    }
                }
            } else {
                let s_cell = s.cells[site];
                let s_content = s.content(site);
                let n_s = u32::from(s.cr_n[site]);
                let site_delta_raw = delta_of(fl, dw_raw0, dw_raw1);
                let site_delta_chg = delta_of(fl, dw_chg0, dw_chg1);
                let cr_from_ends = (if (fl & 0x03) != 0 { w0 } else { 0 })
                    + (if (fl & 0x0c) != 0 { w1 } else { 0 });
                if s_cell == Cell::Fe && n_s as i32 == cr_from_ends {
                    debug_assert!((fl & 0x05) != 0x05 && (fl & 0x0a) != 0x0a);
                    for (c, o_c, fl_c, cond, weight) in [
                        (c0, o0, fl_c0, fl & 0x01 != 0, 1.0),
                        (c1, o1, fl_c1, fl & 0x04 != 0, 1.0),
                        (c0, o0, fl_c0, fl & 0x02 != 0, 0.5),
                        (c1, o1, fl_c1, fl & 0x08 != 0, 0.5),
                    ] {
                        if cond && c > site {
                            let t_before = s.content(c);
                            let n_before = if is_mixed(s_content, t_before) {
                                n_s + u32::from(s.cr_n[c])
                            } else {
                                0
                            };
                            let b_before = self.bond(s_content, t_before, n_before);
                            sum_raw += weight * (0.0 - b_before);
                            let n_after = if is_mixed(s_content, o_c) {
                                (n_s as i32 + site_delta_chg) as u32
                                    + (i32::from(s.cr_n[c]) + delta_of(fl_c, dw_chg0, dw_chg1))
                                        as u32
                            } else {
                                0
                            };
                            sum_chg += weight * (self.bond(s_content, o_c, n_after) - b_before);
                        }
                    }
                    continue;
                }

                for (shell, weight) in [
                    (&lattice.neighbors(site)[..], 1.0),
                    (&lattice.second_neighbors(site)[..], 0.5),
                ] {
                    for &t in shell {
                        let t_changed = t == c0 || t == c1;
                        if !t_changed
                            && !matches!(
                                (s_cell, s.cells[t]),
                                (Cell::Fe, Cell::Cr) | (Cell::Cr, Cell::Fe)
                            )
                        {
                            continue;
                        }
                        let t_fl = lookup_flags(&map_key, &map_fl, t);
                        if t < site && t_fl != 0 {
                            continue;
                        }
                        if t_changed {
                            let t_before = s.content(t);
                            let t_after = if t == c0 { o0 } else { o1 };
                            let n_before = if is_mixed(s_content, t_before) {
                                n_s + u32::from(s.cr_n[t])
                            } else {
                                0
                            };
                            let b_before = self.bond(s_content, t_before, n_before);
                            sum_raw += weight * (0.0 - b_before);
                            let n_after = if is_mixed(s_content, t_after) {
                                (n_s as i32 + site_delta_chg) as u32
                                    + (i32::from(s.cr_n[t]) + delta_of(t_fl, dw_chg0, dw_chg1))
                                        as u32
                            } else {
                                0
                            };
                            sum_chg += weight * (self.bond(s_content, t_after, n_after) - b_before);
                        } else {
                            let d_raw = site_delta_raw + delta_of(t_fl, dw_raw0, dw_raw1);
                            let d_chg = site_delta_chg + delta_of(t_fl, dw_chg0, dw_chg1);
                            if d_raw == 0 && d_chg == 0 {
                                continue;
                            }
                            let n_before = n_s + u32::from(s.cr_n[t]);
                            let b_before = self.energetics.fe_cr_bond_energy(n_before);
                            if d_raw != 0 {
                                sum_raw += weight
                                    * (self
                                        .energetics
                                        .fe_cr_bond_energy((n_before as i32 + d_raw) as u32)
                                        - b_before);
                            }
                            if d_chg != 0 {
                                sum_chg += weight
                                    * (self
                                        .energetics
                                        .fe_cr_bond_energy((n_before as i32 + d_chg) as u32)
                                        - b_before);
                            }
                        }
                    }
                }
            }
        }
        (sum_raw, sum_chg)
    }

    fn add_saddle_interactions(&self, s: &FeCrRisState, mv: &Move, mut sum: f64) -> f64 {
        let a = mv.ends().0;
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
        for site in self.saddle_sites(a, mv.dir()) {
            if let Some(x) = lattice_index(site) {
                sum += table[jumper.index()][x];
            }
        }
        sum
    }

    /// Computes `(raw_barrier, energy_change)` for `mv`, hitting the single-dumbbell branch cache when valid.
    fn raw_and_change(&self, s: &FeCrRisState, rec: usize, jump: usize, mv: &Move) -> (f64, f64) {
        let [(c0, o0), (c1, o1)] = mv.final_overrides();
        if let Move::Dumbbell {
            o,
            k,
            axis,
            plus,
            minus,
            ..
        } = *mv
            && s.live_dumbbells == 1
        {
            let branch = (jump / 2) as u8;
            let cache = s.branch_cache.get();
            if cache.model_uid == self.model_uid
                && cache.version == s.version
                && cache.rec == rec
                && cache.branch == branch
            {
                let ov = mv.final_overrides();
                let after = View {
                    state: s,
                    overrides: &ov,
                };
                let term_after_k = self.dumbbell_term(&after, k, axis, plus, minus);
                let de = if o < k {
                    (cache.bond_de + (0.0 - cache.term_before_o)) + (term_after_k - 0.0)
                } else {
                    (cache.bond_de + (term_after_k - 0.0)) + (0.0 - cache.term_before_o)
                };
                return (cache.raw, de);
            }
            let (mut raw, bond_de) = self.bond_raw_and_change(s, mv);
            let before = View {
                state: s,
                overrides: &[],
            };
            let term_before_o = match before.get(o) {
                Content::Dumbbell { axis, plus, minus } => {
                    self.dumbbell_term(&before, o, axis, plus, minus)
                }
                _ => unreachable!("active dumbbell record missing dumbbell content"),
            };
            raw += 0.0 - term_before_o;
            raw = self.add_saddle_interactions(s, mv, raw);
            s.branch_cache.set(DumbbellBranchCache {
                model_uid: self.model_uid,
                version: s.version,
                rec,
                branch,
                raw,
                bond_de,
                term_before_o,
            });
            let ov = mv.final_overrides();
            let after = View {
                state: s,
                overrides: &ov,
            };
            let term_after_k = self.dumbbell_term(&after, k, axis, plus, minus);
            let de = if o < k {
                (bond_de + (0.0 - term_before_o)) + (term_after_k - 0.0)
            } else {
                (bond_de + (term_after_k - 0.0)) + (0.0 - term_before_o)
            };
            return (raw, de);
        }

        let (mut raw, mut de) = self.bond_raw_and_change(s, mv);
        let has_override_dumbbell =
            matches!(o0, Content::Dumbbell { .. }) || matches!(o1, Content::Dumbbell { .. });
        if s.live_dumbbells > 0 || has_override_dumbbell {
            let before = View {
                state: s,
                overrides: &[],
            };
            let saddle_ov = mv.saddle_overrides();
            let after_raw = View {
                state: s,
                overrides: &saddle_ov,
            };
            let final_ov = mv.final_overrides();
            let after_chg = View {
                state: s,
                overrides: &final_ov,
            };
            let mut holders = [0_usize; 18];
            holders[0] = c0;
            holders[1..9].copy_from_slice(self.lattice.neighbors(c0));
            holders[9] = c1;
            holders[10..18].copy_from_slice(self.lattice.neighbors(c1));
            for i in 1..18 {
                let x = holders[i];
                let mut j = i;
                while j > 0 && holders[j - 1] > x {
                    holders[j] = holders[j - 1];
                    j -= 1;
                }
                holders[j] = x;
            }
            let mut prev = usize::MAX;
            for &x in &holders {
                if x == prev {
                    continue;
                }
                prev = x;
                if x != c0 && x != c1 && s.cells[x] != Cell::Dumbbell {
                    continue;
                }
                let term = |view: &View| match view.get(x) {
                    Content::Dumbbell { axis, plus, minus } => {
                        self.dumbbell_term(view, x, axis, plus, minus)
                    }
                    _ => 0.0,
                };
                let tb = term(&before);
                raw += term(&after_raw) - tb;
                de += term(&after_chg) - tb;
            }
        }
        raw = self.add_saddle_interactions(s, mv, raw);
        (raw, de)
    }

    /// The 6 sites in `1NN(a) ∩ 2NN(b)` then `2NN(a) ∩ 1NN(b)` for jump direction `dir` (`a -> b`).
    #[inline]
    fn saddle_sites(&self, a: usize, dir: usize) -> [usize; 6] {
        let (nb_idx, sn_idx) = SADDLE_SITE_INDICES[dir];
        let nb = self.lattice.neighbors(a);
        let sn = self.lattice.second_neighbors(a);
        [
            nb[nb_idx[0] as usize],
            nb[nb_idx[1] as usize],
            nb[nb_idx[2] as usize],
            sn[sn_idx[0] as usize],
            sn[sn_idx[1] as usize],
            sn[sn_idx[2] as usize],
        ]
    }

    /// Resolves jump `jump` (0..8) of record `rec`; `None` for invalid indices, an empty record, or a blocked target.
    fn jump_move(&self, s: &FeCrRisState, rec: usize, jump: usize) -> Option<Move> {
        if jump >= 8 {
            return None;
        }
        let lattice_species = |site: usize| match s.cells[site] {
            Cell::Fe => Some(Species::Fe),
            Cell::Cr => Some(Species::Cr),
            _ => None,
        };
        match *s.records.get(rec)? {
            Record::Empty => None,
            Record::Vacancy { site: q } => {
                let p = self.lattice.neighbor(q, jump);
                Some(Move::Vacancy {
                    q,
                    p,
                    dir: jump as u8,
                    atom: lattice_species(p)?,
                })
            }
            Record::Dumbbell {
                site: o,
                axis,
                plus,
                minus,
            } => {
                let spec = DUMBBELL_JUMPS[axis][jump];
                let k = self.lattice.neighbor(o, spec.target_dir as usize);
                let atom = lattice_species(k)?;
                let (jumper, rest) = if spec.jumper_is_plus {
                    (plus, minus)
                } else {
                    (minus, plus)
                };
                let (new_plus, new_minus) = if spec.jumper_is_new_plus {
                    (jumper, atom)
                } else {
                    (atom, jumper)
                };
                Some(Move::Dumbbell {
                    o,
                    k,
                    dir: spec.target_dir,
                    axis: spec.new_axis as usize,
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

    /// E_saddle − E_initial in eV without the floor; NaN if there is no such jump.
    pub fn raw_barrier(&self, s: &FeCrRisState, rec: usize, jump: usize) -> f64 {
        self.jump_move(s, rec, jump)
            .map_or(f64::NAN, |mv| self.raw_and_change(s, rec, jump, &mv).0)
    }

    /// E_final − E_initial in eV; NaN if there is no such jump.
    pub fn energy_change(&self, s: &FeCrRisState, rec: usize, jump: usize) -> f64 {
        self.jump_move(s, rec, jump)
            .map_or(f64::NAN, |mv| self.raw_and_change(s, rec, jump, &mv).1)
    }

    /// The barrier `max(raw, max(0, ΔE))` in eV; NaN if there is no such jump.
    pub fn barrier(&self, s: &FeCrRisState, rec: usize, jump: usize) -> f64 {
        self.jump_move(s, rec, jump).map_or(f64::NAN, |mv| {
            let (raw, de) = self.raw_and_change(s, rec, jump, &mv);
            raw.max(de.max(0.0))
        })
    }

    /// The jump rate in s⁻¹; 0 for an empty record or a blocked jump.
    ///
    /// Dumbbell jumps allocate `0.5 * nu0 * exp(ds_mig_kb) * exp(-barrier / kT)` to each of the
    /// two symmetry-equivalent 60° final orientations per compression-site target (`z_eff = 4`),
    /// preserving exact detailed balance while matching Senninger et al. Eq. (3) per neighbor branch.
    pub fn rate(&self, s: &FeCrRisState, rec: usize, jump: usize) -> f64 {
        let Some(mv) = self.jump_move(s, rec, jump) else {
            return 0.0;
        };
        let (raw, de) = self.raw_and_change(s, rec, jump, &mv);
        let floor = de.max(0.0);
        let floored = raw < floor;
        let mut fc = s.floor_cache.get();
        if fc.model_uid != self.model_uid || fc.version != s.version || fc.rec != rec {
            fc = RecordFloorCache {
                model_uid: self.model_uid,
                version: s.version,
                rec,
                valid_mask: 0,
                floored_mask: 0,
            };
        }
        let bit = 1_u8 << jump;
        fc.valid_mask |= bit;
        if floored {
            fc.floored_mask |= bit;
        } else {
            fc.floored_mask &= !bit;
        }
        s.floor_cache.set(fc);

        let barrier = raw.max(floor);
        let boltzmann = if self.kt == 0.0 {
            f64::from(u8::from(barrier <= 0.0))
        } else {
            (-barrier / self.kt).exp()
        };
        match mv {
            Move::Dumbbell { .. } => 0.5 * self.dumbbell_prefactor * boltzmann,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeCrRisEvent {
    /// Vacancy record `rec` jumps in [`DISPLACEMENTS`] direction `dir` (`0..8`).
    VacancyJump {
        rec: usize,
        dir: usize,
    },
    /// Dumbbell record `rec` makes jump `jump` (`0..8`).
    DumbbellJump {
        rec: usize,
        jump: usize,
    },
    Create,
}

#[derive(Clone, Copy, Debug)]
enum ReservoirUndo {
    None,
    PoppedFront(Species),
    PushedBack,
}

/// Zero-allocation `O(1)` rollback token for a single [`FeCrRis`] transition.
#[derive(Clone, Debug)]
pub struct FeCrRisUndo {
    cells: [(usize, Cell, u32); 16],
    cells_len: u8,
    records: [(usize, Record); 6],
    records_len: u8,
    last_changed: [usize; 16],
    last_changed_len: u8,
    last_records: [usize; 6],
    last_records_len: u8,
    live_vacancies: usize,
    live_dumbbells: usize,
    reservoir: ReservoirUndo,
    borrowed_fe: u64,
    created: u64,
    recombined: u64,
    absorbed_v: u64,
    absorbed_i: u64,
    creation_attempts: u64,
    floored_dumbbell_jumps: u64,
    floored_vacancy_jumps: u64,
    peak_defects: u64,
    creation_rng: Option<ChaCha12Rng>,
}

impl FeCrRisUndo {
    fn capture(s: &FeCrRisState, save_rng: bool) -> Self {
        let mut last_changed = [0_usize; 16];
        let changed_len = s.last_changed.len();
        last_changed[..changed_len].copy_from_slice(&s.last_changed);
        let mut last_records = [0_usize; 6];
        let rec_len = s.last_records.len();
        last_records[..rec_len].copy_from_slice(&s.last_records);
        Self {
            cells: [(0, Cell::Fe, NO_RECORD); 16],
            cells_len: 0,
            records: [(0, Record::Empty); 6],
            records_len: 0,
            last_changed,
            last_changed_len: changed_len as u8,
            last_records,
            last_records_len: rec_len as u8,
            live_vacancies: s.live_vacancies,
            live_dumbbells: s.live_dumbbells,
            reservoir: ReservoirUndo::None,
            borrowed_fe: s.borrowed_fe,
            created: s.created,
            recombined: s.recombined,
            absorbed_v: s.absorbed_v,
            absorbed_i: s.absorbed_i,
            creation_attempts: s.creation_attempts,
            floored_dumbbell_jumps: s.floored_dumbbell_jumps,
            floored_vacancy_jumps: s.floored_vacancy_jumps,
            peak_defects: s.peak_defects,
            creation_rng: save_rng.then(|| s.creation_rng.clone()),
        }
    }

    fn save_site(&mut self, s: &FeCrRisState, site: usize) {
        let n = self.cells_len as usize;
        if !self.cells[..n].iter().any(|&(x, _, _)| x == site) {
            self.cells[n] = (site, s.cells[site], s.site_record[site]);
            self.cells_len += 1;
        }
    }

    fn save_record(&mut self, s: &FeCrRisState, rec: usize) {
        let n = self.records_len as usize;
        if !self.records[..n].iter().any(|&(r, _)| r == rec) {
            self.records[n] = (rec, s.records[rec]);
            self.records_len += 1;
        }
    }
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

    fn free_record(s: &mut FeCrRisState, rec: usize, mut undo: Option<&mut FeCrRisUndo>) {
        let was_active = !matches!(s.records[rec], Record::Empty);
        match s.records[rec] {
            Record::Empty => {}
            Record::Vacancy { .. } => s.live_vacancies -= 1,
            Record::Dumbbell { .. } => s.live_dumbbells -= 1,
        }
        if let Record::Vacancy { site } | Record::Dumbbell { site, .. } = s.records[rec] {
            if let Some(u) = undo.as_deref_mut() {
                u.save_site(s, site);
            }
            s.site_record[site] = NO_RECORD;
        }
        if let Some(u) = undo {
            u.save_record(s, rec);
        }
        s.records[rec] = Record::Empty;
        if was_active {
            s.deactivate_record(rec);
        }
        if !s.last_records.contains(&rec) {
            s.last_records.push(rec);
        }
    }

    /// Recombines record `rec` with its nearest opposite defect within [`RC_OFFSETS`], if any.
    fn recombine(
        &self,
        s: &mut FeCrRisState,
        rec: usize,
        mut undo: Option<&mut FeCrRisUndo>,
    ) -> bool {
        let (site, is_dumbbell) = match s.records[rec] {
            Record::Empty => return false,
            Record::Vacancy { site } => {
                if s.live_dumbbells == 0 {
                    return false;
                }
                (site, false)
            }
            Record::Dumbbell { site, .. } => {
                if s.live_vacancies == 0 {
                    return false;
                }
                (site, true)
            }
        };
        let pos = self.lattice.position(site);
        let [lx, ly, lz] = self.lattice.cells();
        let m = [2 * lx as i64, 2 * ly as i64, 2 * lz as i64];
        let mut best: Option<(i64, usize, [i64; 3])> = None;
        if s.active_records.len() < 16 {
            for &r in &s.active_records {
                let other_site = match (s.records[r], is_dumbbell) {
                    (Record::Vacancy { site: vs }, true) => vs,
                    (Record::Dumbbell { site: ds, .. }, false) => ds,
                    _ => continue,
                };
                let opos = self.lattice.position(other_site);
                let o = [
                    shortest_delta(opos[0] - pos[0], m[0]),
                    shortest_delta(opos[1] - pos[1], m[1]),
                    shortest_delta(opos[2] - pos[2], m[2]),
                ];
                let r2 = dot(o, o);
                if matches!(r2, 3 | 4 | 8 | 11)
                    && best.is_none_or(|(br2, br, _)| (r2, r) < (br2, br))
                {
                    best = Some((r2, r, o));
                }
            }
        } else {
            for o in RC_OFFSETS.iter() {
                let other = self.wrapped_site_at([
                    wrap_coord(pos[0] + o[0], m[0]),
                    wrap_coord(pos[1] + o[1], m[1]),
                    wrap_coord(pos[2] + o[2], m[2]),
                ]);
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
        if let Some(u) = undo.as_deref_mut() {
            u.save_site(s, v_site);
            u.save_site(s, i_site);
        }
        s.cells[v_site] = cell_of(fills);
        s.cells[i_site] = cell_of(stays);
        let wf = i8::from(fills == Species::Cr);
        s.add_cr_weight(&self.lattice, v_site, wf);
        s.add_cr_weight(&self.lattice, i_site, -wf);
        s.last_changed.extend([v_site, i_site]);
        Self::free_record(s, i_rec, undo.as_deref_mut());
        Self::free_record(s, v_rec, undo);
        s.recombined += 1;
        true
    }

    fn build_chain(&self, start: usize, dir: usize) -> [usize; 11] {
        let mut cur = start;
        std::array::from_fn(|_| {
            let s = cur;
            cur = self.lattice.neighbor(cur, dir);
            s
        })
    }

    fn create(
        &self,
        s: &mut FeCrRisState,
        mut undo: Option<&mut FeCrRisUndo>,
    ) -> Result<(), FeCrRisError> {
        let mut empty_iter =
            (0..s.records.len()).filter(|&r| matches!(s.records[r], Record::Empty));
        let (Some(v_rec), Some(d_rec)) = (empty_iter.next(), empty_iter.next()) else {
            return Err(FeCrRisError::PoolExhausted {
                capacity: self.params.capacity,
                defects: s.open_records() as usize,
            });
        };
        s.active_records
            .try_reserve(2)
            .map_err(|_| FeCrRisError::Parameter("state allocation exceeds memory limit"))?;
        let n = self.lattice.site_count();
        let draw = |rng: &mut ChaCha12Rng, m: usize| ((open01(rng) * m as f64) as usize).min(m - 1);
        let mut rng = s.creation_rng.clone();
        let mut attempts = 0_u64;
        let mut found = None;
        for _ in 0..1000 {
            let site = draw(&mut rng, n);
            let dir = draw(&mut rng, 8);
            let orient = draw(&mut rng, 12);
            attempts += 1;
            let chain = self.build_chain(site, dir);
            if chain
                .iter()
                .all(|&c| matches!(s.cells[c], Cell::Fe | Cell::Cr) && !self.on_sink(c))
            {
                found = Some((chain, orient));
                break;
            }
        }
        let (chain, orient) = match found {
            Some(pair) => pair,
            None => {
                let mut valid = Vec::new();
                for site in 0..n {
                    for dir in 0..8 {
                        let chain = self.build_chain(site, dir);
                        if chain
                            .iter()
                            .all(|&c| matches!(s.cells[c], Cell::Fe | Cell::Cr) && !self.on_sink(c))
                        {
                            valid.push(chain);
                        }
                    }
                }
                if valid.is_empty() {
                    return Err(FeCrRisError::CreationFailed);
                }
                let chain = valid[draw(&mut rng, valid.len())];
                let orient = draw(&mut rng, 12);
                attempts += 1;
                (chain, orient)
            }
        };
        if let Some(u) = undo.as_deref_mut() {
            for &c in &chain {
                u.save_site(s, c);
            }
            u.save_record(s, v_rec);
            u.save_record(s, d_rec);
        }
        s.creation_rng = rng;
        s.creation_attempts += attempts;
        s.last_changed.clear();
        s.last_records.clear();
        let atoms: [Species; 11] = std::array::from_fn(|i| Self::site_species(s, chain[i]));
        s.cells[chain[0]] = Cell::Vacancy;
        s.add_cr_weight(&self.lattice, chain[0], -i8::from(atoms[0] == Species::Cr));
        for i in 1..=9 {
            s.cells[chain[i]] = cell_of(atoms[i - 1]);
            let dw = i8::from(atoms[i - 1] == Species::Cr) - i8::from(atoms[i] == Species::Cr);
            s.add_cr_weight(&self.lattice, chain[i], dw);
        }
        s.cells[chain[10]] = Cell::Dumbbell;
        let (plus, minus) = if orient % 2 == 0 {
            (atoms[9], atoms[10])
        } else {
            (atoms[10], atoms[9])
        };
        s.records[v_rec] = Record::Vacancy { site: chain[0] };
        s.records[d_rec] = Record::Dumbbell {
            site: chain[10],
            axis: orient / 2,
            plus,
            minus,
        };
        s.add_cr_weight(&self.lattice, chain[10], i8::from(atoms[9] == Species::Cr));
        s.live_vacancies += 1;
        s.live_dumbbells += 1;
        s.site_record[chain[0]] = v_rec as u32;
        s.site_record[chain[10]] = d_rec as u32;
        s.activate_record(v_rec);
        s.activate_record(d_rec);
        s.created += 1;
        s.last_changed.extend_from_slice(&chain);
        s.last_records.extend([v_rec, d_rec]);
        self.recombine(s, v_rec, undo.as_deref_mut());
        self.recombine(s, d_rec, undo);
        Ok(())
    }

    /// Applies jump `jump` of record `rec`; I–V recombination runs before and takes precedence
    /// over sink absorption when the destination sits on a sink plane within [`RC_OFFSETS`].
    fn jump(
        &self,
        s: &mut FeCrRisState,
        rec: usize,
        jump: usize,
        vacancy: bool,
        mut undo: Option<&mut FeCrRisUndo>,
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
        let fc = s.floor_cache.get();
        let bit = 1_u8 << jump;
        let floored = if fc.model_uid == self.model_uid
            && fc.version == s.version
            && fc.rec == rec
            && (fc.valid_mask & bit) != 0
        {
            (fc.floored_mask & bit) != 0
        } else {
            let (raw, de) = self.raw_and_change(s, rec, jump, &mv);
            raw < de.max(0.0)
        };
        let (a, b) = mv.ends();
        if let Some(u) = undo.as_deref_mut() {
            u.save_site(s, a);
            u.save_site(s, b);
            u.save_record(s, rec);
        }
        s.last_changed.clear();
        s.last_records.clear();
        s.last_records.push(rec);
        s.last_changed.extend([a, b]);
        s.site_record[a] = NO_RECORD;
        s.site_record[b] = rec as u32;
        match mv {
            Move::Vacancy { q, p, atom, .. } => {
                s.floored_vacancy_jumps += u64::from(floored);
                s.cells[q] = cell_of(atom);
                s.cells[p] = Cell::Vacancy;
                s.records[rec] = Record::Vacancy { site: p };
                let w = i8::from(atom == Species::Cr);
                s.add_cr_weight(&self.lattice, q, w);
                s.add_cr_weight(&self.lattice, p, -w);
                if !self.recombine(s, rec, undo.as_deref_mut()) && self.on_sink(p) {
                    let fill = match s.reservoir.pop_front() {
                        Some(sp) => {
                            if let Some(u) = undo.as_deref_mut() {
                                u.reservoir = ReservoirUndo::PoppedFront(sp);
                            }
                            sp
                        }
                        None => {
                            s.borrowed_fe += 1;
                            Species::Fe
                        }
                    };
                    s.cells[p] = cell_of(fill);
                    s.add_cr_weight(&self.lattice, p, i8::from(fill == Species::Cr));
                    Self::free_record(s, rec, undo);
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
                ..
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
                let wj = i8::from(jumper == Species::Cr);
                s.add_cr_weight(&self.lattice, o, -wj);
                s.add_cr_weight(&self.lattice, k, wj);
                if !self.recombine(s, rec, undo.as_deref_mut()) && self.on_sink(k) {
                    s.cells[k] = cell_of(jumper);
                    s.add_cr_weight(&self.lattice, k, -i8::from(atom == Species::Cr));
                    s.reservoir.push_back(atom);
                    if let Some(u) = undo.as_deref_mut() {
                        u.reservoir = ReservoirUndo::PushedBack;
                    }
                    Self::free_record(s, rec, undo);
                    s.absorbed_i += 1;
                }
            }
        }
        Ok(())
    }

    fn apply_inner(
        &self,
        state: &mut FeCrRisState,
        event: &FeCrRisEvent,
        undo: Option<&mut FeCrRisUndo>,
    ) -> Result<(), FeCrRisError> {
        match *event {
            FeCrRisEvent::VacancyJump { rec, dir } => self.jump(state, rec, dir, true, undo)?,
            FeCrRisEvent::DumbbellJump { rec, jump } => self.jump(state, rec, jump, false, undo)?,
            FeCrRisEvent::Create => self.create(state, undo)?,
        }
        state.peak_defects = state.peak_defects.max(state.open_records());
        state.bump_version();
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
                if rate != 0.0 {
                    emit(event, rate);
                }
            }
        }
        if self.params.k0_dpa_per_s != 0.0 {
            emit(FeCrRisEvent::Create, self.creation_rate());
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        state.last_changed.clear();
        state.last_records.clear();
        self.apply_inner(state, event, None)
    }
}

impl ReversibleModel for FeCrRis {
    type Undo = FeCrRisUndo;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        let mut undo = FeCrRisUndo::capture(state, *event == FeCrRisEvent::Create);
        self.apply_inner(state, event, Some(&mut undo))?;
        Ok(undo)
    }

    fn undo(&self, state: &mut Self::State, undo: Self::Undo) {
        let n_cells = undo.cells_len as usize;
        let mut old_w = [0_u8; 16];
        for (idx, &(site, _, _)) in undo.cells[..n_cells].iter().enumerate() {
            old_w[idx] = state.cr_weight_at(site) as u8;
        }
        for &(rec, record) in &undo.records[..undo.records_len as usize] {
            let was_active = !matches!(state.records[rec], Record::Empty);
            let now_active = !matches!(record, Record::Empty);
            state.records[rec] = record;
            match (was_active, now_active) {
                (false, true) => state.activate_record(rec),
                (true, false) => state.deactivate_record(rec),
                _ => {}
            }
        }
        for &(site, cell, rec) in &undo.cells[..n_cells] {
            state.cells[site] = cell;
            state.site_record[site] = rec;
        }
        for (idx, &(site, _, _)) in undo.cells[..n_cells].iter().enumerate() {
            let new_w = state.cr_weight_at(site) as u8;
            state.add_cr_weight(&self.lattice, site, new_w as i8 - old_w[idx] as i8);
        }
        match undo.reservoir {
            ReservoirUndo::None => {}
            ReservoirUndo::PoppedFront(sp) => state.reservoir.push_front(sp),
            ReservoirUndo::PushedBack => {
                state.reservoir.pop_back();
            }
        }
        state.borrowed_fe = undo.borrowed_fe;
        state.created = undo.created;
        state.recombined = undo.recombined;
        state.absorbed_v = undo.absorbed_v;
        state.absorbed_i = undo.absorbed_i;
        state.creation_attempts = undo.creation_attempts;
        state.floored_dumbbell_jumps = undo.floored_dumbbell_jumps;
        state.floored_vacancy_jumps = undo.floored_vacancy_jumps;
        state.peak_defects = undo.peak_defects;
        state.live_vacancies = undo.live_vacancies;
        state.live_dumbbells = undo.live_dumbbells;
        if let Some(rng) = undo.creation_rng {
            state.creation_rng = rng;
        }
        state.last_changed.clear();
        state
            .last_changed
            .extend_from_slice(&undo.last_changed[..undo.last_changed_len as usize]);
        state.last_records.clear();
        state
            .last_records
            .extend_from_slice(&undo.last_records[..undo.last_records_len as usize]);
        state.bump_version();
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

#[inline]
fn influence_lut_index(o: [i64; 3]) -> usize {
    (((o[0] + 7) * 15 + (o[1] + 7)) * 15 + (o[2] + 7)) as usize
}

static INFLUENCE_LUT: LazyLock<[bool; 15 * 15 * 15]> = LazyLock::new(|| {
    let mut lut = [false; 15 * 15 * 15];
    for &o in INFLUENCE_OFFSETS.iter() {
        assert!(
            (-7..=7).contains(&o[0]) && (-7..=7).contains(&o[1]) && (-7..=7).contains(&o[2]),
            "influence offset {o:?} exceeds [-7..=7]^3"
        );
        lut[influence_lut_index(o)] = true;
    }
    lut
});

#[inline]
fn wrap_coord(x: i64, m: i64) -> i64 {
    if x < 0 {
        x + m
    } else if x >= m {
        x - m
    } else {
        x
    }
}

#[inline]
fn shortest_delta(d: i64, m: i64) -> i64 {
    let half = m / 2;
    if d > half {
        d - m
    } else if d <= -half {
        d + m
    } else {
        d
    }
}

#[inline]
fn axis_candidates(d: i64, m: i64) -> ([i64; 2], usize) {
    let mut vals = [0_i64; 2];
    let mut len = 0;
    if (-7..=7).contains(&d) {
        vals[len] = d;
        len += 1;
    }
    if d >= m - 7 {
        vals[len] = d - m;
        len += 1;
    } else if d <= 7 - m {
        vals[len] = d + m;
        len += 1;
    }
    (vals, len)
}

#[inline]
fn is_in_influence(cp: [i64; 3], rp: [i64; 3], m: [i64; 3]) -> bool {
    let d0 = shortest_delta(cp[0] - rp[0], m[0]);
    let (v0, l0) = axis_candidates(d0, m[0]);
    if l0 == 0 {
        return false;
    }
    let d1 = shortest_delta(cp[1] - rp[1], m[1]);
    let (v1, l1) = axis_candidates(d1, m[1]);
    if l1 == 0 {
        return false;
    }
    let d2 = shortest_delta(cp[2] - rp[2], m[2]);
    let (v2, l2) = axis_candidates(d2, m[2]);
    if l2 == 0 {
        return false;
    }
    let lut = &*INFLUENCE_LUT;
    for &x in &v0[..l0] {
        for &y in &v1[..l1] {
            for &z in &v2[..l2] {
                if lut[influence_lut_index([x, y, z])] {
                    return true;
                }
            }
        }
    }
    false
}

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
        if !state.active_records.is_empty() {
            let [lx, ly, lz] = self.lattice.cells();
            let m = [2 * lx as i64, 2 * ly as i64, 2 * lz as i64];
            if state.active_records.len() < 128 && state.last_changed.len() <= 16 {
                let mut changed_pos = [[0_i64; 3]; 16];
                let n_changed = state.last_changed.len();
                for (i, &c) in state.last_changed.iter().enumerate() {
                    changed_pos[i] = self.lattice.position(c);
                }
                for &r in &state.active_records {
                    if records.contains(&r) {
                        continue;
                    }
                    let r_site = match state.records[r] {
                        Record::Vacancy { site } | Record::Dumbbell { site, .. } => site,
                        Record::Empty => continue,
                    };
                    let rp = self.lattice.position(r_site);
                    if changed_pos[..n_changed]
                        .iter()
                        .any(|&cp| is_in_influence(cp, rp, m))
                    {
                        records.push(r);
                    }
                }
            } else {
                for &c in &state.last_changed {
                    let p = self.lattice.position(c);
                    for o in INFLUENCE_OFFSETS.iter() {
                        // Offsets are at most 7 in each component and cells are at least 6, so a
                        // single wrap suffices. The offsets keep the sublattice parity.
                        let site = self.wrapped_site_at([
                            wrap_coord(p[0] - o[0], m[0]),
                            wrap_coord(p[1] - o[1], m[1]),
                            wrap_coord(p[2] - o[2], m[2]),
                        ]);
                        if let Some(r) = state.record_at(site) {
                            records.push(r);
                        }
                    }
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
        let (live_vacancies, live_dumbbells) = live_counts(&records);
        let (cr_n, active_records, active_pos) =
            FeCrRisState::derive_tables(&self.lattice, &cells, &records)
                .map_err(|e| e.to_string())?;
        let state = FeCrRisState {
            cells,
            records,
            site_record,
            cr_n,
            active_records,
            active_pos,
            live_vacancies,
            live_dumbbells,
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
            version: 1,
            branch_cache: StdCell::new(DumbbellBranchCache::empty()),
            floor_cache: StdCell::new(RecordFloorCache::empty()),
        };
        self.check_invariants(&state).map_err(|e| e.to_string())?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AXES, BccLattice, Checkpointable, DISPLACEMENTS, DUMBBELL_JUMPS, DUMBBELL_SITE_TYPES,
        Defect, DumbbellParams, FeCrRis, FeCrRisError, FeCrRisEvent, FeCrRisParams, FeCrRisState,
        INFLUENCE_LUT, INFLUENCE_OFFSETS, IncrementalModel, Model, RC_OFFSETS, ReversibleModel,
        SADDLE_SITE_INDICES, Species, creation_rng, dot, dumbbell_kind, influence_lut_index,
        open01, site_type,
    };
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

    fn snapshot(m: &FeCrRis, s: &FeCrRisState) -> (Vec<u8>, Vec<usize>, Vec<usize>, Vec<u64>) {
        let mut bytes = Vec::new();
        m.encode_state(s, &mut bytes);
        let rates: Vec<u64> = (0..m.slot_count(s))
            .map(|slot| m.slot_rate(s, slot).to_bits())
            .collect();
        (
            bytes,
            s.last_changed().to_vec(),
            s.last_records().to_vec(),
            rates,
        )
    }

    #[test]
    fn reversible_apply_and_undo_restores_state_bitwise() {
        let model = FeCrRis::new(FeCrRisParams {
            cells: [6, 6, 6],
            sink_planes: vec![0],
            temperature_k: 773.0,
            k0_dpa_per_s: 1.0e8,
            capacity: 16,
            dumbbell: DumbbellParams::senninger2016_r1(),
        })
        .unwrap();
        let mut state = model.random_state(43, 7, 7).unwrap();
        let mut rng = KmcRng::from_seed(11);

        for _ in 0..80 {
            let mut candidates = Vec::new();
            model.enumerate_events(&state, &mut |e, r| candidates.push((e, r)));
            for (ev, _) in &candidates {
                let before = snapshot(&model, &state);
                let mut expected = state.clone();
                model.apply(&mut expected, ev).unwrap();
                let undo = model.apply_reversible(&mut state, ev).unwrap();
                assert_eq!(snapshot(&model, &state), snapshot(&model, &expected));
                model.undo(&mut state, undo);
                model.check_invariants(&state).unwrap();
                assert_eq!(snapshot(&model, &state), before);
            }
            let idx = ((rng.open01() * candidates.len() as f64) as usize).min(candidates.len() - 1);
            model.apply(&mut state, &candidates[idx].0).unwrap();
        }
        assert!(state.created() > 0 && (state.recombined() > 0 || state.absorbed_i() > 0));

        let before_bad = snapshot(&model, &state);
        assert!(matches!(
            model.apply_reversible(&mut state, &FeCrRisEvent::VacancyJump { rec: 99, dir: 0 }),
            Err(FeCrRisError::NoSuchJump { .. })
        ));
        assert_eq!(snapshot(&model, &state), before_bad);

        let all_sinks = FeCrRis::new(FeCrRisParams {
            cells: [6, 6, 6],
            sink_planes: (0..12).collect(),
            temperature_k: 773.0,
            k0_dpa_per_s: 1.0,
            capacity: 4,
            dumbbell: DumbbellParams::senninger2016_r1(),
        })
        .unwrap();
        let mut blocked = all_sinks.random_state(10, 1, 1).unwrap();
        let before_blocked = snapshot(&all_sinks, &blocked);
        assert_eq!(
            all_sinks
                .apply_reversible(&mut blocked, &FeCrRisEvent::Create)
                .unwrap_err(),
            FeCrRisError::CreationFailed
        );
        assert_eq!(snapshot(&all_sinks, &blocked), before_blocked);
    }

    #[test]
    fn create_falls_back_exhaustively_when_single_chain_remains() {
        let sparse = FeCrRis::new(FeCrRisParams {
            cells: [6, 6, 6],
            sink_planes: vec![],
            temperature_k: 773.0,
            k0_dpa_per_s: 1.0,
            capacity: 432,
            dumbbell: DumbbellParams::senninger2016_r1(),
        })
        .unwrap();
        let n = sparse.lattice().site_count();
        let keep = sparse.build_chain(0, 0);
        let blocker_defects: Vec<super::Defect> = (0..n)
            .filter(|s| !keep.contains(s))
            .map(|site| super::Defect::Vacancy { site })
            .collect();
        let occ = vec![super::Species::Fe; n];
        let mut saw_fallback = false;
        for run_seed in 0..12 {
            let mut sparse_state = sparse
                .state_with_defects(occ.clone(), &blocker_defects, run_seed)
                .unwrap();
            let before_sparse = snapshot(&sparse, &sparse_state);
            let mut expected = sparse_state.clone();
            sparse.apply(&mut expected, &FeCrRisEvent::Create).unwrap();
            let undo = sparse
                .apply_reversible(&mut sparse_state, &FeCrRisEvent::Create)
                .unwrap();
            assert_eq!(
                snapshot(&sparse, &sparse_state),
                snapshot(&sparse, &expected)
            );
            assert_eq!(sparse_state.created(), 1);
            if sparse_state.creation_attempts() == 1001 {
                saw_fallback = true;
            }
            sparse.undo(&mut sparse_state, undo);
            assert_eq!(snapshot(&sparse, &sparse_state), before_sparse);
        }
        assert!(saw_fallback);
    }

    #[test]
    fn lookup_tables_match_reference() {
        for (axis, &u) in AXES.iter().enumerate() {
            for (jump, &spec) in DUMBBELL_JUMPS[axis].iter().enumerate() {
                let (target_dir, d) = DISPLACEMENTS
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| dot(**d, u).abs() == 2)
                    .nth(jump / 2)
                    .unwrap();
                let back = [-d[0], -d[1], -d[2]];
                let new_axis = (0..6)
                    .filter(|&b| dot(back, AXES[b]).abs() == 2 && dot(u, AXES[b]).abs() == 1)
                    .nth(jump % 2)
                    .unwrap();
                assert_eq!(spec.target_dir as usize, target_dir);
                assert_eq!(spec.jumper_is_plus, dot(*d, u) > 0);
                assert_eq!(spec.new_axis as usize, new_axis);
                assert_eq!(spec.jumper_is_new_plus, dot(back, AXES[new_axis]) > 0);
            }
        }

        assert_eq!(dumbbell_kind(Species::Fe, Species::Fe), 0);
        assert_eq!(dumbbell_kind(Species::Cr, Species::Cr), 0);
        assert_eq!(dumbbell_kind(Species::Fe, Species::Cr), 1);
        assert_eq!(dumbbell_kind(Species::Cr, Species::Fe), 2);
        let pairs = [
            (Species::Fe, Species::Fe),
            (Species::Fe, Species::Cr),
            (Species::Cr, Species::Fe),
        ];
        for (axis, by_kind) in DUMBBELL_SITE_TYPES.iter().enumerate() {
            for (kind, &(plus, minus)) in pairs.iter().enumerate() {
                for (dir, &d) in DISPLACEMENTS.iter().enumerate() {
                    assert_eq!(by_kind[kind][dir] as usize, site_type(axis, plus, minus, d));
                }
            }
        }

        let lat = BccLattice::new([6, 6, 6]).unwrap();
        for (dir, &(nb_idx, sn_idx)) in SADDLE_SITE_INDICES.iter().enumerate() {
            let b = lat.neighbor(0, dir);
            let nb0 = lat.neighbors(0);
            let sn0 = lat.second_neighbors(0);
            let expected_nb: Vec<u8> = nb0
                .iter()
                .enumerate()
                .filter(|&(_, &s)| lat.second_neighbors(b).contains(&s))
                .map(|(i, _)| i as u8)
                .collect();
            let expected_sn: Vec<u8> = sn0
                .iter()
                .enumerate()
                .filter(|&(_, &s)| lat.neighbors(b).contains(&s))
                .map(|(i, _)| i as u8)
                .collect();
            assert_eq!(nb_idx.to_vec(), expected_nb);
            assert_eq!(sn_idx.to_vec(), expected_sn);
        }

        let mut true_count = 0_usize;
        for x in -7..=7 {
            for y in -7..=7 {
                for z in -7..=7 {
                    let o = [x, y, z];
                    let actual = INFLUENCE_LUT[influence_lut_index(o)];
                    assert_eq!(actual, INFLUENCE_OFFSETS.contains(&o), "offset {o:?}");
                    true_count += usize::from(actual);
                }
            }
        }
        assert_eq!(true_count, INFLUENCE_OFFSETS.len());
    }

    #[test]
    fn branch_and_floor_caches_respect_model_uid_and_version() {
        let m1 = FeCrRis::new(FeCrRisParams {
            cells: [6, 6, 6],
            sink_planes: vec![],
            temperature_k: 773.0,
            k0_dpa_per_s: 0.0,
            capacity: 4,
            dumbbell: DumbbellParams::senninger2016_r1(),
        })
        .unwrap();
        let m2 = m1.clone().with_dumbbell_saddle_shift(-0.5);

        let lat = m1.lattice();
        let mut occ = vec![Species::Fe; lat.site_count()];
        let center = lat.site_at([4, 4, 4]).unwrap();
        for &n in &lat.neighbors(center)[..3] {
            occ[n] = Species::Cr;
        }
        for &n in &lat.second_neighbors(center)[..2] {
            occ[n] = Species::Cr;
        }
        let defects = [Defect::Dumbbell {
            site: center,
            axis: 0,
            plus: Species::Fe,
            minus: Species::Cr,
        }];
        let mut s = m1.state_with_defects(occ, &defects, 1).unwrap();
        let s_clean = s.clone();

        assert!(
            m1.raw_barrier(&s_clean, 0, 0) >= m1.energy_change(&s_clean, 0, 0).max(0.0),
            "jump 0 should not be floored under m1"
        );
        assert!(
            m2.raw_barrier(&s_clean, 0, 0) < m2.energy_change(&s_clean, 0, 0).max(0.0),
            "jump 0 should be floored under m2"
        );

        let mut saw_rate_diff = false;
        for j in 0..8 {
            let s_ref = s_clean.clone();
            let _ = m1.rate(&s, 0, j);
            let raw2 = m2.raw_barrier(&s, 0, j);
            assert_eq!(raw2.to_bits(), m2.raw_barrier(&s_ref, 0, j).to_bits());
            assert_ne!(raw2.to_bits(), m1.raw_barrier(&s_ref, 0, j).to_bits());

            let _ = m1.rate(&s, 0, j);
            let de2 = m2.energy_change(&s, 0, j);
            let r2 = m2.rate(&s, 0, j);
            assert_eq!(de2.to_bits(), m2.energy_change(&s_ref, 0, j).to_bits());
            assert_eq!(r2.to_bits(), m2.rate(&s_ref, 0, j).to_bits());
            let r1 = m1.rate(&s_ref, 0, j);
            if r2.to_bits() != r1.to_bits() {
                saw_rate_diff = true;
            }
            if j == 0 {
                assert_ne!(r2.to_bits(), r1.to_bits());
            }
        }
        assert!(saw_rate_diff);

        let _ = m1.rate(&s, 0, 0);
        let mut s_apply = s.clone();
        m2.apply_reversible(
            &mut s_apply,
            &FeCrRisEvent::DumbbellJump { rec: 0, jump: 0 },
        )
        .unwrap();
        assert_eq!(s_apply.floored_dumbbell_jumps(), 1);

        let (j_ver, gap_pre, gap_post) = (0..8)
            .find_map(|j| {
                let raw_pre = m1.raw_barrier(&s_clean, 0, j);
                let gap_pre = raw_pre - m1.energy_change(&s_clean, 0, j).max(0.0);
                let mut s_next = s_clean.clone();
                m1.apply_reversible(&mut s_next, &FeCrRisEvent::DumbbellJump { rec: 0, jump: j })
                    .ok()?;
                let raw_post = m1.raw_barrier(&s_next, 0, j);
                let gap_post = raw_post - m1.energy_change(&s_next, 0, j).max(0.0);
                (raw_pre.to_bits() != raw_post.to_bits() && (gap_pre - gap_post).abs() > 1e-6)
                    .then_some((j, gap_pre, gap_post))
            })
            .expect("expected a jump where raw_barrier and floor margin change");

        let _ = m1.rate(&s, 0, j_ver);
        let undo = m1
            .apply_reversible(
                &mut s,
                &FeCrRisEvent::DumbbellJump {
                    rec: 0,
                    jump: j_ver,
                },
            )
            .unwrap();
        let mut encoded = Vec::new();
        m1.encode_state(&s, &mut encoded);
        let s_decoded = m1.decode_state(&encoded).unwrap();
        let mut query_order = [j_ver; 8];
        let mut idx = 1;
        for j in 0..8 {
            if j != j_ver {
                query_order[idx] = j;
                idx += 1;
            }
        }
        assert_ne!(
            m1.raw_barrier(&s, 0, j_ver).to_bits(),
            m1.raw_barrier(&s_clean, 0, j_ver).to_bits()
        );
        for j in query_order {
            assert_eq!(
                m1.raw_barrier(&s, 0, j).to_bits(),
                m1.raw_barrier(&s_decoded, 0, j).to_bits()
            );
            assert_eq!(
                m1.energy_change(&s, 0, j).to_bits(),
                m1.energy_change(&s_decoded, 0, j).to_bits()
            );
            assert_eq!(
                m1.rate(&s, 0, j).to_bits(),
                m1.rate(&s_decoded, 0, j).to_bits()
            );
        }

        let _ = m1.rate(&s, 0, j_ver);
        m1.undo(&mut s, undo);
        for j in query_order {
            assert_eq!(
                m1.raw_barrier(&s, 0, j).to_bits(),
                m1.raw_barrier(&s_clean, 0, j).to_bits()
            );
            assert_eq!(
                m1.energy_change(&s, 0, j).to_bits(),
                m1.energy_change(&s_clean, 0, j).to_bits()
            );
            assert_eq!(
                m1.rate(&s, 0, j).to_bits(),
                m1.rate(&s_clean, 0, j).to_bits()
            );
        }

        let m_mid = m1
            .clone()
            .with_dumbbell_saddle_shift(-0.5 * (gap_pre + gap_post) / 8.0);
        let mut s_f = s_clean.clone();
        let _ = m_mid.rate(&s_f, 0, j_ver);
        let ev_ver = FeCrRisEvent::DumbbellJump {
            rec: 0,
            jump: j_ver,
        };
        m_mid.apply_reversible(&mut s_f, &ev_ver).unwrap();
        let mut cold_bytes = Vec::new();
        m_mid.encode_state(&s_f, &mut cold_bytes);
        let mut s_cold = m_mid.decode_state(&cold_bytes).unwrap();
        let before_cold = s_f.floored_dumbbell_jumps();
        m_mid.apply_reversible(&mut s_f, &ev_ver).unwrap();
        m_mid.apply_reversible(&mut s_cold, &ev_ver).unwrap();
        assert_eq!(
            s_f.floored_dumbbell_jumps() - before_cold,
            s_cold.floored_dumbbell_jumps()
        );
        assert_eq!(s_f.floored_dumbbell_jumps(), 1);
    }
}
