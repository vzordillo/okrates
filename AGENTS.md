# OKRATES Engineering & Documentation Guidelines

## Core Principles

- **`okrates-core` stays minimal**: `#![forbid(unsafe_code)]`, deterministic RNG streams, and zero serialization/I/O dependencies (`serde`, `sha2`, and data contracts live in `okrates-schema`).
- **Transactional & panic-free runtime**: Library and Python FFI paths must never `panic!` or `.expect()` on invalid inputs, numerical overflow/underflow, clock stagnation, or absorbing states. Return typed `Result` errors, and leave simulation state, catalog, clock, step count, and RNG unchanged when `step()` fails.
- **Reproducibility**: All stochastic algorithms use explicit `KmcRng` streams (`ChaCha12Rng`) with domain-separated stream IDs rather than ad-hoc seed arithmetic offsets.

## Strict Docstring, Comment & Annotation Guidelines

1. **No `#![warn(missing_docs)]` and no restating names/types**
   - Do not add doc comments to self-explanatory getters (`model()`, `state()`, `time()`, `steps()`, `lattice()`), struct fields, or enum/error variants whose names and `Display` impls already state their meaning.
2. **Document only non-obvious contracts**
   - Physical units and sign conventions not already in the identifier (`eV`, `K`, `s^-1`, `a/2` half-lattice units, `E_after - E_before`).
   - Mathematical definitions, algorithmic invariants, and preconditions (`D_v = r a^2`, `SumTree` zero-subtree rounding guard, `undo` rollback contract).
   - Non-type-encoded failure modes.
3. **Single source of truth & brevity**
   - Keep function/type docstrings to **1–3 lines** unless defining a non-trivial equation or physical assumption list.
   - Document model assumptions (such as literature interpretation choices or barrier floors) **once** on the module or type—never duplicate them across trait `impl` methods or FFI wrappers.
   - Omit doc comments on trait `impl` methods unless behavior diverges from or narrows the trait contract.
4. **Inline comments explain *why*, never *what***
   - Only comment subtle numerical guards, floating-point rounding behavior, or non-obvious invariants.
   - No section-divider banners (`// ---- T1`) or step-by-step code narration.
   - Never reference private/untracked notes (`ADR-000x`, `R-000x`, `plan §...`, or local machine paths); cite published papers/DOIs or tracked repo paths only.
5. **Python & PyO3 bindings**
   - Put user-facing docstrings only in `python/okrates/` (concise 1–3 lines); keep `crates/okrates-python` `#[pyfunction]` items free of duplicated prose.
   - Encode shapes, defaults, and return schemas in Python type annotations (`dataclass(frozen=True)`, `.pyi` stubs, `NDArray[np.float64]`) rather than prose lists of dict keys or standard `ValueError`/`OverflowError` boilerplate.

## Commit & Pull Request Attribution

- Append `Co-authored-by: Gemini Antigravity <noreply@google.com>` as a commit trailer on agent-authored commits.
- Append `🤖 Generated with [Gemini Antigravity](https://antigravity.google) (Antigravity for Science)` as a footer on agent-created pull request descriptions.
