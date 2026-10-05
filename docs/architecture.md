# Architecture Overview

## AFQMC background

Auxiliary Field Quantum Monte Carlo (AFQMC) computes low-lying electronic states
by stochastically propagating a population of **walkers** (Slater
determinants) in imaginary time. At each time step a **propagator** applies a
Trotter-decomposed exponential of the Hamiltonian, using random **auxiliary
fields** sampled via the Hubbard-Stratonovich transformation. A **trial
wave function** supplies the quantities needed for importance
sampling and the phaseless constraint that controls the sign problem.
**Measurements** (energy, density matrices, etc.) are taken as mixed estimators
between the walkers and the trial at the end of each block of propagation
steps. See [Motta and Zhang, 2017](https://arxiv.org/pdf/1711.02242) for a comprehensive review
of _ab initio_ AFQMC. The code in this package is based on JAX, enabling end-to-end
automatic differentiation, JIT compilation, vectorization, and GPU acceleration.

---

## Code layout

```
trot/
|-- __init__.py
|-- afqmc.py                 High-level AFQMC driver class
|-- config.py                JAX configuration (GPU, precision, logging)
|-- staging.py               Convert PySCF objects to StagedInputs
|-- setup.py                 Assemble a runnable Job from staged inputs
|-- driver.py                QMC execution loop (equilibration + sampling)
|-- walkers.py               Walker init, orthogonalization, stochastic reconfiguration
|-- sharding.py              JAX multidevice sharding utilities
|-- pair_sampling_runtime.py Install frozen local Cholesky sampling in QMC drivers
|-- stat_utils.py            Blocking analysis and outlier rejection
|-- testing.py               Testing helpers
|-- lattices.py              Lattices for models
|-- vap.py                   Symmetry projected GHF (variation after projection)
|
|-- core/
|   |-- __init__.py
|   |-- system.py            System dataclass, WalkerKind type
|   |-- typing.py            Shared type aliases (ham_data, trial_data, etc.)
|   |-- ops.py               TrialOps, MeasOps, HamOps protocols
|   +-- levels.py            MLMC level specifications (LevelSpec, LevelPack)
|
|-- ham/
|   |-- __init__.py
|   |-- chol.py              HamChol (Cholesky-decomposed Hamiltonian)
|   +-- hubbard.py           Hubbard model Hamiltonian
|
|-- trial/
|   Includes trial wave function data and operations for various trial types
|
|-- meas/
|   Contains measurement operations for various trial types
|
|-- prop/
|   |-- __init__.py
|   |-- types.py             PropState, QmcParams, PropOps
|   |-- afqmc.py             AFQMC propagation (init + step)
|   |-- blocks.py            Block execution
|   |-- chol_afqmc_ops.py    Low level Cholesky AFQMC operations
|   |-- cpmc.py              CPMC propagation with fast updates
|   |-- cpmc_slow.py         CPMC propagation without fast updates
|   |-- hubbard_cpmc_ops.py  Low level Hubbard model CPMC operations
|   +-- utils.py             Propagation utilities
```

---

## Calculation flow

Matrix products use JAX's faster `"default"` precision policy. To request
`"highest"` precision globally, call
`trot.config.configure_once(matmul_precision="highest")` before importing
`trot.afqmc`, or set `JAX_DEFAULT_MATMUL_PRECISION=highest` before starting
Python. An explicit argument overrides the environment setting. This policy
controls matrix-product arithmetic independently of mixed-precision array
storage; selecting `"highest"` does not promote float32 arrays to float64.

With mixed precision, the FP32 VHS Cholesky copy stores only the upper triangle
for real factors, assuming the symmetric matrices produced by molecular staging.
This decision uses the input dtype only; no numerical symmetry scan is performed.
Complex factors remain unpacked. Full-precision propagation retains its unpacked
default, and matrix-product precision is unchanged.

Use `setup(..., prop_kwargs={"packed_cholesky": False})` or
`Afqmc(...).build_job(prop_kwargs={"packed_cholesky": False})` to opt out. Callers
supplying nonsymmetric real factors must use this override. Low-level
`make_prop_ops(..., packed_cholesky=None)` uses the automatic policy; explicit
`True` also packs real factors in full precision. The resolved layout is printed
at setup. Device setup and HF/dense-CISD host setup use the same policy.
Measurement storage and the staging archive format are unchanged.

Dense restricted PT2-CCSD energy measurement honors `mixed_precision=True`
for its expensive two-body doubles contractions. It retains the input precision
for Green functions, Green/Cholesky products, HF exchange, the overlap component,
one-body terms, Coulomb dots, scalar reductions, and the final connected-energy
subtraction. For double-precision inputs these retained operations use
FP64/complex128; the large T2 products use FP32/complex64 with the matrix-product
policy. Both precision paths factor the transition Green function as
`G = C @ R`, where `R` has shape `(nocc, norb)`, and precompute
`rot_chol[g] = C.T @ chol[g]` once in the measurement context. Exchange uses
occupied-space matrices; the other per-Cholesky products have shape
`(nocc, norb)`. This removes the full orbital-space cubic matrix products from
the Cholesky loop, as in the Thouless modes implementation. Full-space Green
and doubles-correction matrices are still formed once per walker, and the
dense doubles tensor is unchanged. As in dense CISD, the default
`memory_mode="high"` contracts all Cholesky vectors together, including the
large T2 applications. `memory_mode="low"` uses groups of `chol_batch_size`
vectors (default 64) with the same batched contractions. Only the final partial
group is zero-padded; the entire Hamiltonian is never padded. Real and imaginary
Green parts are contracted separately with real Cholesky tensors, avoiding a
full complex Cholesky copy. Both settings work with walker chunking.
`mixed_precision=False` uses the same half-rotated algebra in input precision.
The dense kernel retains its existing transpose convention. This does not
change the estimator, trial representation, sampling, or precision defaults.

Unrestricted measurement setup uses a shared 64-vector Cholesky contraction
helper (`meas/chol_setup.py`). It builds the full beta-basis rotation, singles
intermediates, and occupied/Thouless half rotations in bounded batches for UHF,
UCISD, UCISDT, and PT2-UCCSD. Dense, K-mode, and Thouless-mode paths inherit
this through their shared context builders. This setup policy applies in both
production memory modes and is independent of the energy batch-size setting.
The final context arrays retain their existing shapes and precision; no full
Hamiltonian padding or prefix/tail concatenation is used. Setup compilations
alone disable GPU autotuning to avoid profiling copies of the full inputs.
Cholesky model shards are processed locally without gathering the full tensor.
The full beta-basis Cholesky output is still stored; batching reduces temporary
workspace, not that persistent storage or production-energy memory.

The mixed-estimator driver reuses the guide's full beta-basis Cholesky tensor
when building a dense or mode PT2-UCCSD estimator with exactly the same beta
orbital matrix. This removes one full tensor allocation without changing the
separate guide and Thouless half rotations. An optional
`EstimatorOps.build_estimator_ctx_from_guide` hook supplies this reuse; custom
estimators retain their ordinary builder, and an explicitly supplied estimator
context is used as-is. Different beta bases fall back to separate construction.
The compiled block scanner also preserves known context-buffer sharing so
automatic chunk selection counts each shared input only once.

Dense UCISD and dense/mode PT2-UCCSD force-bias contractions, including combined-K
and spin-block UCISD modes, cast Cholesky vectors inside batches of
`chol_batch_size` (default 64). Real and imaginary matrix parts are contracted
separately, avoiding full FP32/complex Cholesky temporaries and full-sized
matmul autotuning operands. This batching also applies to their shared residual
energy contraction helper, independently of the doubles-energy memory mode.
Model-sharded inputs use local Cholesky batches. Production autotuning and
matrix-product precision settings are unchanged.

Dense UCISD and PT2-UCCSD measurements likewise accept
`memory_mode="low", chol_batch_size=64` for restricted/unrestricted walkers.
This batches the large doubles contractions in groups of Cholesky vectors,
using the same arithmetic and precision policy as their high-memory path;
only the final partial group is zero-padded. The factories still default to
`memory_mode="high"` (all vectors together). `chol_batch_size=1` selects a
single-vector batch. Other two-body intermediates and stored Hamiltonian
tensors retain their existing layouts. This is compatible with walker
chunking. UCISD's generalized-walker kernel is unchanged. Configure this via
`make_ucisd_meas_ops` or `make_pt2uccsd_estimator_ops` (also supported by
`make_pt2uccsd_meas_ops`); no new staging or high-level AFQMC option is needed.

A simulation has three stages: **staging**, **job assembly**, and
**QMC execution**.

```
PySCF object (RHF / UHF / GHF / CCSD / UCCSD)
  |
  |  staging.stage()
  v
StagedInputs  (HamInput + TrialInput + metadata)
  |                    \--- optionally cached as HDF5
  |  setup.setup()
  v
Job  (System + QmcParams + HamChol + trial_data + ops bundles)
  |
  |  Job.kernel()  -->  driver.run_qmc_energy()
  v
+---------------------------------------------------------------+
| QMC loop                                                      |
|                                                               |
|  1. Build prop_ctx (Trotter operators, MF shifts)             |
|  2. Build meas_ctx (intermediates for estimators,             |
|     like half-rotated integrals)                              |
|  3. Initialise walkers from trial RDM1                        |
|                                                               |
|  4. Equilibration  (n_eql_blocks)                             |
|     for each block:                                           |
|       - n_prop_steps AFQMC steps (sample fields, propagate,   |
|         update weights, population control)                   |
|       - orthogonalise walkers (QR)                            |
|       - measure block energy                                  |
|       - stochastic reconfiguration                            |
|                                                               |
|  5. Sampling  (n_blocks)                                      |
|       same as equilibration, but block energies are recorded  |
|                                                               |
|  6. Outlier rejection + statistical analysis                  |
|       - automatic-window Gamma error (reported by default)    |
|       - blocking/jackknife error and plateau diagnostic       |
+---------------------------------------------------------------+
  |
  v
(mean_energy, stderr, block_energies, block_weights)
```

**Staging** (`staging.py`) takes a PySCF mean field or coupled cluster object
and produces `StagedInputs`: the Cholesky decomposed Hamiltonian (`HamInput`)
and trial wave function data (`TrialInput`), with optional HDF5 caching.

**Job assembly** (`setup.py`) converts `StagedInputs` into JAX arrays,
selects the appropriate `trial_ops` / `meas_ops` /
`prop_ops`, and bundles everything into a `Job`.

**QMC execution** (`driver.py`) runs the equilibration and sampling loops,
calling the JIT compiled `block` function with `jax.lax.scan`.

The runtime and direct drivers pass the already built measurement context
to `PropOps.init_prop_state` through its optional `meas_ctx` keyword. State
initialization reuses this context for the initial energy instead of building
a second copy. Standalone initialization can omit `meas_ctx` and build it as
needed. Custom initializers should accept this keyword as part of the
`InitPropState` protocol.

The standard AFQMC initializer uses an optional `MeasOps.kernels["energy_init"]`
implementation when provided, falling back to `"energy"`. This must compute
the same deterministic local energy; it can trade speed for lower workspace
in this one-time calculation. CISD modes use Cholesky batching here, while
their regular `"energy"` kernel retains the full-Cholesky contractions.

For a Hamiltonian sharded along the first Cholesky axis (`P("model")`),
the setup builders detect the concrete input mesh before JIT tracing. The
Cholesky-square sum and CISD-mode setup use `jax.shard_map` so their batches
index each device's local vectors. The square sum reduces an orbital matrix;
`lci1` and reference scores remain distributed; the initial energy reduces
residual scalar contributions and adds the global base once. The CISD-mode
context stores this setup mesh as static metadata. Direct calls to compiled
`_sum_chol_squares` and `_build_lci1` must supply `mesh` explicitly to select
this path. Single-device setup retains the original 256-vector batching.

---

## Key objects

| Type           | Module                   | Role                                                                                                                                                                                      |
| -------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `System`       | `core/system.py`         | Static system config: `norb`, `nelec`, `walker_kind`                                                                                                                                      |
| `HamChol`      | `ham/chol.py`            | Cholesky decomposed Hamiltonian (`h0`, `h1`, `chol`).                                                                                                                                     |
| `QmcParams`    | `prop/types.py`          | QMC parameters: `dt`, `n_walkers`, `n_blocks`, `seed`, etc.                                                                                                                               |
| `PropState`    | `prop/types.py`          | Immutable simulation state (`NamedTuple`): walkers, weights, overlaps, RNG key, energy shift. Each step returns a new instance                                                            |
| `trial_data`   | `trial/*.py`             | Trial wave function data (e.g. MO and CI coefficients).                                                                                                                                   |
| `TrialOps`     | `core/ops.py`            | Trial wave function operations: `overlap`, `get_rdm1`, optional CPMC functions                                                                                                            |
| `MeasOps`      | `core/ops.py`            | Measurement operations: standard per-walker kernels, optional observables (`"rdm1"`, `"density_corr"`, ...), and an optional population-level block-energy estimator                       |
| `PropOps`      | `prop/types.py`          | Propagation operations: `init_prop_state`, `build_prop_ctx`, `step`                                                                                                                       |
| `meas_ctx`     | `meas/*.py`              | Precomputed measurement intermediates (e.g. half-rotated integrals), built once by `MeasOps.build_meas_ctx` and reused every block                                                        |
| `prop_ctx`     | `prop/chol_afqmc_ops.py` | Precomputed propagation intermediates (`CholAfqmcCtx`: Trotter exponentials, mean-field shifts, flattened Cholesky vectors), built once by `PropOps.build_prop_ctx` and reused every step |
| `Job`          | `setup.py`               | Fully assembled run bundle, built once by `PropOps.build_prop_ctx` and reused every step                                                                                                  |
| `StagedInputs` | `staging.py`             | Intermediate representation: `HamInput` + `TrialInput` + metadata, can be serialized to disk                                                                                              |

### When to use which container

Objects that flow through JAX transformations (`jit`, `lax.scan`, `vmap`) as
dynamic values must be **JAX pytrees** so their array leaves can be traced.

- **`NamedTuple` (as pytree)** — automatically a pytree with all fields as
  children. Good when every field is a JAX array and no validation or
  defaults are needed. Used for `PropState` (threaded through `lax.scan`
  each step).
- **`NamedTuple` (as record)** — also useful as a lightweight immutable
  record even when pytree behaviour is not needed, as long as you don't
  need methods or `default_factory`. Used for `TrialOps` (a simple bag of
  callables with `None` defaults, captured in JIT closures).
- **`@dataclass(frozen=True)` + manual pytree registration** — needed when
  some fields are static metadata (strings, ints) that belong in `aux_data`
  rather than traced children, or when you want `__post_init__` validation
  or non-trailing defaults. Used for `HamChol`, `meas_ctx`, and `prop_ctx`.
- **Plain `@dataclass(frozen=True)` (not a pytree)** — for objects that are
  never traced: static configuration captured in JIT closures or passed via
  `static_argnames`. Preferred over `NamedTuple` when you need methods or
  `field(default_factory=...)`. Used for `QmcParams`, `System`, `MeasOps`
  (which has helper methods like `require_kernel()`), and `PropOps`.

Note that large objects like the Hamiltonian should not be closed over in JIT
closures as they can lead to excessive compilation time and memory usage.

---

## Walker and Hamiltonian representations

`walkers.vmap_chunked` interprets `n_chunks` per data shard. With 400 walkers
on a `(data, model)=(4,1)` mesh, four chunks process 25 walkers at a time on
each device. For `n_chunks > 1`, a `shard_map` makes only the data axis
manual, placing the chunk loop inside the local walker population while
leaving model-axis contractions to automatic partitioning. The unchunked
`vmap` path is unchanged. The automatic memory selector caps its search at
the local population size and reports `walkers_per_data_shard`.

`QmcParams` enables automatic walker chunking by default (`auto_n_chunks=True`).
The driver starts from `n_chunks` and increases it as needed to satisfy the
compiler-estimated device memory budget. Set `auto_n_chunks=False` to keep
the requested chunk count fixed. If device memory statistics are unavailable,
the driver keeps the requested count.

The helper reads the mesh from the mapped argument's abstract type, which
retains Auto mesh axes inside JIT tracing even when the concrete partition
spec is unavailable. Walker mappings use axis zero for data. Calls that map
shared Cholesky indices or sampled pair indices explicitly set
`shard_walkers=False` to retain their original generic chunking. A surrounding
manual data map already supplies local walkers and does not need another
data map; small reference populations not divisible by the data mesh also
retain generic chunking.

RHF local energy with `memory_mode="low"` sums Cholesky vectors in batches
of `chol_batch_size` (default 256), configured through `RhfMeasCfg` or
`make_rhf_meas_ops`. It uses the same deterministic contractions as the
`"high"` path, accumulating each batch immediately. Only the final partial
batch is zero-padded to the batch size; it does not pad/copy the whole
Hamiltonian. Setting the batch size to one retains one-vector processing.
The host RHF setup honors this measurement configuration without an extra
setup option. The default measurement memory mode remains `"high"`.

With model sharding, the low-memory energy sums each device's local vectors
and reduces only the resulting two-body energy over the model axis. This
can nest inside device-local walker chunking. Green functions are explicit
arguments of the inner model map so their mesh types reflect both manual
axes; using captured outer tracers can give a mesh mismatch in JAX.

The `WalkerKind` literal (`"restricted"`, `"unrestricted"`, `"generalized"`)
determines how walker Slater determinants are stored. All walker arrays are complex
in _ab_initio_ AFQMC.
Cholesky Hamiltonian also follows the same basis conventions. Measurement function names
like force bias and energy indicate the walker and Hamiltonian kinds with suffixes, e.g., `_uw_rh` for
unrestricted walkers and restricted Hamiltonians.

### Restricted

Used for R(O)HF like walkers. A single coefficient matrix represents
both spins:

```
walkers: (n_walkers, norb, n_occ)
```

where n_occ = max(nup, ndn). Most commonly used.

### Unrestricted

Separate matrices for each spin:

```
walkers: (w_alpha, w_beta)
  w_alpha: (n_walkers, norb, nup)
  w_beta:  (n_walkers, norb, ndn)
```

### Generalized

Used with GHF basis Hamiltonians. A single matrix over spin-orbitals:

```
walkers: (n_walkers, 2*norb, ne)      where ne = nup + ndn
```

Every trial type (RHF, UHF, CISD, ...) implements overlap functions for all
three walker kinds, so walker kind and trial kind can be mixed freely. Currently
the code supports `"restricted"` Hamiltonians in most cases and we are adding support for
other kinds.

---

## Module summaries

### `ham/` -- Hamiltonian

Stores the molecular Hamiltonian in Cholesky decomposed form. `HamChol` holds
the scalar energy `h0`, one body integrals `h1`, and Cholesky vectors `chol`.
`HamChol` is a JAX pytree so it flows through `jit` and `lax.scan` without special handling.
The module also provides `hubbard.py` for lattice model Hamiltonians.

### `trial/` -- Trial Wave functions

Each module defines a frozen dataclass for trial data (e.g. `RhfTrial`,
`CisdTrial`, `UcisdTrial`), trial related functions like overlap, and a
factory `make_<kind>_trial_ops(sys)` that returns a `TrialOps` bundle with
the correct overlap function for the current walker type. Supported trial
types: RHF, UHF, GHF, CISD, UCISD, GCISD, CIS, EOM-CISD, and multi-GHF.

`auto.py` is useful for prototyping and testing: it only requires the definition
of the overlap and other quantities like force bias and energy are calculated by
taking derivatives.

### `meas/` -- Measurements

Mirrors `trial/` one to one, and defines measurement operations for each trial type.
Each module provides `make_<kind>_meas_ops(sys)` factory returning a `MeasOps` with required
algorithm kernels (`"energy"`, `"force_bias"`) and optional observable kernels (`"rdm1"`,
`"density_corr"`, etc.). Measurement contexts (`meas_ctx`) contain intermediates
evaluated once at the beginning of the calculations, like half-rotated integrals,
that allow for efficient estimator evaluation.

### `prop/` -- Propagation

Drives the imaginary time evolution. `types.py` defines `QmcParams`,
`PropState`, and `PropOps`. `afqmc.py` implements the standard phaseless
AFQMC step (auxiliary field sampling, Trotter propagation, importance sampling and constraint,
population control). `chol_afqmc_ops.py` contains low-level
Cholesky specific operations (Trotter exponentials, mean field shifts).
`blocks.py` packages a full block of propagation steps plus walker
orthogonalisation, measurement, and stochastic reconfiguration.

`cpmc.py` implements constrained path Monte Carlo as an alternative
propagation method for Hubbard models.

### `core/` -- Core Types

Defines the basic types: `System` and `WalkerKind` (`system.py`),
operation protocols `TrialOps` / `MeasOps` / `HamOps` (`ops.py`), type aliases
(`typing.py`), and MLMC level specifications `LevelSpec` / `LevelPack`
(`levels.py`).

---

## Entry points

The package offers three levels of API, from highest to lowest:

### 1. `AFQMC` class (high-level)

Defined in `afqmc.py`. Accepts a PySCF mean field or coupled cluster object
and handles staging, job assembly, and execution internally:

```python
from trot import AFQMC

afqmc = AFQMC(mf, norb_frozen=1, n_walkers=200, n_blocks=200)
mean, err = afqmc.kernel()
```

Key attributes: `walker_kind`, `mixed_precision`, `staged`, `job`, `e_tot`,
`e_err`.

#### Spin-specific trial FNO spaces

UCISD and PT2-UCCSD support compact, separately retained alpha and beta
virtual spaces, including unequal retained counts. This is **trial-only FNO**:
the full spatial Hamiltonian and walker dimension remain unchanged apart from
physical frozen-core removal. A full orbital rotation puts the Hamiltonian in
the alpha FNO basis; the full beta-to-alpha rotation is kept with the trial.
Restricted walkers remain supported.

PySCF input preparation can use its existing UMP2 FNO routine:

```python
from pyscf import mp, cc
from trot.staging import stage

ncore = 2  # Physical core frozen in both CC and AFQMC.
fno_threshold = 1.0e-3
ump2 = mp.UMP2(mf, frozen=ncore).run()  # mf is an already converged UHF object.
frozen, mo_fno = ump2.make_fno(thresh=fno_threshold)
mycc = cc.UCCSD(mf, frozen=frozen, mo_coeff=mo_fno)
mycc.verbose = 5
mycc.kernel()
staged = stage(mycc, norb_frozen_core=ncore, cache="trial_fno.h5")
```

The full MO coefficient matrices must retain the discarded columns, ordered
`[core | occupied | retained virtual | discarded virtual]` separately for each
spin, as returned by `make_fno`. Staging accepts PySCF's two frozen-index lists
and preserves them in the archive. The occupied frozen prefixes must match the
explicit common AFQMC frozen-core count; trial-only occupied freezing is not
implemented for unrestricted trials.

Dense UCISD, combined-K and spin-block modes, and dense/mode PT2-UCCSD use
compact excitation dimensions. For retained counts $v_\alpha$ and $v_\beta$,
the combined mode pair dimension is $o_\alpha v_\alpha + o_\beta v_\beta$;
amplitudes are not padded to the full virtual space. Green-function and
Hamiltonian contractions keep full orbital rows and only the retained trial
virtual columns where appropriate. Dense Cholesky batching and mode pair
sampling use the same compact spaces.

For low-level PT2-UCCSD construction, supply the compact raw `t1a`, `t1b`,
`t2aa`, `t2ab`, `t2bb` amplitudes and the **full** `mo_coeff_b` from the staged
UCISD trial to `make_ptuccsd_thouless_trial_data` or
`make_ptuccsd_thouless_mode_trial_data` (`t2_layout="pyscf"` for PySCF arrays).
These factories infer retained dimensions and build full-row Thouless
reference matrices. Serialized precomputed PT modes without amplitudes must
also retain `nvir_t_outer=(discarded_alpha, discarded_beta)`; legacy mode data
defaults to `(0, 0)`. UCISD mode caches infer the retained dimensions from the
stored singles arrays. FNO selection is independent of eigenmode compression;
neither precision defaults nor sampling policies are changed by using FNO.

#### Retained-mode CISD/UCISD trials

The opt-in CISD workflow is configured as one nested object rather than a
collection of flags on `AFQMC`. Mode compression is a host-side trial
preparation step; walker--Cholesky pair sampling is a runtime measurement
policy tuned after equilibration:

```python
from trot.afqmc import Afqmc
from trot.cisd_workflow import CisdWorkflowConfig

workflow = CisdWorkflowConfig.pair_sampled(
    discarded_norm_target=0.1,
    solver="auto",
)
afqmc = Afqmc(
    mycc,
    cache="afqmc.h5",
    cisd_workflow=workflow,
    n_walkers=200,
    n_eql_blocks=50,
    n_blocks=1000,
)

# Prefer doing this directly after the PySCF CC calculation on its CPU node.
# The raw amplitudes remain in trial/data; the smaller derived modes are cached
# alongside them below trial/derived.
afqmc.prepare_cisd_trial_cache()
```

With `solver="auto"`, host preparation uses dense diagonalization only when
both its estimated peak memory and the LAPACK workspace-index range are safe.
For an inexact mode selection, a dense `MemoryError` is retried with the
matrix-free Lanczos solver.

The GPU job can then load only the selected derived representation:

```python
afqmc = Afqmc.from_staged("afqmc.h5", cisd_workflow=workflow)
afqmc.walker_kind = "restricted"
energy, error = afqmc.kernel(target_error=2.0e-4)
```

Keeping the raw amplitudes makes a later change in compression policy
reproducible without rerunning CC. Derived representations are keyed by the
complete `CisdModeConfig`, so several cutoffs can coexist in one staged file.
If no matching cached representation exists, `setup()` can still construct it
from the raw amplitudes, but doing so during the GPU job is usually less
convenient.

For the experimental low-level option that samples pairs within each GPU's
walker population, see [local pair sampling](local_pair_sampling.md). It
requires frozen sampling settings and a replicated Hamiltonian.

For **pair-sampled** compressed CISD/UCISD energies and PT-CCSD/PT-UCCSD
components with a Hamiltonian sharded across a single node's GPUs, normal Jobs
enable local Cholesky sampling by default, including combined data/model meshes
(`QmcParams.local_cholesky_sampling=True`). They distribute the exact head
and balance the tail proposal across GPUs after any sampling-guide tuning,
then sample only locally stored Cholesky vectors and walkers. On a combined
mesh, both proposals are conditioned on each GPU's local arrays, with importance
weights preserving the global estimator. Hamiltonian, propagation,
guide, and estimator arrays receive the same physical permutation. In mixed
PT runs the layout follows the PT proposal while retaining the guide's own
head and probabilities.

Set `QmcParams(local_cholesky_sampling=False)` to retain the global sampler.
This option does not enable Hamiltonian sharding or pair sampling; deterministic
calculations, single-GPU runs, and walker-only sharded runs keep their existing
paths. For low-level PT runs, `state, runtime = job.prepare_runtime()` supplies
an owned `QmcRuntime` to `run_mixed_estimator_qmc(runtime=runtime, ...)`.
This lets the driver update the Job caches and release the original arrays
after redistribution. Separate borrowed Hamiltonian/context arguments retain
the global sampler. See [Cholesky pair sampling](cholesky_pair_sampling.md) for
eligibility, temporary memory requirements, and low-level installation.

### 2. `setup()` function (mid-level)

Defined in `setup.py`. Builds a `Job` from a PySCF object, `StagedInputs`, or
an HDF5 cache path, with full control over walker kind, precision, QMC
parameters, and custom operation overrides:

```python
from trot.setup import setup

job = setup(mf, walker_kind="restricted", mixed_precision=False)
mean, err, block_e, block_w = job.kernel()
```

### 3. `run_qmc()` / `run_qmc_energy()` (low-level)

Defined in `driver.py`. Operates directly on prebuilt components. Useful
when you need to supply a custom `PropState`, swap out individual operation
bundles, request specific observables, or resume a run:

```python
from trot.driver import run_qmc

result = run_qmc(
    sys=sys,
    params=params,
    ham_data=ham_data,
    trial_data=trial_data,
    trial_ops=trial_ops,
    meas_ops=meas_ops,
    prop_ops=prop_ops,
    block_fn=block_fn,
    observable_names=("rdm1",),
)
```

`run_qmc` returns a `QmcResult` namedtuple with `mean_energy`,
`stderr_energy`, `block_energies`, `block_weights`, `block_observables`, and
`observable_means`. The convenience wrapper `run_qmc_energy` returns only
`(mean, stderr, block_energies, block_weights)`.
