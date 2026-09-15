# Unrestricted pt2CCSD in trot — implementation plan

Port `upt2ccsd`, `upt2ccsd_bar` and `upt2ccsd_sto_chol` from
`afqmc_lab/afqmc/afqmc/wavefunctions/wavefunctions_unrestricted.py` into trot, following
the restricted port (`trot/meas/pt2ccsd.py`, `trot/trial/pt2ccsd.py`) and running through
`AfqmcMixed` on the `AfqmcUh` ("uchol") hamiltonian.

---

## 0. What the afqmc code assumes (checked)

| item | afqmc | trot equivalent |
|---|---|---|
| hamiltonian | `chol[0]`, `chol[1]` — each spin in its own UHF MO basis, frozen core sliced per spin (`integral.get_chol`) | `staging.build_ham_uchol` → `HamCholU(h1_a, h1_b, chol_a, chol_b)` — same construction |
| guide | UHF, `mo_coeff = (I[:, :nocc_a], I[:, :nocc_b])` | `make_uhf_trial_data_uh` (identity columns), `make_prop_ops_u`, `make_uhf_meas_ops_uh` |
| walkers | `(walker_up, walker_dn)` | unrestricted tuple walkers, `init_walkers_uh` |
| t1 | `t1a (nocc_a, nvir_a)`, `t1b` | same |
| t2 | `t2aa = antisym(t2aa)/2 → (i,a,j,b)`, `t2ab → (i,a,j,b)`, `t2bb` likewise (`integral.save_cc_amplitude`) | stage the same way |
| trial ref | `mo_ta = exp_t1a.T @ I[:, :nocc_a]` (`slater_tools.uthouless`) | same |
| bar tensors | `exp_t1a = 1 + X_a`, `h1_bar_a = exp_t1a h1_a exp_mt1a`, `chol_bar_a` likewise, per spin | same (X² = 0, so no expm needed) |
| energy | `E = h0 + <e0> + <e1> - <t2><e0>`, weights `w_guide * <T|w>/<G|w>` | `block_mixed` + `pt2ccsd_blocking`, unchanged |

afqmc's kernels assume `norb_a == norb_b` (`self.norb`); the port takes the orbital sizes
per spin, so `norb_a != norb_b` (the `AfqmcUh` basis_a/basis_b case) also works.

---

## 1. Naming and API

Registered mixed recipes, keyed by (guide, trial) as today:

| trial | guide | measure_type | afqmc class |
|---|---|---|---|
| `"upt2ccsd"` | `"uhf"` | `"chunk"` | `upt2ccsd` (already chunked over cholesky) |
| `"upt2ccsd_bar"` | `"uhf"` | `"bar"` | `upt2ccsd_bar` |
| `"upt2ccsd_sto_chol"` | `"uhf"` | `"sto_chol"` | `upt2ccsd_sto_chol` |

The `u` prefix matches afqmc and trot's other unrestricted trials (`uhf`, `ucisd`,
`uccsd`), and keeps `get_mixed_recipe("pt2ccsd")` unambiguous.

Usage, through the same `AfqmcMixed` class:

```python
mf = scf.UHF(mol).run()
mycc = cc.UCCSD(mf, frozen=nfrozen).run()

af = AfqmcMixed(mycc, trial="upt2ccsd_bar")                # guide "uhf" implied
af = AfqmcMixed(mycc, trial="upt2ccsd_sto_chol",
                trial_kwargs={"chol_cost_ratio": 0.2})
af = AfqmcMixed(mycc)                                      # trial inferred: CCSD -> "pt2ccsd",
                                                           #                UCCSD -> "upt2ccsd"
af = AfqmcMixed(mycc, trial="upt2ccsd", basis_a=c_a, basis_b=c_b)  # as in AfqmcUh
mean, err = af.kernel()
```

- `trial=None` (new default) picks by CC type; an explicit trial that does not match the CC
  type (`UCCSD` + `"pt2ccsd"`, or `CCSD` + `"upt2ccsd"`) is an error, not a silent conversion.
- `basis_a`, `basis_b` are passed to `build_ham_uchol` exactly as `AfqmcUh` does. The CC
  amplitudes must be expressed in those bases (checked by shape).
- `max_memory`, `nchol_chunk`, `mixed_precision` and `trial_kwargs` mean what they mean for
  the restricted trials. The config dump (`trial_meas_cfg`, `nchol_chunk_used`,
  `n_chol_head_used`, `n_chol_samples_used`) works unchanged.

---

## 2. Files and steps

### 2.1 `trot/trial/upt2ccsd.py` (new)

- `Upt2ccsdTrial` pytree: `mo_t_a (norb_a, nocc_a)`, `mo_t_b (norb_b, nocc_b)`,
  `t2aa (na, va, na, va)`, `t2ab (na, va, nb, vb)`, `t2bb (nb, vb, nb, vb)`;
  `nocc`, `nvir`, `norb` properties as `(alpha, beta)` pairs.
- `overlap_u(walker, trial) = det(mo_t_a^H wu) * det(mo_t_b^H wd)`.
- `make_upt2ccsd_trial_data(data, sys)`.

### 2.2 `trot/staging.py`

- `stage_upt2ccsd_trial(cc, *, frozen=None) -> TrialInput(kind="upt2ccsd")` with
  `{mo_t_a, mo_t_b, t2aa, t2ab, t2bb}`, using the afqmc conventions in §0. Rejects
  non-UCCSD objects.

### 2.3 `trot/meas/upt2ccsd.py` (new)

Reuses from `meas/pt2ccsd.py` rather than duplicating: `Pt2ccsdMeasCfg`,
`DEFAULT_NCHOL_CHUNK`, `_equal_chunks`, `resolve_chol_budget`, `chol_sampling_proposal`,
`Pt2ccsdMemoryModel`, `ChunkPlan`, `plan_pt2ccsd_chunking`, and the config attribute used by
`get_pt2ccsd_meas_cfg` (so the flag dump needs no change).

- `Upt2ccsdMeasCtx` pytree: `cfg`, `nchol_chunk` (static); `exp_t1_a/b`, `h1_bar_a/b`,
  `chol_bar_a/b` (bar and sto_chol only).
- `build_bar_intermediates_u(ham_data, trial_data)` — t1 recovered from `mo_t_a/b` as in
  the restricted `t1_from_mo_t`.
- `build_meas_ctx(ham_data: HamCholU, trial_data, cfg)`.
- Kernels, each `(walker=(wu, wd), ham_data, meas_ctx, trial_data[, key]) -> [t2, e0, e1]`:
  - `energy_kernel_uw_uh_chunk` ← `upt2ccsd._calc_energy_pt`
  - `energy_kernel_uw_uh_bar` ← `upt2ccsd_bar._calc_energy_pt`
  - `energy_kernel_uw_uh_sto` ← `upt2ccsd_sto_chol._calc_energy_pt`
    (both spins share one head/tail split and one set of draws, scored by the
    alpha+beta `e2_0_g`, as in afqmc)
  The overlap `ot1` afqmc returns is not part of the kernel output: trot computes the trial
  overlap separately through `MeasOps.overlap`, exactly as the restricted port does.
  Mixed precision follows the restricted port: T2 contractions in `cfg.mixed_*_dtype`,
  greens functions and partial sums in f8/c16.
- `upt2ccsd_memory_model(...)` and `plan_chunking_for_run_u(...)`: the restricted model
  with every term summed over the two spins, so `max_memory` works here too.
- `make_upt2ccsd_meas_ops(sys, measure_type, ...)` — same validation as
  `make_pt2ccsd_meas_ops`; requires `walker_kind == "unrestricted"`; declares the energy
  kernel stochastic for `sto_chol` unless `n_chol_head="full"`.

### 2.4 `trot/mixed.py`

- `MixedRecipe` gains `ham_basis: str = "restricted"`; the unrestricted recipes set
  `"uchol"` and `walker_kind="unrestricted"`.
- Register the three recipes in §1; update docstrings.

### 2.5 `trot/setup_mixed.py`

- `_make_prop_mixed`: dispatch on `ham_data.basis`, so a uchol hamiltonian gets
  `make_prop_ops_u` (today it always builds the restricted propagator).
- `max_memory` error message lists the unrestricted names too.

### 2.6 `trot/afqmc.py` — `AfqmcMixed`

- `__init__`: `trial=None` default resolved from the CC type; `basis_a`, `basis_b`
  keywords; mismatch checks.
- `stage()`: when `recipe.ham_basis == "uchol"`, build the hamiltonian with
  `build_ham_uchol(self._cc, basis_a, basis_b, norb_frozen_core=k, chol_cut)` and pass it
  through `stage_inputs(self._scf, ham=...)`, as `AfqmcUh.stage` does. `k` is resolved from
  `norb_frozen_core` or else the integer `cc.frozen`, so the hamiltonian and the amplitudes
  always freeze the same core.
- Docstring: the unrestricted trials and `basis_a/basis_b`.

### 2.7 Shared fix in `meas/pt2ccsd.py::get_init_pt2trial_energy`

The τ=0 row prints `h0 + e0 + e1 - t2 * e1`; the estimator everywhere else
(`pt2ccsd_blocking`, afqmc `pt2blocking`, `test_initial_energies_match_rhf_and_ccsd`) is
`- t2 * e0`. Fix to `e0`. Only the printed τ=0 trial energy changes; no sampled number does.

### 2.8 Example and tests

- `examples/upt2ccsd.py`: one UHF/UCCSD system, one `AfqmcMixed` run; the other two trials
  and their options in comments (same style as the simplified restricted examples).
- `tests/test_upt2ccsd.py`, same layout as `tests/test_pt2ccsd.py`:
  1. at τ=0 all three kernels give the UCCSD energy (open-shell system with frozen core);
  2. chunk, bar and sto_chol(`n_chol_head="full"`) agree to round-off on perturbed walkers,
     for several `nchol_chunk`;
  3. closed-shell reduction: a restricted CCSD converted to UCCSD, walker `(w, w)`, gives
     the same `(t2, e0, e1)` as the restricted kernels;
  4. sto_chol is unbiased: averaging over keys converges to the exact `e1`;
  5. `AfqmcMixed` smoke runs for all three trials, and `sto_chol` with a full head
     reproduces the bar run exactly;
  6. registry: recipes, guide, walker kind, `trial=None` inference, mismatch errors.

---

## 3. Verification (outside the committed tests)

1. Kernel-by-kernel comparison against the afqmc classes themselves (`upt2ccsd`,
   `upt2ccsd_bar`, `upt2ccsd_sto_chol` with the same key) on random walkers — the direct
   check that the port is faithful.
2. Run the new tests, and re-run `tests/test_pt2ccsd.py` to confirm the restricted path is
   untouched (the τ=0 print fix aside). `pytest` is not installed in the `myafqmc` env,
   so the test functions are called from a small runner.
3. Short `AfqmcMixed` runs of each unrestricted trial; energies against UCCSD.

## 4. Out of scope

- An unchunked unrestricted kernel (afqmc has none) and `upt2ccsd_red` / `_cisd` / `_ad`.
- Multi-GPU model-axis sharding of the uchol hamiltonian (not wired in trot for `AfqmcUh`
  either).
