# External scientific data

The public repository contains no measurement tables, comparison JSONL records,
papers, extracted figures or manuscript. The separate comparison-resource archive
supplied for review has its own index and hashes. Keep it outside this repository.
Use explicit CLI paths or a configuration file to point to those records.

## Public providers

| Resource | Provider and scientific use |
|---|---|
| BigSolDB2.0 | [Zenodo15094979](https://doi.org/10.5281/zenodo.15094979); the original benchmark-support/SoluBench generator input, distinct from SC3's2.1 snapshot |
| BigSolDB2.1 | [Zenodo18552681](https://zenodo.org/records/18552681); upstream input of the archived SC3 construction |
| BigSolDB2.2 | [Zenodo22648301](https://zenodo.org/records/22648301); measurement/method/solid-form provenance, distinct from author2.1 replay |
| MixtureSolDB | [Zenodo18660057](https://doi.org/10.5281/zenodo.18660057), [dataset paper](https://doi.org/10.1038/s41597-026-07047-z); native mixed-solvent records |
| SC3 | [Author repository](https://github.com/VanshRamani/SC3), [preprint](https://arxiv.org/abs/2606.07656); released cleaned checkpoint, source fits and tier arrays |
| SolProp v1.2 | [Zenodo5970538](https://zenodo.org/records/5970538), [method paper](https://doi.org/10.1021/jacs.2c01768); original experimental branches and source identifiers |
| SoluBench | [Author repository](https://github.com/levakrasnovs/SoluBench); released tasks and the original resources named by its generator |
| AEM | [Author repository](https://github.com/HedayatHaddadi/AEM), [paper](https://doi.org/10.1038/s41598-026-49445-8); author numerical diagnostics, not bundled here |
| Pillong screen | [Published study](https://doi.org/10.1039/C7CE00738H); original dissolution/crystal category table |

Original experimental concentration and independent-endpoint tables also come
from the source studies identified by the manuscript and resource index. These
include [SMZ](https://doi.org/10.1021/acs.cgd.5c01406),
[tolbutamide](https://doi.org/10.1016/j.xphs.2020.06.022),
[benzocaine](https://doi.org/10.1016/j.xphs.2020.07.022),
[piracetam](https://doi.org/10.1021/acs.cgd.1c01421),
[tolfenamic acid](https://doi.org/10.1021/acs.cgd.6c00631) and
[ritonavir](https://doi.org/10.1021/acs.molpharmaceut.4c00234).
Primary-paper access and redistribution follow each provider's license. This
code release is not a grant to redistribute their full text or data.

## Normalized input contracts

Predictor and referee schemas are installed at
`solubility_comparisons/schemas/task.json` and `judge.json`. The JSONL interface
retains `task_id`, resource, target family, candidate IDs, entity/source groups,
exposure status, numerical features and assay metadata. Referee values must be
in a separate file. A source-native primary-table provenance check is resolved
but unscored by default; a genuinely distinct reference can explicitly declare
`evaluation_role="reference"`. This role declaration does not replace source
review or certify laboratory independence.

Additional dataframe contracts use these column names:

| Kernel | Required fields |
|---|---|
| Released fits | `Solute_Canon, Solvent_Canon, gid, model, A, B, C, T_min, T_max`; fit eligibility kernels also use `R2, RMSE` |
| Contributor keys | `Solute_Canon, Solvent_Canon, group_id` or `gid` from the released/audited eligibility view |
| Tier arrays | `Solute_Canon, Solvent_Canon, Temperature_K, LogS_consensus, n_contributing_groups, sigma` |
| Native phase curves | `form, Tclear_C, reported_C_mg_ml, mass_volume_C_mg_ml, notes`; queries use safe JSONL contexts |
| Native mixture rows | Original MixtureSolDB solvent/fraction/temperature/mole-fraction headers; window list supplies the exact system keys and source pair |
| Reference cells | `doi, strict_solute, strict_solvent, T_K, Solute_Canon, Solvent_Canon, status, reference_LogC, released_native_tier_T, raw_ids, solvent_name` |
| Held-out contrasts | `Solute_Canon, Solvent_Canon, heldout_gid, T0, T1, n_common, common_source, reference_delta, pointwise, Gold, Silver` |
| Screen table | `PID, SMILES`, paired `S-<solvent>` and `C-<solvent>` columns; missing/unknown codes remain unobserved |
| Process objectives | Capacity: `solvent,T_K,capacity`; process: `solvent,T_K,drive,target_phase_count,n_trials,target_conditional_median_s` |

The original gravimetric-curve adapter reads `Amount/mg`, `Volume/mL`,
`Conc/mgmL-1`, `Tclear1/2` and `notes` from phase-labelled source CSVs. It avoids
rounding through an intermediate derived table before fitting. This is an
explicit header adapter, not automatic recognition of a phase from a filename.

Source-native quantities must be checked before conversion. SC3 uses log10 of
mol/L pure solvent; saturated-solution molarity and mole fraction are different
coordinates. The software never assigns room temperature, missing covariance,
unreported experimental uncertainty or a recovered phase to an unknown record.

## Path configuration

An external TOML file can contain:

```toml
version = 1
[paths]
inputs = "inputs.jsonl"
judge = "judge.jsonl"
output = "new-results"
```

Run `solubility-compare --config CONFIG.toml replay`. Paths are relative to the
configuration file. CLI paths override the configured ones. Do not place real
data in the public repository to make the examples runnable.
