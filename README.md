# Solubility comparison evidence

Computational tools accompanying **Chemical evidence defines the scope of solubility comparisons**.

The package represents a comparison as a named target, its candidate objects,
numerical support, assay evidence and a separately stored referee. It preserves
the difference between a numerical answer, a finite set of possible answers,
and an assay that supports the requested chemical interpretation.

This public repository is **code only**: original implementation, schemas,
documentation and synthetic unit tests. It does not contain experimental data,
question/reference datasets, downloaded papers, figures or manuscripts. The
scientific comparison resource is a separate data archive supplied for review.
Public source releases and expected file contracts are listed in
[DATA_ACCESS.md](DATA_ACCESS.md). Source metadata and final-state assignments
require their documented primary-paper review; the package does not infer them
from numerical agreement.

## Install

Python 3.11 or later is required. The release is verified with Python 3.13 and
the pinned dependency versions in `pyproject.toml`.

```shell
python -m venv .venv
```

Activate it before installation:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Windows Command Prompt: `.venv\Scripts\activate`
- POSIX shell: `source .venv/bin/activate`

```shell
python -m pip install ".[test]"
solubility-compare --help
python -m pytest
```

Full molecular-graph normalization additionally uses the
optional `chemistry` extra. RDKit retains charge, fragments, isotope and stereo;
there is no salt stripping or similarity lookup.

## Inputs and reproducible operations

All paths are supplied explicitly. The names in the commands below denote
files from the separate resource archive; replace them with its indexed paths.
Each command writes to a new output directory and refuses an existing one.

```shell
solubility-compare replay --inputs INPUTS.jsonl --judge JUDGE.jsonl --output RESULTS
solubility-compare evaluate --inputs INPUTS.jsonl --judge JUDGE.jsonl --predictions PREDICTIONS --mode uncertainty_resolved --output STRICT_RESULTS
solubility-compare answer-sets --inputs SAFE_VIEWS.jsonl --profile frozen-development-v1 --output FINITE_RESULTS
```

`replay` predicts without opening the referee, then joins a physically separate
reference file for point and uncertainty-resolved evaluation. `predict` and
`answer-sets` have no referee argument. The schemas reject leaked answer fields,
duplicate IDs and exposed records relabelled as untouched holdouts. Software
checks cannot certify the semantic independence of a field renamed by a caller.

The generic answer-set interface requires a `finite_set_spec` declaring the
expected member count, candidates, direction and gap scale. A singleton requires
every declared member to be supported and to agree. Partial orders retain only
inequalities shared by all members. No probability is assigned to an
interpretation, and completeness is limited to the specified finite list.
`frozen-development-v1` is an explicit compatibility profile for the archived
development views, not a universal material-state rule.

## Scientific kernels

| Operation | Input contract and role |
|---|---|
| `membership` | Released SC3 fits, contributing-group keys and nested tier arrays; reconstruct means and decompose continuing-source versus composition changes |
| `mixtures` | MixtureSolDB-style native rows and an explicit window/identity-evidence list; compare common domains and an equal-source arithmetic-x counterfactual |
| `phase-proxy` | Phase-identified concentration rows plus predictor-safe query temperatures; reproduce the documented native/curve representations without reading phase referees |
| `contrast-dispersion` | Fixed source fits and held-out raw contrasts; paired versus independent-endpoint source-dispersion sensitivities |
| `panels` | Unit-compatible native reference cells and fixed source fits; pointwise versus common-source solvent comparisons |
| `process-screen` | Original Pillong-format category/outcome columns; uniform tied-grade and uniform-candidate descriptive policies |
| `benchmark-support` | Released SoluBench Tasks1/2 and their original BigSolDB rows; native-temperature mapping and within-domain conditioning |
| `identity-link` | Original-table rows and library rows with explicit link keys; source-column assignment, never phase assignment by concentration rank |
| `process-objectives` | Native capacities and measured condition tables; compare conditional induction and target-phase fraction on shared native coordinates |
| `convert-quantity` | Explicit native basis, density and appropriate molar mass; return mol/L pure solvent |

Every operation has `--help`. Python functions accept dataframes or typed
records, so a new source need not imitate a historical directory tree. Optional
JSON/TOML configuration has `version=1` and a `paths` mapping; relative paths
resolve beside the configuration file. CLI arguments override those paths.
No user-home path, institution login or hidden environment configuration is used.

## Evaluation semantics

The declared D gate requires an applicable concentration protocol and a stated
equilibration claim for phase inference. The observed D+O development comparator
requires explicit clear sampling and documented separation for every candidate.
O comes from the concentration assay and is recomputed from its fields; a cached
status or a method name cannot supply missing observations. This comparator is
not a validated universal definition of equilibrium quality.

Selectors retain a numerical answer; changing to a common-source contrast is a
separate representation intervention. Full-pool equal-budget evaluation lets
each selector choose its own accepted IDs. Shared-ID comparisons are retained
separately. Families and source/molecule groups are never pooled into one error
rate. Unresolved references remain in the ledger. Zero coverage has undefined
risk. Unknown uncertainty is not zero; source dispersion and experimental
SD/SE/standard uncertainty are distinct. No experimental covariance is filled
in and no uncertainty is divided by a sample-count square root.

A source-primary mapping check is a resolved record with the default
`evaluation_role="provenance_check"`; it does not enter an efficacy denominator.
A genuinely distinct primary referee may explicitly use `evaluation_role="reference"`.
Uncertainty-resolved qualification remains a separate requirement in either case.

These tools reproduce the specified operations. They do not establish a new
phase, physical causality, an unseen-source performance improvement or mass
recovery from a single-temperature concentration. Review exposure, source reuse,
phase handling and quantity compatibility remain part of each task contract.

## Provenance and release verification

[PROVENANCE.md](PROVENANCE.md) identifies archived method versions, source
repositories and dependencies. [MIGRATION.md](MIGRATION.md) maps the scientific
operations to their maintained interfaces. SC3 and AEM author software is not
copied into this MIT package. Their official repositories remain the route for
upstream curation/training and AEM author diagnostics.

Development used AI-assisted code generation/refactoring with OpenAI Codex.
Authorship, scientific interpretation and release responsibility remain with
Minhao Liu. This assistance is recorded here rather than decorating individual
source files or implying purely manual authorship.

The release QA checks synthetic scientific-risk cases, installation and CLI
execution, and equality against frozen external comparison arrays. The external
inputs and generated scientific tables stay outside the code repository;
`RELEASE_QA.json` records verification metadata only. Run the tests with
`python -m pytest`. Reproducing a scientific result also requires the relevant
external input files and their provenance index.
