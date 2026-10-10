# Computational provenance and attribution

The maintained package refactors the project's versioned analysis programs.
Original files, source arrays and frozen scientific outputs were preserved.
The release changes interfaces, validation, typing, organization and formatting;
it does not redefine source inclusion or tune decisions against a referee.

## Upstream scientific methods

* **SC3:** Ramani and colleagues' released source fitting, eligibility and
  consensus construction is the numerical baseline. The audited snapshot is
  `9d351e63e2341c7cd1e5846dd96e4ce6b0089811`,
  [official repository](https://github.com/VanshRamani/SC3). Our released-fit
  adapter and source-membership decomposition consume its external arrays.
  They are not an independent recreation of every primary measurement or a
  full upstream curation/training implementation. SC3 author source code is
  not redistributed or relicensed in this package.
* **AEM:** Haddadi and colleagues' official implementation was separately
  audited at Git tree SHA `4855d185bddaf7da77b43d2aaab7cdeb86e83c3c`,
  [official repository](https://github.com/HedayatHaddadi/AEM). Its MIT copyright
  is Hedayatollah Haddadi (2025). No AEM source or experiment dataset is copied
  here. Use the author's code for the author diagnostic; this package does not
  claim authorship of AEM or turn its coordinates into chemical labels.
* **SoluBench:** the temperature-support audit consumes the released task
  tables from snapshot `00ad03f584a1764a8aeaf30aec66b3e584aa45c6` of the
  [author repository](https://github.com/levakrasnovs/SoluBench). We do not ship
  their task corpus or claim to reproduce model performance.
* Native phase, source-identity and process observations are credited to their
  experimental authors in `DATA_ACCESS.md` and the manuscript. The original
  papers already establish their reported physical phenomena. Computational
  row linkage is not discovery of a new polymorph or thermodynamic mechanism.

The original package code is MIT, copyright2026 Minhao Liu. NumPy, pandas,
jsonschema, pytest and optional RDKit retain their respective licenses and are
installed dependencies, not vendored source. No third-party program or paper
has been silently moved under our copyright.

## Workflow and exposure

AI-assisted code generation and refactoring used OpenAI Codex. Minhao Liu is
responsible for scientific interpretation, review and release. The repository
does not represent this assistance as exclusively manual implementation.

Exposure is a property of each scientific task, not a filename or DOI match.
Development sources remain developed/previously exposed. An input-only policy
and a later reference join do not make an already inspected source blind.
Runtime guards reject explicit leakage and invalid exposure labels; renamed
semantic leakage still requires a human provenance review.

The numerical finite-set selector is distinct from material-state evidence.
Its completeness concerns the declared finite alternatives, not all physical
possibilities. The D+O comparator is a documented development rule. An
all-abstaining rule does not establish decision improvement.

## Verification

`tools/verify_external_replay.py` reads the separately supplied scientific audit
archive and compares against its frozen outputs. It does not copy experimental
inputs or write scientific output tables into the code repository. Full baseline
and finite-policy prediction payloads are compared, along with coverage tables
and the main numerical kernels. CSV round-trip normalization uses the original
storage representation and is not a new numerical tolerance or changed rule.
The local verification metadata, dependency versions, installation scope and
formatter check are recorded in `RELEASE_QA.json`.
