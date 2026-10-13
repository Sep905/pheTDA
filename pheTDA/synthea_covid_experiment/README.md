# Synthea COVID-19 pheTDA benchmark

This experiment compares two hyperparameter-search strategies over the same
conditional pheTDA search space:

- TPE with Optuna;
- uniform random search.

Community detection uses NetworkX weighted Louvain directly, without a
community-detection ensemble.

Optuna provides a common file-journal execution and persistence layer for both
strategies. The journal is append-only JSON Lines, not SQLite. The
`random` strategy uses `RandomSampler`, not TPE.

## Cohort and features

The preparation step creates one row per COVID-19 patient and uses complete
cases only. It currently produces 8,529 patients and 49 features:

- 12 continuous demographic, day-zero vital, and prior-utilization features;
- gender;
- 22 consolidated baseline SNOMED CT condition groups;
- 14 non-rare day-zero symptom indicators.

Race, ethnicity, county, and payer ownership are excluded. Oxygen saturation is
also excluded from the feature matrix because it is an initial phenotype.

The source condition codes are SNOMED CT, not ICD-9. First-digit grouping is
therefore not valid. Explicit condition groups are defined in `config.py`, and
each retained group must have at least 1% prevalence.

## Phenotypes and scenarios

Initial phenotypes:

- categorical day-zero oxygen-saturation tertile;
- continuous day-zero oxygen saturation.

Final phenotype:

- maximum COVID-19 severity within 90 days: mild, moderate, or severe.

The final-only severity-discovery scenario is run once:

- no initial phenotype;
- final severity phenotype;
- final entropy minimized.

For each categorical and continuous initial-phenotype representation, three
additional scenarios are run:

1. initial phenotype only;
2. both phenotypes, minimizing both entropies;
3. maximize initial entropy while minimizing final entropy.

This gives seven distinct phenotype scenarios. The shared final-only scenario
is not duplicated across the two initial-phenotype sensitivity branches.

## Budgets

Every strategy uses budgets of 50, 100, 250, and 500 evaluations. TPE uses
respectively 10, 15, 25, and 50 startup trials. Random search does not have a
startup phase. The complete default run contains 56 studies and 12,600
completed pipeline evaluations, so run the 50-trial example before committing
to the full benchmark.

MDS is always metric (`mds_metric=True`). Non-metric MDS is excluded because it
dominated runtime in the preliminary benchmark.

## Running

From the repository root, activate the Python 3.11 environment and prepare the
analysis table:

```bash
python -m synthea_covid_experiment.experiment --prepare-only
```

Run everything:

```bash
python -m synthea_covid_experiment.experiment
```

Run a small subset first:

```bash
python -m synthea_covid_experiment.experiment \
  --strategies tpe random \
  --experiments final_only \
  --budgets 50 \
  --ensemble-percentage 0.25
```

Each strategy/scenario/budget combination is a separate named study in
`results/optuna_studies.jsonl`. Relaunching the same command loads completed
trials, marks an interrupted running trial as failed, and continues until the
requested number of completed trials is reached. TPE and random sampler states
are also checkpointed after every completed trial.

After each optimization completes, the experiment ranks the valid Pareto front
and reconstructs two artifact configurations:

- `single`: the highest-ranked Pareto solution;
- `ensemble`: the highest-ranked fraction of Pareto solutions, controlled by
  `--ensemble-percentage` (25% by default, rounded upward).

Search trials do not save graph artifacts. Only the selected solutions are
rerun, which keeps storage manageable while producing their complete outputs.
Interrupted postprocessing is resumable and already-complete selected artifacts
are reused.

## Outputs

Each study directory contains:

- `metrics.json`: status, active execution time, total elapsed wall time,
  objective-wise best values, Pareto size, and selected solution;
- `trials.csv`: all trials, objective values, parameters, state, and duration;
- `pareto_trials.csv`: valid Pareto-front trials ranked by ideal-point distance;
- `selected_solution.csv`: the deterministic single selected solution;
- `ensemble_selected_solutions.csv`: the selected Pareto subset;
- `sampler.pickle`: sampler state used for exact continuation.

Under `selected_solutions/single` and `selected_solutions/ensemble`, each
selected trial is rerun with fixed parameters. Their nested
`<seed>_SemiSupervisedTDA` directories contain:

- `lens/<trial>.pickle`;
- `scomplex/<trial>_G.pickle` and `<trial>_s.pickle`;
- `communities/<trial>.xlsx`.

Both configurations include `selected_trials.xlsx` and `postprocessing.json`.
The ensemble configuration additionally includes sparse positive memberships in
`ensemble_node_community_membership.csv.gz` and
`ensemble_patient_community_membership.csv.gz`, plus
`ensemble_membership_metadata.json`. Omitted node/patient-community pairs mean
non-membership. These files keep trial community systems separate and are inputs
for a future consensus-ensemble algorithm; they are not yet a final merged
partition.

The results root also contains `benchmark_summary.json` and
`benchmark_summary.csv`, which are refreshed after every completed study.
`experiment_manifest.json` records the requested matrix, data fingerprint,
features, seed, and expected evaluation count.

The JSON Lines journal remains the authoritative checkpoint while a run is in
progress. Each line is a valid JSON journal record. Shared distance preprocessing
is cached once under `results/preprocessing` and timed separately so it does not
distort comparisons between search methods.

Existing named studies reject a changed data fingerprint, feature list, seed,
or strategy definition. Use a different `--output-directory` when intentionally
changing one of those inputs.

Artifacts are disabled during optimization, but enabled for the selected single
and ensemble solutions during postprocessing.
