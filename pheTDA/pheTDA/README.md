# pheTDA

pheTDA is a semi-supervised topological data analysis pipeline for discovering
patient stratifications. It constructs a Mapper graph from mixed clinical
features, detects graph communities with Louvain, and uses phenotype-aware graph
entropy together with community silhouette to guide a multi-objective Optuna
search.

The pipeline can use:

- an initial phenotype;
- a final phenotype; or
- both initial and final phenotypes.

Phenotypes guide hyperparameter selection but must not also be included among
the input features used to construct the Mapper graph.

## 📁Repository organization

```text
pheTDA/
|-- pipeline_run.py              # General command-line entry point
|-- optuna_pipeline/
|   |-- pipelines.py             # SemiSupervised_TDA_pipeline and search space
|   `-- solution_selection.py    # Pareto ranking and solution selection
|-- pipeline_objects/
|   |-- Lens_function.py         # Dimensionality-reduction lenses
|   |-- Covering.py              # Mapper covering and within-bin clustering
|   |-- Partitioning.py          # Louvain communities and stratification
|   `-- nn.py                    # Autoencoder used as an optional lens
|-- utils/
|   |-- prepro.py                # Feature preprocessing and distance matrix
|   `-- graph_utils.py           # Entropy, graph annotation, and tie resolution
|-- tests/                       # Pytest behavioral tests
|-- requirements.txt             # Runtime dependencies
|-- requirements-dev.txt         # Pytest and Ruff development dependencies
`-- pyproject.toml               # Pytest and Ruff configuration
```

## Installation

Python 3.9 or newer is required; Python 3.11 is recommended.
Install the runtime requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development checks, also install:

```bash
python -m pip install -r requirements-dev.txt
```

## Input data

`pipeline_run.py` accepts `.csv`, `.xls`, and `.xlsx` files. Each row represents
one sample or patient, and columns contain discovery features and phenotype
labels.

The command divides discovery features into three non-overlapping groups:

- `--continuous_features`: numeric continuous variables;
- `--categorical_features`: categorical variables with two or more levels;
- `--binary_features`: binary indicators.

At least one discovery feature and at least one of `--initial_class` or
`--final_class` must be supplied. Additional requirements are:

- input features and phenotypes must not contain missing values;
- continuous features and continuous phenotypes must be numeric and finite;
- a column cannot occur in more than one feature group;
- phenotype columns cannot also be discovery features;
- the dataset must contain at least three rows.

⚠️ The pipeline does not impute missing data. Perform any required cohort filtering,
encoding decisions, or imputation before running it. Categorical and binary
features may contain strings; they are encoded internally for lens computation.

Feature-list arguments use Python-list syntax. For example:

```text
--continuous_features "['age', 'bmi']"
--categorical_features "['sex', 'smoking_status']"
--binary_features "['diabetes', 'hypertension']"
```

The generated `dataset_id` is the zero-based row position in the input file.
Keep a separate identifier column in the source data if results must later be
joined to external records.

## 🏃 Running pheTDA

Run commands from the repository root. Display all command-line options with:

```bash
python pipeline_run.py --help
```

### Initial phenotype only

```bash
python pipeline_run.py \
  --dataset_path data/cohort.csv \
  --initial_class oxygen_tertile \
  --initial_class_type categorical \
  --continuous_features "['age', 'bmi', 'heart_rate']" \
  --categorical_features "['sex', 'smoking_status']" \
  --binary_features "['diabetes', 'hypertension']" \
  --n_trials 100 \
  --n_startup_trials 15 \
  --solution_mode single \
  --results_path results/initial_only
```

### Final phenotype only

```bash
python pipeline_run.py \
  --dataset_path data/cohort.csv \
  --final_class outcome_severity \
  --final_class_type categorical \
  --continuous_features "['age', 'bmi', 'heart_rate']" \
  --categorical_features "['sex', 'smoking_status']" \
  --binary_features "['diabetes', 'hypertension']" \
  --n_trials 100 \
  --n_startup_trials 15 \
  --solution_mode single \
  --results_path results/final_only
```

### Initial and final phenotypes

```bash
python pipeline_run.py \
  --dataset_path data/cohort.csv \
  --initial_class oxygen_saturation \
  --initial_class_type continuous \
  --final_class outcome_severity \
  --final_class_type categorical \
  --initial_entropy minimize \
  --final_entropy minimize \
  --continuous_features "['age', 'bmi', 'heart_rate']" \
  --categorical_features "['sex', 'smoking_status']" \
  --binary_features "['diabetes', 'hypertension']" \
  --n_trials 250 \
  --n_startup_trials 25 \
  --seed 203 \
  --solution_mode single \
  --results_path results/both_phenotypes
```

The examples use shell line continuations. In Windows PowerShell, either put the
command on one line or replace each trailing `\` with a backtick.

## Objectives and entropy directions

Every run maximizes `stratification_silhouette`. Depending on the supplied
phenotypes, it also optimizes:

- `initial_graph_entropy`;
- `final_graph_entropy`; or
- both entropy objectives.

Entropy directions default to `minimize`:

```text
--initial_entropy minimize
--final_entropy minimize
```

Minimizing entropy favors Mapper nodes that are homogeneous for that phenotype.
Maximizing entropy favors mixing. For example, presentation mixing followed by
outcome separation can be requested with:

```text
--initial_entropy maximize --final_entropy minimize
```

Categorical entropy supports more than two levels. A continuous phenotype is
first discretized using global Freedman-Diaconis bin edges, after which Shannon
entropy is calculated within each Mapper node. The graph score is the
membership-weighted mean of the node entropies.

## Hyperparameter optimization

The runner uses Optuna's TPE sampler. `--n_trials` controls the total number of
evaluated configurations, while `--n_startup_trials` controls the initial random
sampling phase. If the requested startup count exceeds the trial count, it is
automatically limited to the trial count.

The search covers:

- lenses: PCA, metric MDS, spectral embedding, Isomap, LLE, t-SNE, UMAP, and an
  autoencoder, subject to dataset-size requirements;
- Mapper interval count and overlap;
- DBSCAN, agglomerative clustering, or k-medoids within cover elements;
- Louvain resolution;
- the strategy used to resolve patients belonging to nodes from tied
  communities.

The complete objective always includes silhouette, so a valid result must assign
all samples and produce at least two communities but fewer communities than
samples.

## Selecting a Pareto solution

For the default single-solution mode:

```text
--solution_mode single
```

Pareto objective values are normalized within the observed front and oriented
so that larger normalized values are better. The selected solution has the
smallest Euclidean distance from the ideal point. Ties prefer the larger
silhouette and then the lower Optuna trial number.

To retain a fraction of the Pareto front for experimental ensemble work:

```text
--solution_mode ensemble --ensemble_percentage 0.25
```

This selects the highest-ranked 25% of valid Pareto solutions, rounded upward.
The current ensemble output preserves the separate community memberships from
each selected solution; it does not yet merge them into one consensus partition.
Do not provide `--ensemble_percentage` in single mode.

## Outputs

For `--results_path results/my_run` and the default seed, outputs have this
structure:

```text
results/my_run/
|-- distance_matrix.npy
|-- dataset_preprocessed.csv
`-- 203_SemiSupervisedTDA/
    |-- df_results.xlsx
    |-- lens/
    |   `-- <trial>.pickle
    |-- scomplex/
    |   |-- <trial>_G.pickle
    |   `-- <trial>_s.pickle
    |-- communities/
    |   `-- <trial>.xlsx
    `-- selection/
        |-- selected_trials.xlsx
        |-- ensemble_node_community_membership.csv.gz       # ensemble only
        |-- ensemble_patient_community_membership.csv.gz    # ensemble only
        `-- ensemble_membership_metadata.json               # ensemble only
```

`df_results.xlsx` contains objective values and sampled parameters for every
Optuna trial. Community spreadsheets contain one final community assignment per
input row. The graph pickle contains Mapper topology and node annotations; the
simplicial-complex pickle contains Mapper node memberships.

Artifacts are currently saved for every valid optimization trial, so large
trial budgets can consume substantial disk space.

### Reusing preprocessing

Use `--reuse_preprocessing` to reuse `distance_matrix.npy` and
`dataset_preprocessed.csv` from the selected results directory:

```bash
python pipeline_run.py ... --results_path results/my_run --reuse_preprocessing
```

Only use this option when the input rows, their order, and all feature columns
are identical to the run that created the cache. The general runner validates
the cached row count but does not fingerprint the dataset or feature definition.

`pipeline_run.py` does not persist or resume an interrupted Optuna study. A new
invocation starts a new study, although preprocessing can be reused as described
above.

## Computational considerations

Preprocessing constructs a dense pairwise distance matrix, which requires
quadratic memory in the number of samples. Several available lenses also have
quadratic or worse runtime characteristics. Start with a modest cohort and trial
count, monitor memory and runtime, and scale only after confirming feasibility.

## Tests and code checks

After installing `requirements-dev.txt`, run the behavioral test suite with:

```bash
python -m pytest
```

Run static checks and formatting verification with:

```bash
python -m ruff check --no-cache .
python -m ruff format --check .
```

Pytest verifies runtime behavior; Ruff checks source quality and formatting.
