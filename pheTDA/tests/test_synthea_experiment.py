import json
import pickle
from types import SimpleNamespace

import networkx as nx
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import pairwise_distances

import synthea_covid_experiment.experiment as experiment_module
from synthea_covid_experiment.config import (
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    SCENARIOS,
    TRIAL_BUDGETS,
)
from synthea_covid_experiment.experiment import (
    BASELINE_STRATEGIES,
    STRATEGIES,
    UnsupervisedReductionClusteringObjective,
    _load_or_create_sampler,
    _postprocess_baseline_solution,
    _storage,
    _write_experiment_manifest,
)


def test_experiment_matrix_has_seven_scenarios_and_four_budgets():
    assert len(SCENARIOS) == 7
    assert STRATEGIES == (
        "tpe",
        "random",
        "tpe_baseline",
        "random_baseline",
    )
    assert TRIAL_BUDGETS == (
        (50, 10),
        (100, 15),
        (250, 25),
        (500, 50),
    )


def test_final_only_scenario_is_unique_and_minimizes_final_entropy():
    final_only = [
        scenario
        for scenario in SCENARIOS
        if scenario.initial_column is None
        and scenario.final_column == "final_severity_3"
    ]

    assert len(final_only) == 1
    assert final_only[0].name == "final_only"
    assert final_only[0].final_direction == "minimize"


def test_feature_set_contains_49_features_without_oxygen_saturation():
    assert len(FEATURE_COLUMNS) == 49
    assert len(CONTINUOUS_FEATURES) == 12
    assert len(BINARY_FEATURES) == 37
    assert "oxygen_saturation_day0" not in FEATURE_COLUMNS


def test_manifest_runs_each_unsupervised_baseline_once_per_budget(tmp_path):
    args = SimpleNamespace(
        output_directory=tmp_path / "results",
        source_directory=tmp_path / "source",
        prepared_directory=tmp_path / "prepared",
        seed=203,
        ensemble_percentage=0.25,
        strategies=list(STRATEGIES),
        experiments=[scenario.name for scenario in SCENARIOS],
        budgets=[budget for budget, _startup in TRIAL_BUDGETS],
    )

    manifest = _write_experiment_manifest(args, pd.DataFrame({"value": [1]}))

    baseline_studies = [
        study
        for study in manifest["studies"]
        if study["strategy"] in BASELINE_STRATEGIES
    ]
    assert manifest["requested_studies"] == 64
    assert manifest["requested_completed_evaluations"] == 14_400
    assert len(baseline_studies) == 8
    assert {study["total_trials"] for study in baseline_studies} == {
        50,
        100,
        250,
        500,
    }
    assert {
        study["startup_trials"]
        for study in baseline_studies
        if study["strategy"] == "tpe_baseline"
    } == {10, 15, 25, 50}
    assert all(
        study["startup_trials"] is None
        for study in baseline_studies
        if study["strategy"] == "random_baseline"
    )


def test_unsupervised_baselines_use_the_requested_sampler(tmp_path):
    tpe = _load_or_create_sampler(
        "tpe_baseline", 7, 10, tmp_path / "missing-tpe.pickle"
    )
    random = _load_or_create_sampler(
        "random_baseline", 7, 10, tmp_path / "missing-random.pickle"
    )

    assert isinstance(tpe, optuna.samplers.TPESampler)
    assert isinstance(random, optuna.samplers.RandomSampler)


def test_json_journal_reopens_completed_optuna_study(tmp_path):
    journal_path = tmp_path / "study.jsonl"
    study = optuna.create_study(
        study_name="resume-test",
        storage=_storage(journal_path),
        direction="minimize",
    )
    study.optimize(lambda trial: trial.suggest_float("value", 0.0, 1.0), n_trials=2)

    resumed = optuna.load_study(
        study_name="resume-test",
        storage=_storage(journal_path),
    )

    assert len(resumed.trials) == 2
    assert all(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in resumed.trials
    )
    assert journal_path.suffix == ".jsonl"
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        assert isinstance(json.loads(line), dict)


def test_selected_single_and_ensemble_trials_are_rerun_with_artifacts(
    tmp_path, monkeypatch
):
    class ArtifactObjective:
        def __init__(self, results_path):
            run_directory = results_path / "7_SemiSupervisedTDA"
            self.results_path_lens = run_directory / "lens"
            self.results_path_scomplex = run_directory / "scomplex"
            self.results_path_communities = run_directory / "communities"
            for directory in (
                self.results_path_lens,
                self.results_path_scomplex,
                self.results_path_communities,
            ):
                directory.mkdir(parents=True, exist_ok=True)

        def __call__(self, trial):
            graph = nx.Graph()
            graph.add_node("node", communities=[1], membership=[0, 1])
            with (self.results_path_lens / f"{trial.number}.pickle").open(
                "wb"
            ) as output:
                pickle.dump({"trial": trial.number}, output)
            with (self.results_path_scomplex / f"{trial.number}_G.pickle").open(
                "wb"
            ) as output:
                pickle.dump(graph, output)
            with (self.results_path_scomplex / f"{trial.number}_s.pickle").open(
                "wb"
            ) as output:
                pickle.dump({"nodes": {"node": [0, 1]}}, output)
            (self.results_path_communities / f"{trial.number}.xlsx").touch()
            trial.set_user_attr("pipeline_valid", True)
            return (0.1, 0.2)

    monkeypatch.setattr(
        experiment_module,
        "_make_pipeline_objective",
        lambda *args, **kwargs: ArtifactObjective(args[5]),
    )
    trials = [
        SimpleNamespace(
            number=1,
            values=(0.2, 0.5),
            params={},
            user_attrs={"pipeline_valid": True},
        ),
        SimpleNamespace(
            number=2,
            values=(0.5, 0.8),
            params={},
            user_attrs={"pipeline_valid": True},
        ),
    ]
    study = SimpleNamespace(
        best_trials=trials,
        directions=("minimize", "maximize"),
    )

    selections = experiment_module._postprocess_selected_solutions(
        study=study,
        ensemble_percentage=1.0,
        metric_names=["entropy", "silhouette"],
        scenario=SimpleNamespace(),
        dataset=SimpleNamespace(),
        feature_data=SimpleNamespace(),
        distance_matrix=SimpleNamespace(),
        seed=7,
        study_directory=tmp_path,
    )

    assert len(selections["single"]["selected_trial_numbers"]) == 1
    assert set(selections["ensemble"]["selected_trial_numbers"]) == {1, 2}
    assert (
        tmp_path
        / "selected_solutions"
        / "ensemble"
        / "ensemble_patient_community_membership.csv.gz"
    ).exists()
    assert (
        tmp_path
        / "selected_solutions"
        / "ensemble"
        / "ensemble_membership_metadata.json"
    ).exists()


def test_unsupervised_baseline_uses_silhouette_and_saves_single_artifacts(tmp_path):
    features = pd.DataFrame(
        [
            [-2.0, -2.1, 0.0],
            [-1.9, -2.0, 0.1],
            [-2.1, -1.9, -0.1],
            [2.0, 2.1, 1.0],
            [1.9, 2.0, 0.9],
            [2.1, 1.9, 1.1],
        ]
    )
    distance_matrix = pairwise_distances(features)
    objective = UnsupervisedReductionClusteringObjective(
        feature_data=features,
        distance_matrix=distance_matrix,
        seed=7,
        sample_ids=np.arange(len(features)),
        results_path=tmp_path,
        save_trial_artifacts=True,
    )
    trial = optuna.trial.FixedTrial(
        {
            "lens_function": "PCA",
            "cluster_method": "agglomerative_complete",
            "n_clusters": 2,
        },
        number=3,
    )

    score = objective(trial)

    assert np.isfinite(score)
    assert trial.user_attrs["pipeline_valid"] is True
    assert (objective.results_path_lens / "3.pickle").exists()
    assert (objective.results_path_projections / "3.csv.gz").exists()
    assert (objective.results_path_clustering / "3_scaler.pickle").exists()
    labels_path = objective.results_path_clustering / "3_labels.csv.gz"
    assert labels_path.exists()
    labels = pd.read_csv(labels_path)
    assert labels["cluster"].nunique() == 2


def test_unsupervised_baseline_postprocesses_only_one_solution(tmp_path):
    features = pd.DataFrame(
        [
            [-2.0, -2.1, 0.0],
            [-1.9, -2.0, 0.1],
            [-2.1, -1.9, -0.1],
            [2.0, 2.1, 1.0],
            [1.9, 2.0, 0.9],
            [2.1, 1.9, 1.1],
        ]
    )
    distance_matrix = pairwise_distances(features)
    objective = UnsupervisedReductionClusteringObjective(
        feature_data=features,
        distance_matrix=distance_matrix,
        seed=7,
        sample_ids=np.arange(len(features)),
        results_path=tmp_path / "optimization",
    )
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(
        {
            "lens_function": "PCA",
            "cluster_method": "agglomerative_complete",
            "n_clusters": 2,
        }
    )
    study.optimize(objective, n_trials=1)

    marker = _postprocess_baseline_solution(
        study=study,
        metric_names=["stratification_silhouette"],
        feature_data=features,
        distance_matrix=distance_matrix,
        seed=7,
        study_directory=tmp_path,
    )

    assert marker["selected_trial_numbers"] == [0]
    assert (tmp_path / "selected_solutions" / "single").exists()
    assert not (tmp_path / "selected_solutions" / "ensemble").exists()
