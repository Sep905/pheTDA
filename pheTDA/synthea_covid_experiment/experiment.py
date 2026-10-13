"""Run resumable pheTDA and unsupervised-baseline search benchmarks."""

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

from optuna_pipeline import (
    SemiSupervised_TDA_pipeline,
    rank_pareto_trials,
    save_ensemble_memberships,
    save_selection_report,
    select_pareto_trials,
)
from optuna_pipeline.pipelines import (
    sample_lens_hyperparameters,
    sample_lens_type,
)
from pipeline_objects import Lens_function
from utils import distance_matrix_computation

from .config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    SCENARIOS_BY_NAME,
    TRIAL_BUDGETS,
    UNSUPERVISED_BASELINE_NAME,
    Scenario,
)
from .prepare_data import prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "10k_synthea_covid19_csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
DEFAULT_PREPARED = Path(__file__).resolve().parent / "data"
RANDOM_BASELINE_STRATEGY = "random_baseline"
TPE_BASELINE_STRATEGY = "tpe_baseline"
BASELINE_STRATEGIES = (TPE_BASELINE_STRATEGY, RANDOM_BASELINE_STRATEGY)
PHETDA_STRATEGIES = ("tpe", "random")
STRATEGIES = (*PHETDA_STRATEGIES, *BASELINE_STRATEGIES)
BASELINE_SCENARIO = Scenario(name=UNSUPERVISED_BASELINE_NAME)


class UnsupervisedReductionClusteringObjective:
    """Unsupervised baseline: lens projection followed by direct clustering."""

    def __init__(
        self,
        feature_data,
        distance_matrix,
        seed,
        sample_ids,
        results_path,
        save_trial_artifacts=False,
    ):
        self.feature_data = feature_data
        self.distance_matrix = np.asarray(distance_matrix)
        self.seed = seed
        self.sample_ids = np.asarray(sample_ids)
        self.save_trial_artifacts = save_trial_artifacts
        self.results_path = Path(results_path) / f"{seed}_UnsupervisedBaseline"
        self.results_path_lens = self.results_path / "lens"
        self.results_path_projections = self.results_path / "projections"
        self.results_path_clustering = self.results_path / "clustering"
        if save_trial_artifacts:
            for directory in (
                self.results_path_lens,
                self.results_path_projections,
                self.results_path_clustering,
            ):
                directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, trial):
        trial.set_user_attr("pipeline_valid", False)
        lens_name = sample_lens_type(trial, self.feature_data)
        lens_parameters = sample_lens_hyperparameters(
            trial,
            lens_name,
            self.feature_data,
            projection_dimension=2,
            seed=self.seed,
        )
        projection_lens = Lens_function(
            lens_name,
            lens_parameters,
            self.feature_data,
            self.distance_matrix,
        )
        projections = np.asarray(projection_lens.projections, dtype=float)
        if projections.shape != (len(self.feature_data), 2):
            return -np.inf
        if not np.isfinite(projections).all():
            return -np.inf

        projection_scaler = StandardScaler()
        scaled_projections = projection_scaler.fit_transform(projections)
        clusterer = _sample_baseline_clusterer(trial, self.seed)
        labels = np.asarray(clusterer.fit_predict(scaled_projections))
        if np.any(labels == -1):
            return -np.inf
        number_of_labels = np.unique(labels).size
        if number_of_labels < 2 or number_of_labels >= len(labels):
            return -np.inf

        score = float(
            silhouette_score(
                self.distance_matrix,
                labels,
                metric="precomputed",
            )
        )
        if not np.isfinite(score):
            return -np.inf
        if self.save_trial_artifacts:
            self._save_artifacts(
                trial.number,
                projection_lens.lens_function,
                projections,
                scaled_projections,
                projection_scaler,
                clusterer,
                labels,
            )
        trial.set_user_attr("pipeline_valid", True)
        return score

    def _save_artifacts(
        self,
        trial_number,
        lens,
        projections,
        scaled_projections,
        projection_scaler,
        clusterer,
        labels,
    ):
        artifact_id = str(trial_number)
        _save_pickle(lens, self.results_path_lens / f"{artifact_id}.pickle")
        _save_pickle(
            clusterer,
            self.results_path_clustering / f"{artifact_id}_model.pickle",
        )
        _save_pickle(
            projection_scaler,
            self.results_path_clustering / f"{artifact_id}_scaler.pickle",
        )
        pd.DataFrame(
            {
                "dataset_id": self.sample_ids,
                "projection_1": projections[:, 0],
                "projection_2": projections[:, 1],
                "scaled_projection_1": scaled_projections[:, 0],
                "scaled_projection_2": scaled_projections[:, 1],
            }
        ).to_csv(
            self.results_path_projections / f"{artifact_id}.csv.gz",
            index=False,
            compression="gzip",
        )
        pd.DataFrame(
            {"dataset_id": self.sample_ids, "cluster": labels}
        ).to_csv(
            self.results_path_clustering / f"{artifact_id}_labels.csv.gz",
            index=False,
            compression="gzip",
        )


def _utc_now():
    return datetime.now(timezone.utc)


def _atomic_json_dump(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _save_pickle(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        pickle.dump(value, output)
    temporary.replace(path)


def _sample_baseline_clusterer(trial, seed):
    """Sample the same clustering methods/ranges used by pheTDA Mapper."""
    cluster_method = trial.suggest_categorical(
        "cluster_method",
        [
            "DBSCAN",
            "agglomerative_average",
            "agglomerative_complete",
            "agglomerative_single",
            "k_medoids",
        ],
    )
    if cluster_method == "DBSCAN":
        return DBSCAN(
            min_samples=trial.suggest_int("min_points", 1, 5),
            eps=trial.suggest_float("epsilon", 0.1, 0.5, step=0.1),
        )

    number_of_clusters = trial.suggest_int("n_clusters", 2, 5)
    if cluster_method.startswith("agglomerative_"):
        return AgglomerativeClustering(
            n_clusters=number_of_clusters,
            linkage=cluster_method.split("_", maxsplit=1)[1],
        )
    return KMedoids(
        metric="euclidean",
        n_clusters=number_of_clusters,
        init="heuristic",
        random_state=seed,
    )


def _is_baseline_strategy(strategy):
    return strategy in BASELINE_STRATEGIES


def _is_tpe_strategy(strategy):
    return strategy in {"tpe", TPE_BASELINE_STRATEGY}


def _load_or_create_sampler(strategy, seed, startup_trials, sampler_path):
    if sampler_path.exists():
        with sampler_path.open("rb") as sampler_file:
            return pickle.load(sampler_file)
    if _is_tpe_strategy(strategy):
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=startup_trials,
            multivariate=True,
            group=True,
        )
    if strategy in {"random", RANDOM_BASELINE_STRATEGY}:
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"unknown search strategy: {strategy!r}")


def _metric_definition(scenario):
    directions = []
    names = []
    if scenario.initial_column is not None:
        directions.append(scenario.initial_direction)
        names.append("initial_graph_entropy")
    if scenario.final_column is not None:
        directions.append(scenario.final_direction)
        names.append("final_graph_entropy")
    directions.append("maximize")
    names.append("stratification_silhouette")
    return directions, names


def _study_seed(base_seed, scenario_name, total_trials):
    scenario_number = (
        len(SCENARIOS_BY_NAME)
        if scenario_name == UNSUPERVISED_BASELINE_NAME
        else list(SCENARIOS_BY_NAME).index(scenario_name)
    )
    budget_number = [budget for budget, _startup in TRIAL_BUDGETS].index(total_trials)
    return base_seed + scenario_number * 100 + budget_number


def _dataset_hash(dataset):
    hashes = pd.util.hash_pandas_object(dataset, index=True)
    return hashlib.sha256(hashes.to_numpy().tobytes()).hexdigest()


def _mark_interrupted_trials_failed(study):
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL, skip_if_finished=True)


def _completed_trials(study):
    return study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))


def _valid_trials(study):
    return [
        trial
        for trial in _completed_trials(study)
        if trial.user_attrs.get("pipeline_valid", False)
    ]


def _trial_duration_seconds(trial):
    if trial.datetime_start is None or trial.datetime_complete is None:
        return 0.0
    return (trial.datetime_complete - trial.datetime_start).total_seconds()


def _metrics_payload(
    study,
    scenario,
    strategy,
    total_trials,
    startup_trials,
    active_seconds,
    status,
):
    completed = _completed_trials(study)
    valid = _valid_trials(study)
    failed = study.get_trials(deepcopy=False, states=(TrialState.FAIL,))
    first_started = study.user_attrs.get("first_started_at")
    elapsed = None
    if first_started is not None:
        elapsed = (_utc_now() - datetime.fromisoformat(first_started)).total_seconds()

    selected = None
    valid_pareto = [
        trial
        for trial in study.best_trials
        if trial.user_attrs.get("pipeline_valid", False)
    ]
    if valid_pareto:
        ranked = rank_pareto_trials(valid_pareto, study.directions)
        trial, distance = ranked[0]
        selected = {
            "trial_number": trial.number,
            "values": dict(zip(study.metric_names, trial.values)),
            "ideal_distance": float(distance),
            "parameters": trial.params,
        }

    best_observed = {}
    for index, (name, direction) in enumerate(
        zip(study.metric_names, study.directions)
    ):
        finite_values = [
            float(trial.values[index])
            for trial in valid
            if np.isfinite(trial.values[index])
        ]
        if finite_values:
            best_observed[name] = (
                min(finite_values)
                if direction.name.lower() == "minimize"
                else max(finite_values)
            )

    return {
        "status": status,
        "study_name": study.study_name,
        "strategy": strategy,
        "scenario": scenario.name,
        "total_trial_budget": total_trials,
        "startup_trials": startup_trials if _is_tpe_strategy(strategy) else None,
        "metric_names": study.metric_names,
        "directions": [direction.name.lower() for direction in study.directions],
        "completed_trials": len(completed),
        "valid_trials": len(valid),
        "failed_trials": len(failed),
        "pareto_trials": len(valid_pareto),
        "active_execution_seconds": active_seconds,
        "elapsed_wall_clock_seconds_including_interruptions": elapsed,
        "sum_completed_trial_durations_seconds": sum(
            _trial_duration_seconds(trial) for trial in completed
        ),
        "best_observed_per_objective": best_observed,
        "selected_solution": selected,
        "updated_at": _utc_now().isoformat(),
    }


def _export_study(study, study_directory, metric_names, ensemble_percentage):
    study.trials_dataframe().to_csv(study_directory / "trials.csv", index=False)
    valid_pareto = [
        trial
        for trial in study.best_trials
        if trial.user_attrs.get("pipeline_valid", False)
    ]
    if not valid_pareto:
        return
    ranked = rank_pareto_trials(valid_pareto, study.directions)
    rows = []
    for rank, (trial, distance) in enumerate(ranked, start=1):
        row = {
            "pareto_rank": rank,
            "trial_number": trial.number,
            "ideal_distance": float(distance),
        }
        row.update(dict(zip(metric_names, trial.values)))
        row.update({f"parameter_{name}": value for name, value in trial.params.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(study_directory / "pareto_trials.csv", index=False)
    pd.DataFrame(rows[:1]).to_csv(
        study_directory / "selected_solution.csv", index=False
    )

    if ensemble_percentage is None:
        return
    ensemble_count = len(
        select_pareto_trials(
            study,
            mode="ensemble",
            ensemble_percentage=ensemble_percentage,
        )
    )
    pd.DataFrame(rows[:ensemble_count]).to_csv(
        study_directory / "ensemble_selected_solutions.csv", index=False
    )


def _make_pipeline_objective(
    scenario,
    dataset,
    feature_data,
    distance_matrix,
    seed,
    results_path,
    save_trial_artifacts,
):
    return SemiSupervised_TDA_pipeline(
        distance_matrix=distance_matrix,
        dataset=feature_data,
        random_seed=seed,
        sample_ids=pd.Series(np.arange(len(dataset)), name="dataset_id"),
        results_path=results_path,
        initial_class=(
            dataset[scenario.initial_column]
            if scenario.initial_column is not None
            else None
        ),
        initial_class_type=scenario.initial_type,
        initial_entropy=scenario.initial_direction,
        final_class=(
            dataset[scenario.final_column]
            if scenario.final_column is not None
            else None
        ),
        final_class_type="categorical",
        final_entropy=scenario.final_direction,
        pipeline_type_name="SemiSupervisedTDA",
        save_trial_artifacts=save_trial_artifacts,
        validate_inputs=False,
    )


def _trial_artifact_paths(objective, trial_number):
    artifact_id = str(trial_number)
    return [
        objective.results_path_lens / f"{artifact_id}.pickle",
        objective.results_path_scomplex / f"{artifact_id}_G.pickle",
        objective.results_path_scomplex / f"{artifact_id}_s.pickle",
        objective.results_path_communities / f"{artifact_id}.xlsx",
    ]


def _baseline_artifact_paths(objective, trial_number):
    artifact_id = str(trial_number)
    return [
        objective.results_path_lens / f"{artifact_id}.pickle",
        objective.results_path_projections / f"{artifact_id}.csv.gz",
        objective.results_path_clustering / f"{artifact_id}_model.pickle",
        objective.results_path_clustering / f"{artifact_id}_scaler.pickle",
        objective.results_path_clustering / f"{artifact_id}_labels.csv.gz",
    ]


def _postprocess_baseline_solution(
    study,
    metric_names,
    feature_data,
    distance_matrix,
    seed,
    study_directory,
):
    selected_trials = select_pareto_trials(study, mode="single")
    selection_directory = study_directory / "selected_solutions" / "single"
    selection_directory.mkdir(parents=True, exist_ok=True)
    objective = UnsupervisedReductionClusteringObjective(
        feature_data=feature_data,
        distance_matrix=distance_matrix,
        seed=seed,
        sample_ids=np.arange(len(feature_data)),
        results_path=selection_directory,
        save_trial_artifacts=True,
    )
    selected_numbers = [trial.number for trial, _distance in selected_trials]
    marker_path = selection_directory / "postprocessing.json"
    expected_paths = [selection_directory / "selected_trials.xlsx"]
    for trial_number in selected_numbers:
        expected_paths.extend(_baseline_artifact_paths(objective, trial_number))
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("status") == "complete"
            and marker.get("selected_trial_numbers") == selected_numbers
            and all(path.exists() for path in expected_paths)
        ):
            return marker

    save_selection_report(
        selected_trials,
        metric_names,
        selection_directory,
        mode="single",
    )
    trial, _distance = selected_trials[0]
    artifact_paths = _baseline_artifact_paths(objective, trial.number)
    rerun_result = {
        "trial_number": trial.number,
        "original_values": list(trial.values),
    }
    if all(path.exists() for path in artifact_paths):
        rerun_result["artifacts_reused"] = True
    else:
        fixed_trial = optuna.trial.FixedTrial(trial.params, number=trial.number)
        rerun_value = objective(fixed_trial)
        if not fixed_trial.user_attrs.get("pipeline_valid", False):
            raise RuntimeError(
                f"selected baseline trial {trial.number} became invalid during rerun"
            )
        rerun_result.update(
            {
                "rerun_values": [float(rerun_value)],
                "artifacts_reused": False,
            }
        )
    marker = {
        "status": "complete",
        "mode": "single",
        "selected_trial_numbers": selected_numbers,
        "rerun_results": [rerun_result],
        "updated_at": _utc_now().isoformat(),
    }
    _atomic_json_dump(marker, marker_path)
    return marker


def _rerun_selected_trials(
    selected_trials,
    mode,
    metric_names,
    scenario,
    dataset,
    feature_data,
    distance_matrix,
    seed,
    study_directory,
    ensemble_percentage=None,
):
    selection_directory = study_directory / "selected_solutions" / mode
    selection_directory.mkdir(parents=True, exist_ok=True)
    objective = _make_pipeline_objective(
        scenario,
        dataset,
        feature_data,
        distance_matrix,
        seed,
        selection_directory,
        save_trial_artifacts=True,
    )
    selected_numbers = [trial.number for trial, _distance in selected_trials]
    marker_path = selection_directory / "postprocessing.json"
    marker = None
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_paths = [selection_directory / "selected_trials.xlsx"]
    for trial_number in selected_numbers:
        expected_paths.extend(_trial_artifact_paths(objective, trial_number))
    if mode == "ensemble":
        expected_paths.extend(
            [
                selection_directory / "ensemble_node_community_membership.csv.gz",
                selection_directory / "ensemble_patient_community_membership.csv.gz",
                selection_directory / "ensemble_membership_metadata.json",
            ]
        )
    if (
        marker is not None
        and marker.get("status") == "complete"
        and marker.get("selected_trial_numbers") == selected_numbers
        and marker.get("ensemble_percentage") == ensemble_percentage
        and all(path.exists() for path in expected_paths)
    ):
        return marker

    save_selection_report(
        selected_trials,
        metric_names,
        selection_directory,
        mode,
    )
    rerun_results = []
    for trial, _distance in selected_trials:
        artifact_paths = _trial_artifact_paths(objective, trial.number)
        if all(path.exists() for path in artifact_paths):
            rerun_results.append(
                {
                    "trial_number": trial.number,
                    "original_values": list(trial.values),
                    "artifacts_reused": True,
                }
            )
            continue
        fixed_trial = optuna.trial.FixedTrial(trial.params, number=trial.number)
        rerun_values = objective(fixed_trial)
        if not fixed_trial.user_attrs.get("pipeline_valid", False):
            raise RuntimeError(
                f"selected trial {trial.number} became invalid during {mode} rerun"
            )
        rerun_results.append(
            {
                "trial_number": trial.number,
                "original_values": list(trial.values),
                "rerun_values": list(rerun_values),
                "artifacts_reused": False,
            }
        )

    if mode == "ensemble":
        save_ensemble_memberships(
            selected_trials,
            objective.results_path_scomplex,
            selection_directory,
        )

    marker = {
        "status": "complete",
        "mode": mode,
        "ensemble_percentage": ensemble_percentage,
        "selected_trial_numbers": selected_numbers,
        "rerun_results": rerun_results,
        "updated_at": _utc_now().isoformat(),
    }
    _atomic_json_dump(marker, marker_path)
    return marker


def _postprocess_selected_solutions(
    study,
    ensemble_percentage,
    metric_names,
    scenario,
    dataset,
    feature_data,
    distance_matrix,
    seed,
    study_directory,
):
    single = select_pareto_trials(study, mode="single")
    ensemble = select_pareto_trials(
        study,
        mode="ensemble",
        ensemble_percentage=ensemble_percentage,
    )
    return {
        "single": _rerun_selected_trials(
            single,
            "single",
            metric_names,
            scenario,
            dataset,
            feature_data,
            distance_matrix,
            seed,
            study_directory,
        ),
        "ensemble": _rerun_selected_trials(
            ensemble,
            "ensemble",
            metric_names,
            scenario,
            dataset,
            feature_data,
            distance_matrix,
            seed,
            study_directory,
            ensemble_percentage,
        ),
    }


def _finalize_completed_study(
    study,
    scenario,
    strategy,
    total_trials,
    startup_trials,
    ensemble_percentage,
    metric_names,
    dataset,
    feature_data,
    distance_matrix,
    seed,
    study_directory,
    metrics_path,
    previous_active,
    session_start,
):
    _export_study(
        study,
        study_directory,
        metric_names,
        None if _is_baseline_strategy(strategy) else ensemble_percentage,
    )
    active = previous_active + time.perf_counter() - session_start
    _atomic_json_dump(
        _metrics_payload(
            study,
            scenario,
            strategy,
            total_trials,
            startup_trials,
            active,
            "postprocessing",
        ),
        metrics_path,
    )
    try:
        if _is_baseline_strategy(strategy):
            selections = {
                "single": _postprocess_baseline_solution(
                    study,
                    metric_names,
                    feature_data,
                    distance_matrix,
                    seed,
                    study_directory,
                )
            }
        else:
            selections = _postprocess_selected_solutions(
                study,
                ensemble_percentage,
                metric_names,
                scenario,
                dataset,
                feature_data,
                distance_matrix,
                seed,
                study_directory,
            )
    except (Exception, KeyboardInterrupt) as error:
        active = previous_active + time.perf_counter() - session_start
        study.set_user_attr("active_execution_seconds", active)
        payload = _metrics_payload(
            study,
            scenario,
            strategy,
            total_trials,
            startup_trials,
            active,
            (
                "postprocessing_interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "postprocessing_failed"
            ),
        )
        payload["postprocessing_error"] = repr(error)
        _atomic_json_dump(payload, metrics_path)
        raise

    active = previous_active + time.perf_counter() - session_start
    study.set_user_attr("active_execution_seconds", active)
    study.set_user_attr("completed_at", _utc_now().isoformat())
    payload = _metrics_payload(
        study,
        scenario,
        strategy,
        total_trials,
        startup_trials,
        active,
        "complete",
    )
    payload["selected_artifacts"] = {
        mode: {
            "selected_trial_numbers": details["selected_trial_numbers"],
            "directory": str(study_directory / "selected_solutions" / mode),
        }
        for mode, details in selections.items()
    }
    _atomic_json_dump(payload, metrics_path)


def _run_study(
    strategy,
    scenario,
    total_trials,
    startup_trials,
    dataset,
    feature_data,
    distance_matrix,
    output_directory,
    storage,
    base_seed,
    dataset_hash,
    ensemble_percentage,
):
    study_name = f"{strategy}__{scenario.name}__n{total_trials}"
    study_directory = output_directory / "studies" / study_name
    study_directory.mkdir(parents=True, exist_ok=True)
    sampler_path = study_directory / "sampler.pickle"
    seed = _study_seed(base_seed, scenario.name, total_trials)
    sampler = _load_or_create_sampler(strategy, seed, startup_trials, sampler_path)
    if _is_baseline_strategy(strategy):
        directions = ["maximize"]
        metric_names = ["stratification_silhouette"]
    else:
        directions, metric_names = _metric_definition(scenario)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        directions=directions,
        load_if_exists=True,
    )
    study_specification = {
        "strategy": strategy,
        "scenario": scenario.name,
        "total_trials": total_trials,
        "startup_trials": startup_trials if _is_tpe_strategy(strategy) else None,
        "seed": seed,
        "dataset_hash": dataset_hash,
        "feature_columns": FEATURE_COLUMNS,
    }
    saved_specification = study.user_attrs.get("benchmark_specification")
    if saved_specification is None:
        study.set_user_attr("benchmark_specification", study_specification)
    elif saved_specification != study_specification:
        raise ValueError(
            f"configuration changed for existing study {study_name}; use a new "
            "output directory"
        )
    if study.metric_names is None:
        study.set_metric_names(metric_names)
    elif list(study.metric_names) != metric_names:
        raise ValueError(f"metric definition changed for existing study {study_name}")
    if "first_started_at" not in study.user_attrs:
        study.set_user_attr("first_started_at", _utc_now().isoformat())

    _mark_interrupted_trials_failed(study)
    previous_active = float(study.user_attrs.get("active_execution_seconds", 0.0))
    metrics_path = study_directory / "metrics.json"
    session_start = time.perf_counter()
    if len(_completed_trials(study)) >= total_trials:
        _finalize_completed_study(
            study,
            scenario,
            strategy,
            total_trials,
            startup_trials,
            ensemble_percentage,
            metric_names,
            dataset,
            feature_data,
            distance_matrix,
            seed,
            study_directory,
            metrics_path,
            previous_active,
            session_start,
        )
        return
    if _is_baseline_strategy(strategy):
        objective = UnsupervisedReductionClusteringObjective(
            feature_data=feature_data,
            distance_matrix=distance_matrix,
            seed=seed,
            sample_ids=np.arange(len(dataset)),
            results_path=study_directory,
            save_trial_artifacts=False,
        )
    else:
        objective = _make_pipeline_objective(
            scenario,
            dataset,
            feature_data,
            distance_matrix,
            seed,
            study_directory,
            save_trial_artifacts=False,
        )

    def checkpoint(completed_study, _trial):
        active = previous_active + time.perf_counter() - session_start
        completed_study.set_user_attr("active_execution_seconds", active)
        _save_pickle(sampler, sampler_path)
        _atomic_json_dump(
            _metrics_payload(
                completed_study,
                scenario,
                strategy,
                total_trials,
                startup_trials,
                active,
                "running",
            ),
            metrics_path,
        )

    try:
        maximum_attempts = total_trials * 3
        while len(_completed_trials(study)) < total_trials:
            if len(study.trials) >= maximum_attempts:
                raise RuntimeError(
                    f"{study_name} exceeded {maximum_attempts} attempts before "
                    f"reaching {total_trials} completed trials"
                )
            remaining = total_trials - len(_completed_trials(study))
            study.optimize(
                objective,
                n_trials=remaining,
                callbacks=[checkpoint],
                catch=(Exception,),
                show_progress_bar=True,
            )
    except KeyboardInterrupt:
        active = previous_active + time.perf_counter() - session_start
        study.set_user_attr("active_execution_seconds", active)
        _save_pickle(sampler, sampler_path)
        _atomic_json_dump(
            _metrics_payload(
                study,
                scenario,
                strategy,
                total_trials,
                startup_trials,
                active,
                "interrupted",
            ),
            metrics_path,
        )
        raise

    study.set_user_attr("optimization_completed_at", _utc_now().isoformat())
    _save_pickle(sampler, sampler_path)
    _finalize_completed_study(
        study,
        scenario,
        strategy,
        total_trials,
        startup_trials,
        ensemble_percentage,
        metric_names,
        dataset,
        feature_data,
        distance_matrix,
        seed,
        study_directory,
        metrics_path,
        previous_active,
        session_start,
    )


def _load_or_compute_features(dataset, cache_directory):
    cache_directory.mkdir(parents=True, exist_ok=True)
    distance_path = cache_directory / "distance_matrix.npy"
    feature_path = cache_directory / "dataset_preprocessed.csv"
    timing_path = cache_directory / "preprocessing_metrics.json"
    dataset_hash = _dataset_hash(dataset[FEATURE_COLUMNS])
    cache_signature = {
        "dataset_hash": dataset_hash,
        "patients": len(dataset),
        "features": FEATURE_COLUMNS,
    }
    if distance_path.exists() and feature_path.exists() and timing_path.exists():
        cached_metadata = json.loads(timing_path.read_text(encoding="utf-8"))
        if cached_metadata.get("cache_signature") == cache_signature:
            distance_matrix = np.load(distance_path, allow_pickle=False)
            features = pd.read_csv(feature_path)
            _validate_preprocessing(distance_matrix, features, len(dataset))
            return distance_matrix, features

    started = time.perf_counter()
    distance_matrix, features = distance_matrix_computation(
        dataset,
        CONTINUOUS_FEATURES,
        CATEGORICAL_FEATURES,
        BINARY_FEATURES,
        cache_directory,
    )
    _atomic_json_dump(
        {
            "preprocessing_seconds": time.perf_counter() - started,
            "patients": len(dataset),
            "features": len(FEATURE_COLUMNS),
            "cache_signature": cache_signature,
            "created_at": _utc_now().isoformat(),
        },
        timing_path,
    )
    _validate_preprocessing(distance_matrix, features, len(dataset))
    return distance_matrix, features


def _validate_preprocessing(distance_matrix, features, number_of_samples):
    if distance_matrix.shape != (number_of_samples, number_of_samples):
        raise ValueError("cached distance matrix has the wrong shape")
    if len(features) != number_of_samples:
        raise ValueError("cached feature data has the wrong number of rows")
    if not np.isfinite(distance_matrix).all():
        raise ValueError("distance matrix contains non-finite values")
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("distance matrix is not symmetric")
    if not np.allclose(np.diag(distance_matrix), 0):
        raise ValueError("distance matrix diagonal is not zero")


def _update_benchmark_summary(output_directory):
    payloads = []
    studies_directory = output_directory / "studies"
    if not studies_directory.exists():
        return
    for metrics_path in sorted(studies_directory.glob("*/metrics.json")):
        payloads.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    _atomic_json_dump(payloads, output_directory / "benchmark_summary.json")
    if payloads:
        pd.json_normalize(payloads, sep=".").to_csv(
            output_directory / "benchmark_summary.csv", index=False
        )
        from .analyze_results import generate_report

        generate_report(output_directory)


def _write_experiment_manifest(args, dataset):
    args.output_directory.mkdir(parents=True, exist_ok=True)
    budgets = dict(TRIAL_BUDGETS)
    phenotype_studies = [
        {
            "strategy": strategy,
            "scenario": experiment_name,
            "total_trials": total_trials,
            "startup_trials": (
                budgets[total_trials] if _is_tpe_strategy(strategy) else None
            ),
        }
        for strategy in args.strategies
        if strategy in PHETDA_STRATEGIES
        for experiment_name in args.experiments
        for total_trials in args.budgets
    ]
    baseline_studies = [
        {
            "strategy": strategy,
            "scenario": UNSUPERVISED_BASELINE_NAME,
            "total_trials": total_trials,
            "startup_trials": (
                budgets[total_trials] if _is_tpe_strategy(strategy) else None
            ),
        }
        for strategy in args.strategies
        if _is_baseline_strategy(strategy)
        for total_trials in args.budgets
    ]
    studies = [*phenotype_studies, *baseline_studies]
    manifest = {
        "updated_at": _utc_now().isoformat(),
        "source_directory": str(args.source_directory.resolve()),
        "prepared_directory": str(args.prepared_directory.resolve()),
        "output_directory": str(args.output_directory.resolve()),
        "journal": str((args.output_directory / "optuna_studies.jsonl").resolve()),
        "base_seed": args.seed,
        "ensemble_percentage": args.ensemble_percentage,
        "dataset_hash": _dataset_hash(dataset),
        "patients": len(dataset),
        "features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "requested_studies": len(studies),
        "requested_completed_evaluations": sum(
            study["total_trials"] for study in studies
        ),
        "studies": studies,
    }
    _atomic_json_dump(manifest, args.output_directory / "experiment_manifest.json")
    return manifest


def _storage(journal_path):
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    file_path = str(journal_path.resolve())
    lock = optuna.storages.journal.JournalFileOpenLock(file_path)
    backend = optuna.storages.journal.JournalFileBackend(file_path, lock_obj=lock)
    return optuna.storages.JournalStorage(backend)


def _ensemble_fraction(value):
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError(
            "ensemble percentage must be greater than 0 and at most 1"
        )
    return fraction


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run resumable Synthea COVID pheTDA and baseline benchmarks."
    )
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prepared-directory", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES)
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(SCENARIOS_BY_NAME),
        default=list(SCENARIOS_BY_NAME),
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        choices=[budget for budget, _startup in TRIAL_BUDGETS],
        default=[budget for budget, _startup in TRIAL_BUDGETS],
    )
    parser.add_argument("--seed", type=int, default=203)
    parser.add_argument(
        "--ensemble-percentage",
        type=_ensemble_fraction,
        default=0.25,
        help="fraction of valid Pareto solutions to rerun for ensemble artifacts",
    )
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    prepared_path, _metadata_path = prepare_dataset(
        args.source_directory,
        args.prepared_directory,
        force=args.force_prepare,
    )
    dataset = pd.read_csv(prepared_path)
    if args.prepare_only:
        print(f"Prepared dataset: {prepared_path}")
        return

    manifest = _write_experiment_manifest(args, dataset)

    distance_matrix, feature_data = _load_or_compute_features(
        dataset, args.output_directory / "preprocessing"
    )
    storage = _storage(args.output_directory / "optuna_studies.jsonl")
    budgets = dict(TRIAL_BUDGETS)
    for strategy in args.strategies:
        if _is_baseline_strategy(strategy):
            continue
        for experiment_name in args.experiments:
            scenario = SCENARIOS_BY_NAME[experiment_name]
            for total_trials in args.budgets:
                _run_study(
                    strategy,
                    scenario,
                    total_trials,
                    budgets[total_trials],
                    dataset,
                    feature_data,
                    distance_matrix,
                    args.output_directory,
                    storage,
                    args.seed,
                    manifest["dataset_hash"],
                    args.ensemble_percentage,
                )
                _update_benchmark_summary(args.output_directory)
    for strategy in args.strategies:
        if not _is_baseline_strategy(strategy):
            continue
        for total_trials in args.budgets:
            _run_study(
                strategy,
                BASELINE_SCENARIO,
                total_trials,
                budgets[total_trials],
                dataset,
                feature_data,
                distance_matrix,
                args.output_directory,
                storage,
                args.seed,
                manifest["dataset_hash"],
                args.ensemble_percentage,
            )
            _update_benchmark_summary(args.output_directory)


if __name__ == "__main__":
    main()
