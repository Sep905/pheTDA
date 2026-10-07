"""Select final candidates from an Optuna multi-objective Pareto front."""

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _direction_name(direction):
    name = getattr(direction, "name", direction)
    name = str(name).lower()
    if name not in {"minimize", "maximize"}:
        raise ValueError(f"unknown optimization direction: {direction!r}")
    return name


def rank_pareto_trials(pareto_trials, directions):
    """Rank Pareto trials by normalized distance from the observed ideal point.

    Every non-constant objective is scaled to ``[0, 1]`` and oriented so that
    one is best. Constant objectives are ignored because they cannot
    distinguish trials. Equal distances prefer the highest silhouette score,
    followed by the lowest trial number.
    """
    trials = list(pareto_trials)
    if not trials:
        return []

    values = np.asarray([trial.values for trial in trials], dtype=float)
    if values.ndim != 2 or values.shape[1] != len(directions):
        raise ValueError("trial values and optimization directions are not aligned")
    if not np.isfinite(values).all():
        raise ValueError("Pareto trial objectives must contain only finite values")

    minima = values.min(axis=0)
    spans = values.max(axis=0) - minima
    informative = spans > 0
    distances = np.zeros(len(trials), dtype=float)
    if informative.any():
        normalized = (values[:, informative] - minima[informative]) / spans[informative]
        selected_directions = [
            _direction_name(direction)
            for direction, keep in zip(directions, informative)
            if keep
        ]
        for column, direction in enumerate(selected_directions):
            if direction == "minimize":
                normalized[:, column] = 1.0 - normalized[:, column]
        distances = np.linalg.norm(1.0 - normalized, axis=1)

    ranked = list(zip(trials, distances))
    ranked.sort(
        key=lambda item: (
            round(float(item[1]), 12),
            -float(item[0].values[-1]),
            item[0].number,
        )
    )
    return ranked


def select_pareto_trials(study, mode="single", ensemble_percentage=None):
    """Return ranked selected trials and their ideal-point distances."""
    validate_selection_options(mode, ensemble_percentage)
    valid_pareto_trials = [
        trial
        for trial in study.best_trials
        if trial.user_attrs.get("pipeline_valid", False)
    ]
    ranked = rank_pareto_trials(valid_pareto_trials, study.directions)
    if not ranked:
        raise RuntimeError("optimization produced no valid Pareto-optimal solutions")

    if mode == "single":
        return ranked[:1]

    number_to_select = max(1, math.ceil(len(ranked) * ensemble_percentage))
    return ranked[:number_to_select]


def validate_selection_options(mode, ensemble_percentage):
    if mode not in {"single", "ensemble"}:
        raise ValueError("solution_mode must be either 'single' or 'ensemble'")
    if mode == "single":
        if ensemble_percentage is not None:
            raise ValueError(
                "ensemble_percentage must be omitted when solution_mode='single'"
            )
        return
    if ensemble_percentage is None:
        raise ValueError(
            "ensemble_percentage is required when solution_mode='ensemble'"
        )
    if not 0 < ensemble_percentage <= 1:
        raise ValueError("ensemble_percentage must be greater than 0 and at most 1")


def save_selection_report(
    selected_trials,
    metric_names,
    selection_directory,
    mode,
):
    """Save selected trial identifiers, objectives, parameters, and ranking."""
    selection_directory = Path(selection_directory)
    selection_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, (trial, distance) in enumerate(selected_trials, start=1):
        row = {
            "selection_rank": rank,
            "trial_number": trial.number,
            "selection_mode": mode,
            "ideal_distance": float(distance),
        }
        row.update(dict(zip(metric_names, trial.values)))
        row.update({f"parameter_{name}": value for name, value in trial.params.items()})
        rows.append(row)
    report = pd.DataFrame(rows)
    report.to_excel(selection_directory / "selected_trials.xlsx", index=False)
    return report


def save_ensemble_memberships(selected_trials, graph_directory, selection_directory):
    """Save sparse positive memberships for the selected ensemble trials.

    The outputs retain each trial's distinct community system. They are an
    intermediate representation for a future consensus-ensemble algorithm,
    not a final merged partition. An omitted node/patient-community pair means
    non-membership, so dense binary matrices can be reconstructed if needed.
    """
    graph_directory = Path(graph_directory)
    selection_directory = Path(selection_directory)
    selection_directory.mkdir(parents=True, exist_ok=True)
    node_rows = []
    patient_rows = []
    trial_metadata = []

    for trial, _distance in selected_trials:
        graph_path = graph_directory / f"{trial.number}_G.pickle"
        if not graph_path.exists():
            raise FileNotFoundError(
                f"graph artifact is missing for selected trial {trial.number}"
            )
        with graph_path.open("rb") as graph_file:
            graph = pickle.load(graph_file)

        communities = sorted(
            {
                community
                for _node, attributes in graph.nodes(data=True)
                for community in attributes.get("communities", [])
            }
        )
        if not communities:
            raise ValueError(
                f"selected trial {trial.number} has no node-community memberships"
            )

        patient_communities = {}
        for node, attributes in graph.nodes(data=True):
            node_communities = set(attributes.get("communities", []))
            for community in sorted(node_communities):
                node_rows.append(
                    {
                        "trial_number": trial.number,
                        "node": str(node),
                        "community_id": community,
                    }
                )
            for patient in attributes.get("membership", []):
                patient_communities.setdefault(patient, set()).update(node_communities)

        for patient in sorted(patient_communities):
            memberships = patient_communities[patient]
            for community in sorted(memberships):
                patient_rows.append(
                    {
                        "trial_number": trial.number,
                        "dataset_id": patient,
                        "community_id": community,
                    }
                )

        trial_metadata.append(
            {
                "trial_number": int(trial.number),
                "node_count": int(graph.number_of_nodes()),
                "patient_count": len(patient_communities),
                "community_count": len(communities),
                "positive_node_memberships": sum(
                    len(set(attributes.get("communities", [])))
                    for _node, attributes in graph.nodes(data=True)
                ),
                "positive_patient_memberships": sum(
                    len(memberships) for memberships in patient_communities.values()
                ),
            }
        )

    node_memberships = pd.DataFrame(
        node_rows, columns=["trial_number", "node", "community_id"]
    )
    patient_memberships = pd.DataFrame(
        patient_rows, columns=["trial_number", "dataset_id", "community_id"]
    )
    node_path = selection_directory / "ensemble_node_community_membership.csv.gz"
    patient_path = (
        selection_directory / "ensemble_patient_community_membership.csv.gz"
    )
    metadata_path = selection_directory / "ensemble_membership_metadata.json"
    node_memberships.to_csv(
        node_path,
        index=False,
        compression="gzip",
    )
    patient_memberships.to_csv(
        patient_path,
        index=False,
        compression="gzip",
    )
    metadata = {
        "format": "sparse_positive_memberships",
        "omitted_pair_membership": 0,
        "stored_pair_membership": 1,
        "community_key": ["trial_number", "community_id"],
        "files": {
            "node_memberships": node_path.name,
            "patient_memberships": patient_path.name,
        },
        "selected_trial_count": len(trial_metadata),
        "trials": trial_metadata,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return node_memberships, patient_memberships
