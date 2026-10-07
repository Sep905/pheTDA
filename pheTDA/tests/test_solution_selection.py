import json
import pickle
from types import SimpleNamespace

import networkx as nx
import pandas as pd
import pytest

from optuna_pipeline.solution_selection import (
    rank_pareto_trials,
    save_ensemble_memberships,
    select_pareto_trials,
    validate_selection_options,
)


def _trial(number, values, valid=True):
    return SimpleNamespace(
        number=number,
        values=values,
        params={"example": number},
        user_attrs={"pipeline_valid": valid},
    )


def test_equal_ideal_distances_prefer_silhouette_then_trial_number():
    low_silhouette = _trial(1, [0.0, 0.5])
    high_silhouette_late = _trial(7, [1.0, 1.0])
    high_silhouette_early = _trial(3, [1.0, 1.0])

    ranked = rank_pareto_trials(
        [low_silhouette, high_silhouette_late, high_silhouette_early],
        ["minimize", "maximize"],
    )

    assert [trial.number for trial, _distance in ranked] == [3, 7, 1]


def test_constant_objectives_do_not_affect_ranking():
    trials = [_trial(4, [2.0, 0.1]), _trial(2, [2.0, 0.9])]

    ranked = rank_pareto_trials(trials, ["minimize", "maximize"])

    assert [trial.number for trial, _distance in ranked] == [2, 4]


def test_ensemble_selects_ceiling_of_requested_fraction():
    trials = [_trial(number, [float(number), 1.0 - number / 10]) for number in range(5)]
    study = SimpleNamespace(
        best_trials=trials,
        directions=["minimize", "maximize"],
    )

    selected = select_pareto_trials(study, "ensemble", 0.21)

    assert len(selected) == 2


def test_invalid_pareto_trials_are_not_selected():
    study = SimpleNamespace(
        best_trials=[_trial(0, [0.0, 1.0], valid=False), _trial(1, [1.0, 0.0])],
        directions=["minimize", "maximize"],
    )

    selected = select_pareto_trials(study)

    assert selected[0][0].number == 1


@pytest.mark.parametrize("percentage", [None, 0.0, -0.1, 1.1])
def test_ensemble_percentage_must_be_in_valid_range(percentage):
    with pytest.raises(ValueError):
        validate_selection_options("ensemble", percentage)


def test_single_mode_rejects_ensemble_percentage():
    with pytest.raises(ValueError):
        validate_selection_options("single", 0.5)


def test_ensemble_memberships_are_sparse_and_keep_trials_separate(tmp_path):
    graph = nx.Graph()
    graph.add_node("left", communities=[1], membership=[0, 1])
    graph.add_node("overlap", communities=[2], membership=[1, 2])
    graph_directory = tmp_path / "graphs"
    graph_directory.mkdir()
    with (graph_directory / "5_G.pickle").open("wb") as graph_file:
        pickle.dump(graph, graph_file)

    selected = [(_trial(5, [0.1, 0.8]), 0.0)]
    nodes, patients = save_ensemble_memberships(
        selected, graph_directory, tmp_path / "selection"
    )

    assert list(nodes.columns) == ["trial_number", "node", "community_id"]
    assert list(patients.columns) == ["trial_number", "dataset_id", "community_id"]
    assert len(nodes) == 2
    assert len(patients) == 4
    patient_one = patients[patients["dataset_id"] == 1]
    assert set(patient_one["community_id"]) == {1, 2}
    assert pd.read_csv(
        tmp_path / "selection" / "ensemble_node_community_membership.csv.gz"
    ).equals(nodes)
    assert pd.read_csv(
        tmp_path / "selection" / "ensemble_patient_community_membership.csv.gz"
    ).equals(patients)
    metadata = json.loads(
        (
            tmp_path / "selection" / "ensemble_membership_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["format"] == "sparse_positive_memberships"
    assert metadata["omitted_pair_membership"] == 0
    assert metadata["trials"] == [
        {
            "trial_number": 5,
            "node_count": 2,
            "patient_count": 3,
            "community_count": 2,
            "positive_node_memberships": 2,
            "positive_patient_memberships": 4,
        }
    ]
