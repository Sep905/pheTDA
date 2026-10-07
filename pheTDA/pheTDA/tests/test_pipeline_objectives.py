import networkx as nx
import numpy as np
import pandas as pd
import pytest

import optuna_pipeline.pipelines as pipeline_module


class FakeTrial:
    number = 0

    def __init__(self):
        self.user_attrs = {}

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value

    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_int(self, name, low, high, **kwargs):
        return low

    def suggest_float(self, name, low, high, **kwargs):
        return low


class FakeLens:
    def __init__(self, *args):
        self.projections = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        self.lens_function = {"lens": "fake"}


class FakeCovering:
    def __init__(self, *args):
        self.scomplex = {"nodes": {"left": [0, 1], "right": [2, 3]}}
        self.G = nx.Graph()
        self.G.add_node("left", membership=[0, 1])
        self.G.add_node("right", membership=[2, 3])
        self.G.add_edge("left", "right", weight=1)


class FakePartitioning:
    def __init__(self, parameters, graph, scomplex, seed, sample_ids):
        self.G = graph
        self.communities = [["left"], ["right"]]

    def introduce_stratification(self):
        return pd.DataFrame({"dataset_id": range(4), "communities": [1, 1, 2, 2]})


@pytest.fixture
def pipeline_inputs():
    dataset = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3))
    distances = np.array(
        [
            [0.0, 0.1, 0.9, 1.0],
            [0.1, 0.0, 0.8, 0.9],
            [0.9, 0.8, 0.0, 0.1],
            [1.0, 0.9, 0.1, 0.0],
        ]
    )
    return {
        "distance_matrix": distances,
        "dataset": dataset,
        "random_seed": 1,
        "sample_ids": pd.Series(range(4)),
    }


@pytest.fixture
def isolated_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline_module, "Lens_function", FakeLens)
    monkeypatch.setattr(pipeline_module, "Covering", FakeCovering)
    monkeypatch.setattr(pipeline_module, "Partitioning", FakePartitioning)
    monkeypatch.setattr(
        pipeline_module.SemiSupervised_TDA_pipeline,
        "save_pipeline_results",
        lambda *args: None,
    )


@pytest.mark.parametrize(
    ("phenotypes", "expected_score_count"),
    [
        ({"initial_class": pd.Series(["a", "a", "b", "b"])}, 2),
        (
            {
                "final_class": pd.Series([0.1, 0.2, 0.8, 0.9]),
                "final_class_type": "continuous",
            },
            2,
        ),
        (
            {
                "initial_class": pd.Series(["a", "a", "b", "b"]),
                "final_class": pd.Series([0.1, 0.2, 0.8, 0.9]),
                "final_class_type": "continuous",
            },
            3,
        ),
    ],
)
def test_objective_count_follows_supplied_phenotypes(
    tmp_path,
    pipeline_inputs,
    isolated_pipeline,
    phenotypes,
    expected_score_count,
):
    objective = pipeline_module.SemiSupervised_TDA_pipeline(
        **pipeline_inputs,
        results_path=tmp_path,
        **phenotypes,
    )

    trial = FakeTrial()
    scores = objective(trial)

    assert len(scores) == expected_score_count
    assert np.isfinite(scores).all()
    assert trial.user_attrs["pipeline_valid"] is True


def test_pipeline_requires_at_least_one_phenotype(tmp_path, pipeline_inputs):
    with pytest.raises(ValueError, match="at least one"):
        pipeline_module.SemiSupervised_TDA_pipeline(
            **pipeline_inputs, results_path=tmp_path
        )


def test_distance_matrix_must_be_symmetric(tmp_path, pipeline_inputs):
    pipeline_inputs["distance_matrix"][0, 1] = 0.5

    with pytest.raises(ValueError, match="symmetric"):
        pipeline_module.SemiSupervised_TDA_pipeline(
            **pipeline_inputs,
            results_path=tmp_path,
            initial_class=pd.Series(["a", "a", "b", "b"]),
        )


def test_disabled_trial_artifacts_do_not_create_empty_directories(
    tmp_path, pipeline_inputs
):
    pipeline_module.SemiSupervised_TDA_pipeline(
        **pipeline_inputs,
        results_path=tmp_path,
        initial_class=pd.Series(["a", "a", "b", "b"]),
        save_trial_artifacts=False,
    )

    assert list(tmp_path.iterdir()) == []
