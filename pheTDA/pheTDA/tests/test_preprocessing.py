import numpy as np
import pandas as pd
import pytest

from utils.prepro import distance_matrix_computation


def test_mixed_preprocessing_returns_finite_symmetric_distances(tmp_path):
    dataset = pd.DataFrame(
        {
            "age": [20.0, 40.0, 60.0],
            "group": ["a", "b", "a"],
            "flag": [0, 1, 0],
        }
    )

    distances, projected = distance_matrix_computation(
        dataset,
        continuous_features=["age"],
        categorical_features=["group"],
        binary_features=["flag"],
        results_path=tmp_path,
    )

    assert distances.shape == (3, 3)
    assert np.isfinite(distances).all()
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0)
    assert not projected.isna().any().any()
    assert (tmp_path / "distance_matrix.npy").exists()
    assert (tmp_path / "dataset_preprocessed.csv").exists()


def test_missing_features_are_rejected_instead_of_imputed(tmp_path):
    dataset = pd.DataFrame({"age": [20.0, np.nan, 60.0]})

    with pytest.raises(ValueError, match="missing values"):
        distance_matrix_computation(
            dataset,
            continuous_features=["age"],
            categorical_features=[],
            binary_features=[],
            results_path=tmp_path,
        )


def test_feature_cannot_appear_in_multiple_groups(tmp_path):
    dataset = pd.DataFrame({"feature": [0, 1, 0]})

    with pytest.raises(ValueError, match="multiple groups"):
        distance_matrix_computation(
            dataset,
            continuous_features=["feature"],
            categorical_features=[],
            binary_features=["feature"],
            results_path=tmp_path,
        )


def test_unknown_feature_is_rejected(tmp_path):
    dataset = pd.DataFrame({"known": [0, 1, 2]})

    with pytest.raises(ValueError, match="absent"):
        distance_matrix_computation(
            dataset,
            continuous_features=["unknown"],
            categorical_features=[],
            binary_features=[],
            results_path=tmp_path,
        )
