"""Dataset preprocessing and mixed-type distance computation."""

from pathlib import Path

import gower
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def _validate_feature_columns(dataset, feature_groups):
    columns = [column for group in feature_groups for column in group]
    duplicates = [column for column in set(columns) if columns.count(column) > 1]
    if duplicates:
        raise ValueError(f"feature columns occur in multiple groups: {duplicates}")
    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"feature columns are absent from the dataset: {missing}")
    if not columns:
        raise ValueError("at least one feature column must be supplied")
    return columns


def _encoded_projection_data(dataset, continuous_features, categorical_features):
    frames = []
    if continuous_features:
        scaled = StandardScaler().fit_transform(dataset[continuous_features])
        frames.append(pd.DataFrame(scaled, columns=continuous_features))
    if categorical_features:
        frames.append(
            pd.get_dummies(
                dataset[categorical_features].astype(str),
                columns=categorical_features,
                drop_first=False,
                dtype=float,
            ).reset_index(drop=True)
        )
    return pd.concat(frames, axis=1)


def distance_matrix_computation(
    dataset,
    continuous_features,
    categorical_features,
    binary_features,
    results_path,
):
    """Compute a distance matrix appropriate to the supplied feature types.

    Standardized Euclidean distance is used for continuous-only data, Jaccard
    distance for categorical-only data, and Gower distance for mixed data.
    Input features must not contain missing values. The returned feature frame
    is numeric and suitable for all lens functions.
    """
    feature_columns = _validate_feature_columns(
        dataset, [continuous_features, categorical_features, binary_features]
    )
    categorical_and_binary = [*categorical_features, *binary_features]
    features = dataset.loc[:, feature_columns].copy()
    columns_with_missing_values = features.columns[features.isna().any()].tolist()
    if columns_with_missing_values:
        raise ValueError(
            "input features contain missing values in columns: "
            f"{columns_with_missing_values}"
        )
    projection_data = _encoded_projection_data(
        features, continuous_features, categorical_and_binary
    )

    has_continuous = bool(continuous_features)
    has_categorical = bool(categorical_and_binary)
    if has_continuous and not has_categorical:
        distance_matrix = pairwise_distances(
            projection_data.to_numpy(), metric="euclidean"
        )
    elif has_categorical and not has_continuous:
        distance_matrix = pairwise_distances(
            projection_data.to_numpy(dtype=bool), metric="jaccard"
        )
    else:
        gower_data = features[[*continuous_features, *categorical_and_binary]]
        categorical_mask = [False] * len(continuous_features) + [True] * len(
            categorical_and_binary
        )
        distance_matrix = gower.gower_matrix(gower_data, cat_features=categorical_mask)

    # Remove small floating-point asymmetries/diagonal residuals so the matrix
    # satisfies the precomputed-distance requirements of downstream estimators.
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0.0)

    results_directory = Path(results_path)
    results_directory.mkdir(parents=True, exist_ok=True)
    np.save(results_directory / "distance_matrix.npy", distance_matrix)
    projection_data.to_csv(results_directory / "dataset_preprocessed.csv", index=False)
    return distance_matrix, projection_data
