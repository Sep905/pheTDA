import numpy as np
import pandas as pd

from optuna_pipeline.pipelines import sample_lens_hyperparameters
from pipeline_objects.Lens_function import Lens_function


class RecordingTrial:
    def __init__(self):
        self.names = []

    def suggest_categorical(self, name, choices):
        self.names.append(name)
        return choices[0]

    def suggest_int(self, name, low, high, **kwargs):
        self.names.append(name)
        assert low <= high
        return low

    def suggest_float(self, name, low, high, **kwargs):
        self.names.append(name)
        assert low <= high
        return low


def test_every_lens_hyperparameter_has_a_lens_specific_name():
    dataset = pd.DataFrame(np.zeros((12, 4)))
    prefixes = {
        "Isomap": "isomap_",
        "LLE": "lle_",
        "MDS": "mds_",
        "t-SNE": "tsne_",
        "UMAP": "umap_",
        "AutoEncoder": "autoencoder_",
    }

    for lens, prefix in prefixes.items():
        trial = RecordingTrial()
        parameters = sample_lens_hyperparameters(trial, lens, dataset, 2, 3)

        assert trial.names
        assert all(name.startswith(prefix) for name in trial.names)
        if lens == "MDS":
            assert parameters["mds_metric"] is True
            assert "mds_metric" not in trial.names


def test_pca_lens_returns_two_dimensional_projection():
    dataset = pd.DataFrame(np.random.default_rng(3).normal(size=(10, 4)))
    distances = np.linalg.norm(
        dataset.to_numpy()[:, None, :] - dataset.to_numpy()[None, :, :], axis=2
    )

    lens = Lens_function(
        "PCA",
        {"projection_dimension": 2, "seed": 7},
        dataset,
        distances,
    )

    assert lens.projections.shape == (10, 2)
    assert np.isfinite(lens.projections).all()


def test_isomap_and_umap_neighbor_parameters_do_not_share_a_name():
    dataset = pd.DataFrame(np.zeros((12, 4)))
    isomap_trial = RecordingTrial()
    umap_trial = RecordingTrial()

    sample_lens_hyperparameters(isomap_trial, "Isomap", dataset, 2, 3)
    sample_lens_hyperparameters(umap_trial, "UMAP", dataset, 2, 3)

    assert "isomap_n_neighbors" in isomap_trial.names
    assert "umap_n_neighbors" in umap_trial.names
    assert set(isomap_trial.names).isdisjoint(umap_trial.names)
