"""Projection (lens) functions used by the Mapper pipeline."""

import inspect

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)


class Lens_function:
    def __init__(self, lens_name, lens_parameters, dataset, distance_matrix):
        self.lens_function = self._create_lens(lens_name, lens_parameters)
        fit_data = (
            dataset
            if lens_name in {"PCA", "AutoEncoder"}
            else self._fit_data(lens_name, distance_matrix)
        )
        projections = self.lens_function.fit_transform(fit_data)
        if hasattr(projections, "detach"):
            projections = projections.detach().cpu().numpy()
        self.projections = np.asarray(projections)

    @staticmethod
    def _fit_data(lens_name, distance_matrix):
        if lens_name != "Spectral":
            return distance_matrix

        distance_matrix = np.asarray(distance_matrix, dtype=float)
        positive_distances = distance_matrix[distance_matrix > 0]
        scale = np.median(positive_distances) if positive_distances.size else 1.0
        affinity = np.exp(-(distance_matrix**2) / (2 * scale**2))
        np.fill_diagonal(affinity, 1.0)
        return affinity

    @staticmethod
    def _create_lens(lens_name, parameters):
        common = {
            "n_components": parameters["projection_dimension"],
            "random_state": parameters["seed"],
        }
        if lens_name == "PCA":
            return PCA(**common)
        if lens_name == "LLE":
            return LocallyLinearEmbedding(
                **common,
                n_neighbors=parameters["lle_n_neighbors"],
                max_iter=parameters["lle_max_iter"],
                method=parameters["lle_method"],
                reg=parameters["lle_reg"],
            )
        if lens_name == "Spectral":
            return SpectralEmbedding(**common, affinity="precomputed")
        if lens_name == "MDS":
            mds_parameters = {
                **common,
                "n_init": parameters["mds_n_init"],
                "max_iter": parameters["mds_max_iter"],
                "eps": parameters["mds_eps"],
            }
            if "metric_mds" in inspect.signature(MDS).parameters:
                mds_parameters.update(
                    {
                        "metric_mds": parameters["mds_metric"],
                        "metric": "precomputed",
                        "init": "random",
                    }
                )
            else:
                mds_parameters.update(
                    {
                        "metric": parameters["mds_metric"],
                        "dissimilarity": "precomputed",
                    }
                )
            return MDS(**mds_parameters)
        if lens_name == "Isomap":
            return Isomap(
                n_components=parameters["projection_dimension"],
                path_method="D",
                metric="precomputed",
                n_neighbors=parameters["isomap_n_neighbors"],
            )
        if lens_name == "t-SNE":
            iteration_argument = (
                "max_iter"
                if "max_iter" in inspect.signature(TSNE).parameters
                else "n_iter"
            )
            return TSNE(
                **common,
                init="random",
                metric="precomputed",
                perplexity=parameters["tsne_perplexity"],
                learning_rate=parameters["tsne_learning_rate"],
                **{iteration_argument: parameters["tsne_max_iter"]},
            )
        if lens_name == "UMAP":
            return umap.UMAP(
                **common,
                metric="precomputed",
                n_neighbors=parameters["umap_n_neighbors"],
                min_dist=parameters["umap_min_dist"],
            )
        if lens_name == "AutoEncoder":
            from .nn import AutoEncoder

            return AutoEncoder(
                input_dim=parameters["input_dimension"],
                num_layers=parameters["autoencoder_num_layers"],
                use_batchnorm=parameters["autoencoder_use_batchnorm"],
                use_dropout=parameters["autoencoder_use_dropout"],
                dropout_prob=parameters["autoencoder_dropout_probability"],
                activation_function=parameters["autoencoder_activation"],
                learning_rate=parameters["autoencoder_learning_rate"],
                w_decay=parameters["autoencoder_weight_decay"],
                batch_size=parameters["autoencoder_batch_size"],
                epochs=parameters["autoencoder_epochs"],
                random_state=parameters["seed"],
            )
        raise ValueError(f"unknown lens function: {lens_name!r}")
