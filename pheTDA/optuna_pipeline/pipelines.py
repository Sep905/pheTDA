"""Optuna objective for the complete semi-supervised TDA pipeline."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

from pipeline_objects import Covering, Lens_function, Partitioning
from utils import entropy_count


class SemiSupervised_TDA_pipeline:
    """Run and score the complete lens-to-stratification TDA pipeline."""

    def __init__(
        self,
        distance_matrix,
        dataset,
        random_seed,
        sample_ids,
        results_path,
        initial_class=None,
        initial_class_type="categorical",
        initial_entropy="minimize",
        final_class=None,
        final_class_type="categorical",
        final_entropy="minimize",
        pipeline_type_name="SemiSupervisedTDA",
        save_trial_artifacts=True,
        validate_inputs=True,
    ):
        self.distance_matrix = np.asarray(distance_matrix)
        self.dataset = dataset
        self.seed = random_seed
        self.sample_ids = sample_ids
        self.projection_dimension = 2
        self.pipeline_type_name = pipeline_type_name
        self.save_trial_artifacts = save_trial_artifacts
        self.results_path = Path(results_path)
        self.entropy_targets = []

        self._add_entropy_target(
            "initial", initial_class, initial_class_type, initial_entropy
        )
        self._add_entropy_target("final", final_class, final_class_type, final_entropy)
        if not self.entropy_targets:
            raise ValueError("at least one initial or final phenotype is required")

        if validate_inputs:
            _validate_pipeline_inputs(
                self.distance_matrix,
                self.dataset,
                self.sample_ids,
                self.entropy_targets,
            )
        if self.save_trial_artifacts:
            init_results_directories(self)

    def _add_entropy_target(self, name, values, phenotype_type, direction):
        if values is None:
            return
        if phenotype_type not in {"categorical", "continuous"}:
            raise ValueError(
                f"{name}_class_type must be either 'categorical' or 'continuous'"
            )
        if direction not in {"minimize", "maximize"}:
            raise ValueError(f"{name}_entropy must be either 'minimize' or 'maximize'")
        self.entropy_targets.append(
            {
                "name": name,
                "values": values,
                "type": phenotype_type,
                "direction": direction,
            }
        )

    @property
    def bad_scores(self):
        entropy_scores = tuple(
            np.inf if target["direction"] == "minimize" else -np.inf
            for target in self.entropy_targets
        )
        return (*entropy_scores, -np.inf)

    def __call__(self, trial):
        trial.set_user_attr("pipeline_valid", False)
        lens_name = sample_lens_type(trial, self.dataset)
        lens_parameters = sample_lens_hyperparameters(
            trial,
            lens_name,
            self.dataset,
            self.projection_dimension,
            self.seed,
        )
        mapper_parameters = sample_mapper_hyperparameters(trial, self.seed)
        partition_parameters = {
            **sample_community_hyperparameters(trial),
            **sample_partition_hyperparameters(trial),
        }

        projection_lens = Lens_function(
            lens_name, lens_parameters, self.dataset, self.distance_matrix
        )

        covering = Covering(
            mapper_parameters, projection_lens.projections, self.distance_matrix
        )
        if graph_creation_checks_fail(
            projection_lens.projections, covering.scomplex, covering.G
        ):
            return self.bad_scores

        partition = Partitioning(
            partition_parameters,
            covering.G,
            covering.scomplex,
            self.seed,
            self.sample_ids,
        )
        if node_in_communities_check_fail(partition.G, partition.communities):
            return self.bad_scores

        stratification = partition.introduce_stratification()
        if not _valid_silhouette_labels(stratification["communities"]):
            return self.bad_scores

        graph_entropies = []
        for target in self.entropy_targets:
            graph_entropy, partition.G = entropy_count(
                covering.scomplex,
                target["values"],
                target["type"],
                G=partition.G,
                node_attribute=f"{target['name']}_entropy",
            )
            graph_entropies.append(float(graph_entropy))

        # Graph spread is intentionally retained in utils.graph_utils for
        # possible future use, but it is not computed or optimized here.
        # graph_spread, partition.G = spread_measure(...)

        stratification_silhouette = silhouette_score(
            X=self.distance_matrix,
            labels=stratification["communities"],
            metric="precomputed",
        )

        if self.save_trial_artifacts:
            self.save_pipeline_results(
                trial,
                projection_lens.lens_function,
                partition.G,
                covering.scomplex,
                stratification,
            )
        trial.set_user_attr("pipeline_valid", True)
        scores = (*graph_entropies, float(stratification_silhouette))
        print(*scores)
        return scores

    def save_pipeline_results(self, trial, lens_function, G, scomplex, stratification):
        artifact_id = str(trial.number)
        with (self.results_path_lens / f"{artifact_id}.pickle").open("wb") as output:
            pickle.dump(lens_function, output)
        with (self.results_path_scomplex / f"{artifact_id}_G.pickle").open(
            "wb"
        ) as output:
            pickle.dump(G, output)
        with (self.results_path_scomplex / f"{artifact_id}_s.pickle").open(
            "wb"
        ) as output:
            pickle.dump(scomplex, output)
        stratification.to_excel(
            self.results_path_communities / f"{artifact_id}.xlsx", index=False
        )


def sample_lens_type(trial, dataset):
    lens_names = ["MDS", "AutoEncoder"]
    if min(dataset.shape) >= 2:
        lens_names.insert(0, "PCA")
    if len(dataset) >= 3:
        lens_names.extend(["Spectral", "Isomap", "UMAP"])
    if len(dataset) >= 4:
        lens_names.append("LLE")
    if len(dataset) >= 6:
        lens_names.append("t-SNE")
    return trial.suggest_categorical("lens_function", lens_names)


def sample_lens_hyperparameters(trial, lens_name, dataset, projection_dimension, seed):
    parameters = {"projection_dimension": projection_dimension, "seed": seed}
    maximum_neighbors = max(2, min(150, len(dataset) - 1))

    if lens_name == "Isomap":
        parameters["isomap_n_neighbors"] = trial.suggest_int(
            "isomap_n_neighbors", 2, maximum_neighbors
        )
    elif lens_name == "LLE":
        methods = ["standard", "modified", "ltsa"]
        if len(dataset) >= 7:
            methods.append("hessian")
        method = trial.suggest_categorical("lle_method", methods)
        minimum_neighbors = 6 if method == "hessian" else 3
        parameters.update(
            {
                "lle_n_neighbors": trial.suggest_int(
                    "lle_n_neighbors", minimum_neighbors, maximum_neighbors
                ),
                "lle_max_iter": trial.suggest_int("lle_max_iter", 100, 500, step=100),
                "lle_reg": trial.suggest_float("lle_reg", 1e-4, 1e-2, log=True),
                "lle_method": method,
            }
        )
    elif lens_name == "MDS":
        parameters.update(
            {
                "mds_n_init": trial.suggest_int("mds_n_init", 3, 5),
                "mds_max_iter": trial.suggest_int("mds_max_iter", 100, 500, step=100),
                "mds_eps": trial.suggest_float("mds_eps", 1e-6, 1e-2, log=True),
                "mds_metric": True,
            }
        )
    elif lens_name == "t-SNE":
        parameters.update(
            {
                "tsne_perplexity": trial.suggest_int(
                    "tsne_perplexity", 5, min(50, len(dataset) - 1)
                ),
                "tsne_learning_rate": trial.suggest_float(
                    "tsne_learning_rate", 10.0, 1000.0, log=True
                ),
                "tsne_max_iter": trial.suggest_int(
                    "tsne_max_iter", 500, 1500, step=250
                ),
            }
        )
    elif lens_name == "UMAP":
        parameters.update(
            {
                "umap_n_neighbors": trial.suggest_int(
                    "umap_n_neighbors", 2, maximum_neighbors
                ),
                "umap_min_dist": trial.suggest_float(
                    "umap_min_dist", 0.0, 0.9, step=0.1
                ),
            }
        )
    elif lens_name == "AutoEncoder":
        use_dropout = trial.suggest_categorical(
            "autoencoder_use_dropout", [True, False]
        )
        parameters.update(
            {
                "input_dimension": dataset.shape[1],
                "autoencoder_use_batchnorm": trial.suggest_categorical(
                    "autoencoder_use_batchnorm", [True, False]
                ),
                "autoencoder_use_dropout": use_dropout,
                "autoencoder_activation": trial.suggest_categorical(
                    "autoencoder_activation", ["ReLU", "sigmoid", "tanh"]
                ),
                "autoencoder_learning_rate": trial.suggest_float(
                    "autoencoder_learning_rate", 1e-4, 1e-1, log=True
                ),
                "autoencoder_weight_decay": trial.suggest_float(
                    "autoencoder_weight_decay", 1e-6, 1e-2, log=True
                ),
                "autoencoder_batch_size": trial.suggest_categorical(
                    "autoencoder_batch_size", [32, 64, 128]
                ),
                "autoencoder_epochs": trial.suggest_categorical(
                    "autoencoder_epochs", [100, 250, 500, 1000]
                ),
                "autoencoder_num_layers": trial.suggest_int(
                    "autoencoder_num_layers", 2, 5
                ),
            }
        )
        parameters["autoencoder_dropout_probability"] = (
            trial.suggest_float("autoencoder_dropout_probability", 0.0, 0.4, step=0.1)
            if use_dropout
            else 0.0
        )

    return parameters


def sample_mapper_hyperparameters(trial, seed):
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
    parameters = {
        "number_of_intervals": trial.suggest_int("number_of_intervals", 8, 22, step=2),
        "percentage_overlap": trial.suggest_float(
            "percentage_overlap", 0.2, 0.5, step=0.1
        ),
        "cluster_method_name": cluster_method,
        "seed": seed,
    }
    if cluster_method == "DBSCAN":
        parameters.update(
            {
                "min_points": trial.suggest_int("min_points", 1, 5),
                "epsilon": trial.suggest_float("epsilon", 0.1, 0.5, step=0.1),
            }
        )
    else:
        parameters["n_clusters"] = trial.suggest_int("n_clusters", 2, 5)
    return parameters


def sample_community_hyperparameters(trial):
    return {
        "louvain_resolution": trial.suggest_categorical(
            "louvain_resolution", [1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10]
        )
    }


def sample_partition_hyperparameters(trial):
    return {
        "ties_resolving_strategy": trial.suggest_categorical(
            "ties_resolving_strategy",
            ["new_community_for_ties", "size", "centrality_ensemble"],
        )
    }


def graph_creation_checks_fail(projections, scomplex, G):
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return True
    graph_samples = {
        sample
        for memberships in scomplex.get("nodes", {}).values()
        for sample in memberships
    }
    expected_samples = set(range(projections.shape[0]))
    return graph_samples != expected_samples


def node_in_communities_check_fail(G, communities):
    assigned_nodes = {node for community in communities for node in community}
    return assigned_nodes != set(G.nodes)


def _valid_silhouette_labels(labels):
    if labels.isna().any():
        return False
    number_of_labels = labels.nunique()
    return 2 <= number_of_labels < len(labels)


def _validate_pipeline_inputs(distance_matrix, dataset, sample_ids, entropy_targets):
    number_of_samples = len(dataset)
    if number_of_samples < 3:
        raise ValueError("the pipeline requires at least three samples")
    if distance_matrix.shape != (number_of_samples, number_of_samples):
        raise ValueError("distance_matrix must be square and aligned with dataset")
    if len(sample_ids) != number_of_samples:
        raise ValueError("dataset and sample_ids must have equal length")
    for target in entropy_targets:
        if len(target["values"]) != number_of_samples:
            raise ValueError(
                f"dataset and {target['name']} phenotype must have equal length"
            )
    if not np.isfinite(distance_matrix).all():
        raise ValueError("distance_matrix must contain only finite values")
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("distance_matrix must be symmetric")
    if not np.allclose(np.diag(distance_matrix), 0):
        raise ValueError("distance_matrix diagonal must be zero")


def init_results_directories(pipeline):
    run_directory = (
        pipeline.results_path / f"{pipeline.seed}_{pipeline.pipeline_type_name}"
    )
    pipeline.results_path_lens = run_directory / "lens"
    pipeline.results_path_scomplex = run_directory / "scomplex"
    pipeline.results_path_communities = run_directory / "communities"
    for directory in (
        pipeline.results_path_lens,
        pipeline.results_path_scomplex,
        pipeline.results_path_communities,
    ):
        directory.mkdir(parents=True, exist_ok=True)
