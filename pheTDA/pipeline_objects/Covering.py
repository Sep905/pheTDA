"""Mapper covering and graph construction."""

import inspect

import kmapper as km
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids


class Covering:
    def __init__(self, covering_parameters, projections, distance_matrix):
        self.mapper = km.KeplerMapper()
        clusterer = self._clusterer(covering_parameters)
        self.scomplex = self.mapper.map(
            lens=projections,
            X=distance_matrix,
            cover=km.Cover(
                n_cubes=covering_parameters["number_of_intervals"],
                perc_overlap=covering_parameters["percentage_overlap"],
            ),
            clusterer=clusterer,
            precomputed=True,
            remove_duplicate_nodes=True,
        )
        self.G = km.adapter.to_nx(self.scomplex)

        # KeplerMapper edges represent overlap. Store the shared-sample count as
        # edge strength for weighted community-detection algorithms.
        for source, target in self.G.edges:
            source_members = self.scomplex["nodes"][source]
            target_members = self.scomplex["nodes"][target]
            self.G[source][target]["weight"] = len(
                set(source_members).intersection(target_members)
            )

    @staticmethod
    def _clusterer(parameters):
        method = parameters["cluster_method_name"]
        if method == "DBSCAN":
            return DBSCAN(
                metric="precomputed",
                min_samples=parameters["min_points"],
                eps=parameters["epsilon"],
            )
        if method.startswith("agglomerative_"):
            distance_argument = (
                "metric"
                if "metric" in inspect.signature(AgglomerativeClustering).parameters
                else "affinity"
            )
            return AgglomerativeClustering(
                n_clusters=parameters["n_clusters"],
                linkage=method.split("_", maxsplit=1)[1],
                **{distance_argument: "precomputed"},
            )
        if method == "k_medoids":
            return KMedoids(
                metric="precomputed",
                n_clusters=parameters["n_clusters"],
                init="heuristic",
                random_state=parameters["seed"],
            )
        raise ValueError(f"unknown Mapper cluster method: {method!r}")
