import kmapper as km
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from datetime import datetime
from collections import defaultdict
from scipy.sparse import issparse, hstack
import itertools
import pandas as pd

class Covering():
    def __init__(self, covering_params, projections, distance_matrix, sample_ids,initial_class, dataset):
        self.mapper = km.KeplerMapper()
        self.apply_cover(covering_params, projections, distance_matrix, sample_ids,initial_class, dataset)

    def apply_cover(self, covering_params, projections, distance_matrix, sample_ids,initial_class, dataset):

        if covering_params['cluster_method_name'] == "DBSCAN":

            cluster_method = DBSCAN(metric="precomputed",
                                    min_samples=covering_params['minPoints_val'],
                                    eps=covering_params['eps_val'],
                                    n_jobs=1)


        elif "agglomerative" in covering_params['cluster_method_name']:
            linkage = covering_params['cluster_method_name'].split("_")[1]
            cluster_method = AgglomerativeClustering(metric="precomputed",
                                                         n_clusters=covering_params['n_clusters'],
                                                         linkage=linkage)

        self.scomplex = map_(lens = projections,  mapper=self.mapper,
                                   X = distance_matrix,  
                                   clusterer=cluster_method,
                                   cover=km.Cover(n_cubes=covering_params['number_of_interval'], perc_overlap=np.round(covering_params['percentage_overlap'],3)), 
                                   precomputed=True,
                                   remove_duplicate_nodes = True)

    
        self.G = km.adapter.to_nx(self.scomplex)



#### modified keplermapper to avoid data loss   
def map_(
        lens, mapper,
        X=None,
        clusterer=None,
        cover=None,
        nerve=None,
        precomputed=False,
        remove_duplicate_nodes=False,
    ):
        """Apply Mapper algorithm on this projection and build a simplicial complex. Returns a dictionary with nodes and links.

        Parameters
        ----------
        lens: Numpy Array
            Lower dimensional representation of data. In general will be output of `fit_transform`.

        X: Numpy Array
            Original data or data to run clustering on. If `None`, then use `lens` as default. X can be a SciPy sparse matrix.

        clusterer: Default: DBSCAN
            Scikit-learn API compatible clustering algorithm. Must provide `fit` and `predict`.

        cover: kmapper.Cover
            Cover scheme for lens. Instance of kmapper.cover providing methods `fit` and `transform`.

        nerve: kmapper.Nerve
            Nerve builder implementing `__call__(nodes)` API

        precomputed : Boolean
            Tell Mapper whether the data that you are clustering on is a precomputed distance matrix. If set to
            `True`, the assumption is that you are also telling your `clusterer` that `metric='precomputed'` (which
            is an argument for DBSCAN among others), which
            will then cause the clusterer to expect a square distance matrix for each hypercube. `precomputed=True` will give a square matrix
            to the clusterer to fit on for each hypercube.

        remove_duplicate_nodes: Boolean
            Removes duplicate nodes before edges are determined. A node is considered to be duplicate
            if it has exactly the same set of points as another node.

        nr_cubes: Int

            .. deprecated:: 1.1.6

                define Cover explicitly in future versions

            The number of intervals/hypercubes to create. Default = 10.

        overlap_perc: Float
            .. deprecated:: 1.1.6

                define Cover explicitly in future versions

            The percentage of overlap "between" the intervals/hypercubes. Default = 0.1.



        """

        start = datetime.now()
        
        mapper.cover = cover 
        nerve = nerve or GraphNerve()

        nodes = defaultdict(list)
        meta = defaultdict(list)
        graph = {}

        # If inverse image is not provided, we use the projection as the inverse image (suffer projection loss)
        if X is None:
            X = lens

        if mapper.verbose > 0:
            print(
                "Mapping on data shaped %s using lens shaped %s\n"
                % (str(X.shape), str(lens.shape))
            )

        # Prefix'ing the data with an ID column
        ids = np.array([x for x in range(lens.shape[0])])
        lens = np.c_[ids, lens]
        if issparse(X):
            X = hstack([ids[np.newaxis].T, X], format="csr")
        else:
            X = np.c_[ids, X]

        # Cover scheme defines a list of elements
        bins = mapper.cover.fit(lens)

        # Algo's like K-Means, have a set number of clusters. We need this number
        # to adjust for the minimal number of samples inside an interval before
        # we consider clustering or skipping it.
        cluster_params = clusterer.get_params()

        min_cluster_samples = None
        for parameter in ["n_clusters", "min_cluster_size", "min_samples"]:
            value = cluster_params.get(parameter)
            if value and isinstance(value, int):
                min_cluster_samples = value
                break
        if not min_cluster_samples:
            min_cluster_samples = 2


        if mapper.verbose > 1:
            print(
                "Minimal points in hypercube before clustering: {}".format(
                    min_cluster_samples
                )
            )

        # Subdivide the projected data X in intervals/hypercubes with overlap
        if mapper.verbose > 0:
            bins = list(bins)  # extract list from generator
            total_bins = len(bins)
            print("Creating %s hypercubes." % total_bins)


        for i, hypercube in enumerate(mapper.cover.transform(lens)):

            # If at least min_cluster_samples samples inside the hypercube
            if hypercube.shape[0] >= min_cluster_samples:
                # Cluster the data point(s) in the cube, skipping the id-column
                # Note that we apply clustering on the inverse image (original data samples) that fall inside the cube.
                ids = [int(nn) for nn in hypercube[:, 0]]
                X_cube = X[ids]

                fit_data = X_cube[:, 1:]
                if precomputed:
                    fit_data = fit_data[:, ids]

                cluster_predictions = clusterer.fit_predict(fit_data)

                if mapper.verbose > 1:
                    print(
                        "   > Found %s clusters in hypercube %s."
                        % (
                            np.unique(
                                cluster_predictions[cluster_predictions > -1]
                            ).shape[0],
                            i,
                        )
                    )

                for pred in np.unique(cluster_predictions):
                    # if not predicted as noise
                    if not np.isnan(pred):
                        cluster_id = "cube{}_cluster{}".format(i, int(pred)+1)

                        nodes[cluster_id] = (
                            hypercube[:, 0][cluster_predictions == pred]
                            .astype(int)
                            .tolist()
                        )

            else:
                cluster_id = "cube{}_cluster{}".format(i, 0)

                nodes[cluster_id] = (
                            hypercube[:, 0]
                            .astype(int)
                            .tolist()
                        )

            # else:
            #     if mapper.verbose > 1:
            #         print("Cube_%s is empty.\n" % (i))

        if remove_duplicate_nodes:
            nodes = mapper._remove_duplicate_nodes(nodes)
        
        links, simplices = nerve.compute(nodes)

        graph["nodes"] = nodes
        graph["links"] = links
        graph["simplices"] = simplices
        graph["meta_data"] = {
            "projection": mapper.projection if mapper.projection else "custom",
            "n_cubes": mapper.cover.n_cubes,
            "perc_overlap": mapper.cover.perc_overlap,
            "clusterer": str(clusterer),
            "scaler": str(mapper.scaler),
            "nerve_min_intersection": nerve.min_intersection
        }
        graph["meta_nodes"] = meta

        if mapper.verbose > 0:
            mapper._summary(graph, str(datetime.now() - start))

        return graph

class Nerve:
    """Base class for implementations of a nerve finder to build a Mapper complex."""

    def __init__(self):
        pass

    def compute(self, nodes, links):
        raise NotImplementedError()

class GraphNerve(Nerve):
    """Creates the 1-skeleton of the Mapper complex.

    Parameters
    -----------

    min_intersection: int, default is 1
        Minimum intersection considered when computing the nerve. An edge will be created only when the intersection between two nodes is greater than or equal to `min_intersection`
    """

    def __init__(self, min_intersection=1):
            self.min_intersection = min_intersection

    def __repr__(self):
            return "GraphNerve(min_intersection={})".format(self.min_intersection)

        
    def compute(self, nodes):
            """Helper function to find edges of the overlapping clusters.

            Parameters
            ----------
            nodes:
                A dictionary with entires `{node id}:{list of ids in node}`

            Returns
            -------
            edges:
                A 1-skeleton of the nerve (intersecting  nodes)

            simplicies:
                Complete list of simplices

            """

            result = defaultdict(list)

            # Create links when clusters from different hypercubes have members with the same sample id.
            candidates = itertools.combinations(nodes.keys(), 2)
            for candidate in candidates:
                # if there are non-unique members in the union
                if ( len(set(nodes[candidate[0]]).intersection(nodes[candidate[1]]))  >= self.min_intersection ):
                    result[candidate[0]].append(candidate[1])

            edges = [[x, end] for x in result for end in result[x]]
            simplices = [[n] for n in nodes] + edges
            return result, simplices