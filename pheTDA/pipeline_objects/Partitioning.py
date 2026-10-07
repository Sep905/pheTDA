from networkx.algorithms.community import louvain_communities

from utils import (
    associate_sample_to_communities,
    set_edge_community,
    set_node_community,
)


class Partitioning:
    def __init__(self, partitioning_params, G, scomplex, seed, sample_ids):
        self.G = G
        self.scomplex = scomplex
        self.sample_ids = sample_ids
        self.ties_resolving_strategy = partitioning_params["ties_resolving_strategy"]
        self.make_partition(partitioning_params, seed)

    def make_partition(self, partitioning_params, seed):
        self.communities = louvain_communities(
            self.G,
            weight="weight",
            resolution=partitioning_params["louvain_resolution"],
            seed=seed,
        )
        set_node_community(self.G, self.communities)
        set_edge_community(self.G)

    def introduce_stratification(self):
        return associate_sample_to_communities(
            self.G,
            self.scomplex,
            self.communities,
            self.sample_ids,
            self.ties_resolving_strategy,
        )
