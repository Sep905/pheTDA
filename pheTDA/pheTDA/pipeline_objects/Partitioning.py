import networkx as nx
from utils import entropy_count, set_node_community, set_edge_community, associate_sample_to_communities
import numpy as np

class Partitioning:
    def __init__(self, partitioning_params, G, scomplex, seed, sample_ids, dataset):

        self.G = G
        self.scomplex = scomplex
        self.sample_ids = sample_ids

        if 'ties_resolving_strategy' in partitioning_params.keys():
            self.ties_resolving_strategy = partitioning_params['ties_resolving_strategy']

        self.make_partition(partitioning_params, seed)
        self.dataset = dataset

    def make_partition(self, partitioning_params, seed):


        self.partitions = nx.community.louvain_communities(self.G, resolution=partitioning_params['resolution'],weight="weight",seed=seed)

        ##### set the node and the edge attributes to the networkx graph
        set_node_community(self.G, self.partitions)
        set_edge_community(self.G)
    

    def introduce_stratification(self):

        new_dataset_ids_communities = associate_sample_to_communities(self.G, 
                                                                      self.scomplex,
                                                                      self.partitions,
                                                                      self.sample_ids,
                                                                      self.ties_resolving_strategy)
        

        return new_dataset_ids_communities
        


