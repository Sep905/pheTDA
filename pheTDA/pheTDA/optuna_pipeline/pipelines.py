import optuna
from pipeline_objects import Lens_function, Covering, Partitioning

import networkx as nx
from utils import entropy_count, spread_measure
from sklearn.metrics import silhouette_score
import pickle
import numpy as np
import pandas as pd

import os


class SemiSupervised_TDA_pipeline:
    def __init__(self, distance_matrix, dataset_features,  random_seed, dataset, sample_ids, initial_class, initial_class_type, 
                 results_path, entropy):
        self.distance_matrix = distance_matrix
        self.dataset_features = dataset_features
        self.dataset = dataset
        self.seed = random_seed
        self.sample_ids = sample_ids
        self.initial_class = initial_class
        self.initial_class_type = initial_class_type
        self.projection_dimension = 2
        self.entropy = entropy

        self.results_path = results_path
        init_results_directories(self)


    def __call__(self, trial):

        # define the bad situation score for pruning the trials
        if self.entropy == "minimize":
            bad_scores = (float('inf'),float('-inf'))
        elif self.entropy == "maximize":
            bad_scores = (float('-inf'),float('-inf'))

        lens = sample_lens_type(trial)
        lens_dict_hyperparameters = sample_lens_hyperparameters(trial, self.dataset_features, self.projection_dimension, self.seed)
        mapper_dict_hyperparameters = sample_mapper_hyperparameters(trial)
        communities_dict_hyperparameters = sample_community_hyperparameters(trial)
        partition_dict_hyperparameters = sample_partition_hyperparameters(trial)

        #### Create the lens and project
        projection_lens = Lens_function(lens, lens_dict_hyperparameters, self.dataset_features, self.distance_matrix)

        #### Create covering and apply to obtain the graph (simplicial comples and networkx graph)
        covering_and_graph = Covering(mapper_dict_hyperparameters, projection_lens.projections, self.distance_matrix, self.sample_ids, self.initial_class, self.dataset_features)

        #### Graph creation checks
        #if graph_creation_checks_fail(projection_lens.projections, covering_and_graph.scomplex, covering_and_graph.G):
        #    return bad_scores

        covering_and_graph.G, covering_and_graph.scomplex = define_node_edge_attribute(covering_and_graph.G, covering_and_graph.scomplex, self.dataset_features)


        #### Make paritions with community detection algorithm 
        partition = Partitioning(communities_dict_hyperparameters | partition_dict_hyperparameters,
                                 covering_and_graph.G, covering_and_graph.scomplex, self.seed, self.sample_ids, self.dataset)

        if node_in_communities_check_fail(partition.G, partition.partitions):
            return bad_scores


        #### Introduce the new stratification
        new_dataset_ids_communities = partition.introduce_stratification()


        ### Compute objectives
        # 1 -> graph entropy (weighted node entropy by node size) according to node attribute prevalence
        graph_entropy,partition.G= entropy_count(covering_and_graph.scomplex, partition.G, self.initial_class, self.initial_class_type)

        # 2 -> cluster goodness
        silhouette_score_after_ties_adj = silhouette_score(X=self.distance_matrix, labels=new_dataset_ids_communities['communities'], metric="precomputed")

        ### Save results
        self.save_pipeline_results(trial, projection_lens.lens_function, partition.G, covering_and_graph.scomplex, new_dataset_ids_communities)

        return graph_entropy, silhouette_score_after_ties_adj

    def save_pipeline_results(self, trial, lens_function, G, scomplex, new_dataset_ids_communities):
        # Save the optuna results with their trial numbers
        artifact_id = trial.number 

        # save fitted lens function
        with open(self.results_path_lens + "/" + str(artifact_id) + ".pickle", "wb") as fout:
            pickle.dump(lens_function, fout)

        # save networkx and scomplex generated
        with open(self.results_path_scomplex + "/" + str(artifact_id) + "_G.pickle", "wb") as fout:
            pickle.dump(G, fout)
        with open(self.results_path_scomplex + "/" + str(artifact_id) + "_s.pickle", "wb") as fout:
            pickle.dump(scomplex, fout)

        # save communities df
        new_dataset_ids_communities.to_excel(self.results_path_communities + "/" + str(artifact_id) + ".xlsx",index=False)

def sample_lens_type(trial):
    lens = trial.suggest_categorical("lens_function",['PCA','MDS','Isomap','t-SNE','UMAP','AutoEncoder'])
    return lens

def sample_lens_hyperparameters(trial, dataset_features, projection_dimension, seed):
    # Isomap
    n_neighbors_isomap = trial.suggest_int("n_neighbors_isomap", 15, 150, step=5) 
    # UMAP
    n_neighbors_umap = trial.suggest_int("n_neighbors_umap", 5, 150, step=5) 
    # MDS
    n_init = trial.suggest_int("n_iterations", 3, 5, step=1) 
    max_it = trial.suggest_int("max_it_mds", 100, 500, step=200)
    eps = trial.suggest_int("eps_mds", 100, 500, step=200)
    metric =  trial.suggest_categorical('metric',[True,False])
    # t-SNE
    perplexity = trial.suggest_int("perplexity", 5, 50, step=5)
    learning_rate_tsne = trial.suggest_int("learning_rate_tsne", 10, 1010, step=50)
    n_iter = trial.suggest_int("n_iter", 500, 1500, step=500)
    # UMAP
    min_dist = trial.suggest_float("min_dist", 0.1, 0.9, step=0.1)
    # AutoEncoder
    use_batchnorm =  trial.suggest_categorical('use_batchnorm',[True,False])
    use_dropout =  trial.suggest_categorical('use_dropout',[True,False])
    activation_function =  trial.suggest_categorical('activation_function',['ReLU','sigmoid','tanh'])
    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.4, step=0.1)
    learning_rate_ae = trial.suggest_float("learning_rate_ae", 1e-4, 1e-1, log=True)
    w_decay = trial.suggest_float("w_decay", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_categorical("epochs", [100, 250, 500, 1000])
    if dataset_features.shape[1]<=16:
        num_layers = 3
    elif dataset_features.shape[1]>16 and dataset_features.shape[1]<=32:
        num_layers = trial.suggest_int("num_layers", 3, 4, step=1)
    else:
        num_layers = trial.suggest_int("num_layers", 3, 5, step=1)

    return {"n_neighbors_isomap":n_neighbors_isomap, "n_neighbors_umap":n_neighbors_umap, 
            "n_init":n_init, "max_it_mds":max_it, "eps_mds":eps, "metric":metric,
            "perplexity":perplexity, "learning_rate_tsne":learning_rate_tsne, "n_iter":n_iter,
            "min_dist":min_dist,
            "use_batchnorm":use_batchnorm, "use_dropout":use_dropout, "activation_function":activation_function,
            "dropout_prob":dropout_prob, "learning_rate_ae":learning_rate_ae, "w_decay":w_decay, "batch_size":batch_size, 
            "epochs":epochs, "num_layers":num_layers,
            "projection_dimension":projection_dimension, "seed":seed}

def sample_mapper_hyperparameters(trial):
    # Mapper
    number_of_interval = trial.suggest_int("number_of_interval", 8, 22, step = 2)
    percentage_overlap = trial.suggest_float("percentage_overlap", 0.2, 0.5, step = 0.1)

    cluster_method = trial.suggest_categorical("cluster_method", ['DBSCAN','agglomerative_average','agglomerative_complete',
                                                                    'agglomerative_single'])
    # and hyperparameters
    # DBSCAN
    minPoints_val = trial.suggest_int("minPoints_val", 1, 5, step = 1)
    eps_val = trial.suggest_float("eps_val", 0.1, 0.5, step = 0.1)
    # agglomerative and k-medoids
    n_clusters = trial.suggest_int("n_clusters", 2, 5, step = 1)


    return {"number_of_interval":number_of_interval, "percentage_overlap":percentage_overlap, "cluster_method_name":cluster_method,
                "minPoints_val":minPoints_val, "eps_val":eps_val, "n_clusters":n_clusters}

def sample_community_hyperparameters(trial):

    resolution = trial.suggest_categorical("resolution", [1e-3, 5e-2, 1e-2, 5e-1, 1e-1, 1, 5, 10])

    return { "resolution":resolution}

def sample_partition_hyperparameters(trial):
    ties_resolving_strategy = trial.suggest_categorical("ties_resolving_strategy", ['size','centrality_ensemble'])
    return {"ties_resolving_strategy":ties_resolving_strategy}


def graph_creation_checks_fail(projections, scomplex, G):
    # 1) are all the sampples present in the graph created?
    all_patient_ = []
    for key,values in scomplex["nodes"].items():
        for v in values:
            all_patient_.append(v)

    n_sample_not_in_graph = projections.shape[0] - len(set(all_patient_))

    if n_sample_not_in_graph != 0 or G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return True
    else:
        # 2) is the graph created a unique component (connected) ?
        connected_component = nx.is_connected(G)
        if not(connected_component):
            return True
        else:
            return False
        
def node_in_communities_check_fail(G,partitions):

    # 1) is there any sample that does not appear in a community?
    unique_nodes_without_assigned_community = G.number_of_nodes() - len(set(node_ for comm_ in partitions for node_ in comm_))

    # 2) are there 2 or more communities?

    if unique_nodes_without_assigned_community != 0 or len(partitions)<=1:
        return True
    else:
        return False


def init_results_directories(pipeline):
    pipeline.results_path_lens = pipeline.results_path  + "/lens"
    pipeline.results_path_scomplex = pipeline.results_path  + "/scomplex"
    pipeline.results_path_communities = pipeline.results_path  + "/communities"

    if not(os.path.exists(pipeline.results_path_lens)):
        os.mkdir(pipeline.results_path_lens)

    if not(os.path.exists(pipeline.results_path_scomplex)):
        os.mkdir(pipeline.results_path_scomplex)

    if not(os.path.exists(pipeline.results_path_communities)):
        os.mkdir(pipeline.results_path_communities)

    
def define_node_edge_attribute(G, scomplex, dataset):
    """
    Optimized version using pre-computed distance matrix.
    
    Args:
        G: NetworkX graph with string node names
        scomplex: dict with 'nodes' mapping node -> list of sample indices
        dataset: pandas DataFrame with samples

    Returns:
        G: Updated graph with node/edge attributes
        scomplex: Unchanged scomplex
    """
    
    # assign edge weight as percentange of shared example between nodes
    for edge in G.edges:
        node_A = scomplex['nodes'][edge[0]]
        node_B = scomplex['nodes'][edge[1]]
        
        G[edge[0]][edge[1]]['weight']  = len(set(node_A).intersection(set(node_B)))


    return G, scomplex

