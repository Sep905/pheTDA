import argparse
import math
import pandas as pd
import numpy as np
import gower as gw
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances,pairwise_distances
from kmapper.plotlyviz import *
import kmapper as km
import torch
import umap as umap
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import DBSCAN, AgglomerativeClustering, spectral_clustering
from sklearn_extra.cluster import KMedoids
import networkx as nx
import optuna
import pickle
import os


############# PREPROCESSING
def preprocessing_dataset(dataset,
                  sample_id,
                  numerical_features, 
                  binary_features, 
                  categorical_features, 
                  encode, 
                  scale):
    """
    Function that makes dummies variables from categorical variables and scale the numerical variable in the range [0,1].
    """
    
    if sample_id==None:
        dataset['Sample_ID'] = np.linspace(1,dataset.shape[0],dataset.shape[0],dtype=int)
        
    dataset_copy = dataset.copy(True)

    # encode categorical with one-hot, while binary variables are keeped as is
    if encode:

        if (len(categorical_features)!=0) and (len(binary_features)==0):

            df_categorical_dummies = pd.get_dummies(dataset_copy[categorical_features].astype(str),drop_first=False)
            df_others = dataset_copy.loc[:,~dataset_copy.columns.isin(categorical_features)]

            dataset_copy = pd.concat([df_others,df_categorical_dummies],axis=1)

        elif (len(categorical_features)==0) and (len(binary_features)!=0):

            df_binary_dummies = pd.get_dummies(dataset_copy[binary_features].astype(str),drop_first=True)
            df_others = dataset_copy.loc[:,~dataset_copy.columns.isin(binary_features)]

            dataset_copy = pd.concat([df_others,df_binary_dummies],axis=1)

        elif (len(categorical_features)!=0) and (len(binary_features)!=0):
            
            df_categorical_dummies = pd.get_dummies(dataset_copy[categorical_features].astype(str),drop_first=False)
            df_binary_dummies = pd.get_dummies(dataset_copy[binary_features].astype(str),drop_first=True)
            df_others = dataset_copy.loc[:,~dataset_copy.columns.isin(categorical_features + binary_features)]

            dataset_copy = pd.concat([df_others,df_categorical_dummies,df_binary_dummies],axis=1)

    # scale numerical
    if scale:
        scaler = preprocessing.StandardScaler()
        df_numerical = dataset_copy.loc[:, dataset_copy.columns.isin(numerical_features)]
        df_others = dataset_copy.loc[:,~dataset_copy.columns.isin(numerical_features)]
        X_numerical_scaled = scaler.fit_transform(df_numerical)
        df_train_numerical_scaled = pd.DataFrame(X_numerical_scaled,columns=df_numerical.columns)
        dataset_copy = pd.concat([df_others,df_train_numerical_scaled],axis=1)

    return dataset_copy, dataset_copy['Sample_ID']


# function to compute the distance matrix according to the features' type
def compute_distance_matrix(dataset,
                            numerical_features, 
                            binary_features,
                            categorical_features, 
                            distance_matrix_path):
    """
    Function for distance matrix calculation, according to the features' type
    INPUT:
        - dataset               (pandas DataFrame) dataset for which calculate the distance
        - numerical_features     (list) list containing the dataset's numerical features
        - categorical_features   (list) list containing the dataset's categorical features 
        ## 
        Note that the variables not in numerical features + categorical features are considered binary!
        This has implication when variable encoding
        ##
        - visualize             (boolean) flag to specify that the distance matrix will be plotted with a heatmap
        - distance_matrix_path  (string)  path in which the distance matrix will be saved as .npy
    OUTPUT:
        - distance_matrix       (numpy ndarray) patients distance matrix
    """
    
    # find all the categorical variables
    bool_categorical = []
    for features in list(dataset.columns):
        if features in numerical_features:
            bool_categorical.append(False)
        else:
            bool_categorical.append(True)
    
    # only categorical variables -> Jaccard distance
    if sum(bool_categorical)==len(bool_categorical):
        print("only categorical variables -> Jaccard distance")

        # encode clinical variables
        dataset_preprocessed = preprocessing_dataset(dataset,  None,
                                numerical_features, 
                                binary_features, 
                                categorical_features, 
                                True, False)

        distance_matrix = pairwise_distances(dataset_preprocessed.values, metric = "jaccard")
    
    # both categorical and continue variables -> Gower distance
    elif sum(bool_categorical)>0 and sum(bool_categorical)!=len(bool_categorical):
        print("both categorical and continue variables -> Gower distance")
        
        # the gower package already creates dummy variables and standardize numerical variables
        distance_matrix = gw.gower_matrix(dataset,cat_features=bool_categorical)
        dataset['Sample_ID'] = np.linspace(1,dataset.shape[0],dataset.shape[0],dtype=int)
        dataset_preprocessed = dataset
        
    # only numerical variables -> euclidean distance 
    elif sum(bool_categorical)==0:
        print("only numerical variables -> euclidean distance")

        # scale variables
        dataset_preprocessed = preprocessing_dataset(dataset,  None,
                                numerical_features, 
                                binary_features, 
                                categorical_features, 
                                False, True)

        distance_matrix = euclidean_distances(dataset_preprocessed,dataset_preprocessed)
     
    if distance_matrix_path!=None:
        # save the distance matrix as .npy file
        np.save(distance_matrix_path,distance_matrix,allow_pickle=False)

    return distance_matrix, dataset_preprocessed



##### AutoEncoder class and its training function 
class AutoEncoder(torch.nn.Module):
    def __init__(self, n_deep, n_neurons_start, input_dimension, projection_dimension):
        super().__init__()
        
        self.loss_function = torch.nn.MSELoss(reduction="sum")
        
        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()
        
        n_neurons_start_decrement = n_neurons_start

        # fill the encoder
        for n in range(n_deep):

            if n == 0:
                self.encoder.append(torch.nn.Linear(input_dimension, n_neurons_start_decrement))
                self.encoder.append(torch.nn.ReLU())
                
            else:
                self.encoder.append(torch.nn.Linear(n_neurons_start_decrement, int(n_neurons_start_decrement/2)))
                self.encoder.append(torch.nn.ReLU())

                n_neurons_start_decrement = int(n_neurons_start_decrement/2)
            
        self.encoder.append(torch.nn.Linear(n_neurons_start_decrement, projection_dimension))

        # fill the decoder
        reverse_n_neurons_start = int(n_neurons_start/(2**(n_deep-1)))
        
        self.decoder.append(torch.nn.Linear(projection_dimension, reverse_n_neurons_start))
        self.decoder.append(torch.nn.ReLU())
        
        for n in range(n_deep):
            
            if n == n_deep - 1:
                self.decoder.append(torch.nn.Linear(reverse_n_neurons_start, input_dimension))
                
            else:
                self.decoder.append(torch.nn.Linear(reverse_n_neurons_start, int(reverse_n_neurons_start*2)))
                self.decoder.append(torch.nn.ReLU())

                if n != n_deep-1:
                    reverse_n_neurons_start = int(reverse_n_neurons_start*2)
                                    
                                    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
                                    
        return encoded, decoded
               
        
def AE_training(model, data, epochs, learning_rate, w_decay, tolerance, random_seed):
    
    torch.manual_seed(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = (torch.Tensor(data)).to(device)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = w_decay)
    
    model.train()
    train_losses = []

    # arguments for the early stopping
    train_min_loss = 0
    patience = 50
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        encoded, decoded = model(data)
        loss = model.loss_function(data,decoded)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        if epoch == 0:
            train_min_loss = loss.item()
        else:
            if (loss.item()-tolerance)<train_min_loss:
                train_min_loss= loss.item() 
                patience = 50
            else:
                patience-=1
                
            if patience == 0:
                break
                
    return train_losses 


def set_node_community(G, communities):
    """
    Function that assing to each node in the networkx graph the community as attribute
    INPUT:
    - G:           (networkx graph)  networkx graph obtained from the Mapper simplicial complex
    - communities: (list of int)    list of integers, containing the community assigned to each node in the graph G
    """
    for c, nodes_community_c in enumerate(communities):
        for node_c in nodes_community_c:
            G.nodes[node_c]['community'] = c + 1

            
def set_edge_community(G):
    """
    Function which searches for edges within the community and adds them.
    INPUT:
    - G:           (networkx graph)  networkx graph obtained from the Mapper simplicial complex        
    """
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge marked with the community (number)
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge marked with a 0
            G.edges[v, w]['community'] = 0

def associate_sample_to_communities(G,scomplex,communities,dataset_ids, strategy):

    # compute the laplacian centrality for each node if this is the strategy
    if strategy == "laplacian centrality":
        dict_strategy = {}
        for v, c in nx.centrality.laplacian_centrality(G,weight="weight").items():
            dict_strategy[v] = c

    # compute the pagerank centrality for each node if this is the strategy
    if strategy == "pagerank centrality":
        dict_strategy = nx.pagerank(G,weight="weight")

    # loop over patiets ids and graph nodes, to find all the nodes in which the patients appears
    assigned_communities = []
    for j in dataset_ids:
        patient_nodes = {}
        for key,values in scomplex["nodes"].items():
            if j in values:
                patient_nodes[key] = G.nodes[key]['community']

        # compute lists that contain:
        # all the communities in which a patient appears 
        candidate_communities = list(pd.Series(patient_nodes.values(),dtype=int).value_counts().index )     
        # and its frequency (how many nodes FROM A SPECIFIC COMMUNITY contain that patient? )
        canditate_frequencies = list(pd.Series(patient_nodes.values(),dtype=int).value_counts().values) 
        # note that this list are ordered according to the frequencies

        # case 1: the patient appears only in one node -> directly assing the community
        if len(candidate_communities)==1:
            assigned_communities.append(candidate_communities[0])
            
        # case 2: the patient appears in more than one node
        elif len(candidate_communities)>1:

            # if there is a community of majority (there is a community in which the patient appear frequently )
            if canditate_frequencies[0]>canditate_frequencies[1]:
                assigned_communities.append(candidate_communities[0])

            # otherwise resolve ties according to the user defined strategy
            else:

                max_frequency = max(canditate_frequencies)
                index_max = [i for i, j in enumerate(canditate_frequencies) if j == max_frequency]

                list_community_number_of_patients = []
                        
                for index in index_max:
                    total_n_patients_in_all_nodes_for_specific_community = 0
                    for node,community in patient_nodes.items():
                        if community == candidate_communities[index]:

                            # assign according to the node size
                            if strategy == "node size":
                                total_n_patients_in_all_nodes_for_specific_community += len(scomplex['nodes'][node])
                            elif strategy == "node degree":
                                total_n_patients_in_all_nodes_for_specific_community += G.degree([node],weight="weight")[node]
                            elif strategy == "laplacian centrality" or strategy == "pagerank centrality":
                                total_n_patients_in_all_nodes_for_specific_community += dict_strategy[node]

                    list_community_number_of_patients.append(total_n_patients_in_all_nodes_for_specific_community)

                assigned_communities.append(candidate_communities[list_community_number_of_patients.index(max(list_community_number_of_patients))])  
               
        # a patient  not assigned to any community is marked with the community 0
        else:
            assigned_communities.append(0)

        
    new_dataset_ids = pd.DataFrame(dataset_ids).copy(deep=True)
    new_dataset_ids['communities'] = assigned_communities    

    return new_dataset_ids

################# TDA pipeline as optuna objective
class Objective:
    
    def __init__(self, distance_matrix, lens,  cluster_method,  random_seed, sample_ids, results_path, strategy):
        self.mapper = km.KeplerMapper()
        self.distance_matrix = distance_matrix
        self.lens = lens
        self.cluster_method = cluster_method
        self.random_seed = random_seed
        self.sample_ids = sample_ids
        self.path_run = str(self.lens) + "_" + str(self.cluster_method) + "_" + str(self.random_seed)

        # make 3 folder to contain all the trials output: 
        # 1) the lens model fitted;
        # 2) the networkx graph created, from the KeplerMapper scomplex;
        # 3) the dataframe with two columns -> samples ID and the assigned communities
        os.mkdir(results_path + "/" + self.path_run)
        os.mkdir(results_path + "/" + self.path_run + "/model")
        os.mkdir(results_path + "/" + self.path_run + "/scomplex")
        os.mkdir(results_path + "/" + self.path_run + "/communities")
        self.results_path_model = results_path + "/" + self.path_run + "/model"
        self.results_path_scomplex = results_path + "/"  + self.path_run + "/scomplex"
        self.results_path_communities = results_path + "/"  + self.path_run + "/communities"
        self.strategy = strategy

    def __call__(self, trial):

        # according to the specified name of the lens, create the function and project the data
        if self.lens=="PCA":
            lens_model = PCA(n_components=2, random_state=self.random_seed)
            projection = lens_model.fit_transform(self.distance_matrix)

        elif self.lens=="MDS":
            n_it = trial.suggest_int("n_iterations", 3, 5, step=1)
            max_it = trial.suggest_int("max_iterations", 100, 500, step=50)
            lens_model = MDS(n_components=2, random_state=self.random_seed,n_init=n_it,max_iter=max_it, metric=True,dissimilarity="precomputed")
            projection = lens_model.fit_transform(self.distance_matrix)

        elif self.lens=="tSNE":
            p = trial.suggest_int("perplexity", 5, 50, step=5)
            lr = trial.suggest_int("learning_rate", 10, 1000, step=50)
            lens_model = TSNE(n_components=2, random_state=self.random_seed, perplexity = p , learning_rate= lr ,init="random",metric="precomputed")
            projection = lens_model.fit_transform(self.distance_matrix)

        elif self.lens=="UMAP":
            n_neig = trial.suggest_int("n_neighbors", 5, int(self.distance_matrix.shape[0]/4), step=10)
            min_dist = trial.suggest_float("min_dist", 0.1, 0.9, step=0.1)
            lens_model = umap.UMAP(n_components=2, random_state=self.random_seed, n_neighbors= n_neig, min_dist=min_dist,metric="precomputed")
            projection = lens_model.fit_transform(self.distance_matrix)

        elif self.lens=="AE":
            number_of_layers = trial.suggest_int("num_layers", 2, 4, step = 1)
            number_of_neurons_start = trial.suggest_int("number_neuron_start", 200, 500, step = 100)
            learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, step = 1e-1)
            weight_decay = trial.suggest_float("wdecay", 1e-6, 1e-3, step = 1e-1)

            lens_model = AutoEncoder(number_of_layers,number_of_neurons_start,self.distance_matrix.shape[1],2)
            training_loss_model = AE_training(lens_model,self.distance_matrix, 4, learning_rate, weight_decay, 1e-2, self.random_seed)
            projection = lens_model.eval()(torch.Tensor(self.distance_matrix).to(device='cuda'))[0].detach().cpu()

        cub = trial.suggest_int("cub", 6, 22, step = 2)
        over = trial.suggest_float("over", 0.2, 0.5, step = 0.1)

        # cluster the data into the intervals according to the clustering method defined
        if self.cluster_method == "DBSCAN":

            min_sample_DBSCAN = trial.suggest_int("min_smaple_DBSCAN", 2, 4, step = 1)
            eps_DBSCAN = trial.suggest_float("eps_DBSCAN", 0.1, 0.5, step = 0.1)

            cluster_method = DBSCAN(metric="precomputed",min_samples=min_sample_DBSCAN,eps=eps_DBSCAN)

        elif self.cluster_method == "agglomerative_average":
            n_clusters = trial.suggest_int("n_clusters", 2, 4, step = 1)

            cluster_method = AgglomerativeClustering(metric="precomputed",n_clusters=n_clusters,linkage="average")

        elif self.cluster_method == "agglomerative_complete":
            n_clusters = trial.suggest_int("n_clusters", 2, 4, step = 1)

            cluster_method = AgglomerativeClustering(metric="precomputed",n_clusters=n_clusters,linkage="complete")

        elif self.cluster_method == "agglomerative_single":
            n_clusters = trial.suggest_int("n_clusters", 2, 4, step = 1)

            cluster_method = AgglomerativeClustering(metric="precomputed",n_clusters=n_clusters,linkage="single")

        elif self.cluster_method == "spectral_clustering":
            n_clusters = trial.suggest_int("n_clusters", 2, 4, step = 1)

            cluster_method = spectral_clustering(metric="precomputed",n_clusters=n_clusters, assign_labels = "cluster_qr")

        elif self.cluster_method == "kmedoids":
            n_clusters = trial.suggest_int("n_clusters", 2, 4, step = 1)

            cluster_method = KMedoids(metric="precomputed",n_clusters=n_clusters, init = "heuristic")


        scomplex = self.mapper.map(projection, self.distance_matrix,  cover=km.Cover(n_cubes=cub, perc_overlap=np.round(over,3)), clusterer=cluster_method,  
                            precomputed=True,remove_duplicate_nodes = True)

        # make the networkx graph and compute the fraction of isolated nodes
        G = km.adapter.to_nx(scomplex)
        fraction_of_isolated_nodes = len(list(nx.isolates(G)) )/len(G.nodes)

        # assign edge weight as percentange of shared example between nodes
        for edge in G.edges:
            node_A = scomplex['nodes'][edge[0]]
            node_B = scomplex['nodes'][edge[1]]

            G[edge[0]][edge[1]]['weight']  = len(set(node_A).intersection(set(node_B)))

        # find communities with louvain algorithm
        resolution_louvain = trial.suggest_float("resolution_louvain", 1e-2, 1e+1, step = 0.1)
        partition = nx.community.louvain_communities(G, weight = 'weight', resolution=resolution_louvain, seed =self.random_seed)
        
        # set the node and the edge attributes to the networkx graph
        set_node_community(G, partition)
        set_edge_community(G)

        # compute the modularity
        modularity = nx.algorithms.community.modularity(G,partition, resolution=resolution_louvain)

        # assign the patients to the communities
        new_dataset_ids_communities = associate_sample_to_communities(G,scomplex,partition,self.sample_ids,self.strategy)

        # evaluate with silhouette score
        silhouette = silhouette_score(X = self.distance_matrix, labels = new_dataset_ids_communities['communities'],metric="precomputed")

        # save the trail outputs: note that each of them is saved with a generated ID
        artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=self.results_path_model)
        # Save the model using ArtifactStore
        with open(self.results_path_model + "./"+ str(self.lens) + "_" + str(self.cluster_method) + "_" + str(self.random_seed) + ".pickle", "wb") as fout:
            pickle.dump(lens_model, fout)
        artifact_id = optuna.artifacts.upload_artifact(
            artifact_store=artifact_store,
            file_path=self.results_path_model +  "/" + str(self.lens) + "_" + str(self.cluster_method) + "_" + str(self.random_seed) + ".pickle",
            study_or_trial=trial.study,
        )
        trial.set_user_attr("artifact_id", artifact_id)
        with open(self.results_path_scomplex + "./" + str(artifact_id) + ".pickle", "wb") as fout:
            pickle.dump(G, fout)
        new_dataset_ids_communities.to_excel(self.results_path_communities + "./" + str(artifact_id) + ".xlsx",index=False)

        return fraction_of_isolated_nodes, modularity, silhouette
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TDA')
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--rn', type=int, default=127)
    parser.add_argument('--lens', type=str, default="")
    parser.add_argument('--clustering_method', type=str, default="")
    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--results_path', type=str, default="")
    parser.add_argument('--ties_strategy', type=str, default="")
    args = parser.parse_args()  

    # read the dataset and specify the type of the variables
    dataset = pd.read_xlsx(args.dataset_path)
    numerical_var = []
    binary_var = []
    categorical_var = []

    # compute distance matrix
    distance_matrix, dataset_preprocessed = compute_distance_matrix(dataset, numerical_var, binary_var, categorical_var, None)

    # create optuna objective
    objective = Objective(distance_matrix, args.lens, args.clustering_method, args.rn, dataset_preprocessed['Sample_ID'], args.results_path, args.ties_strategy)

    # create optuna study with directions for the object to optimize
    study = optuna.create_study(directions=["minimize","maximize", "maximize"], sampler = optuna.samplers.TPESampler(seed = args.rn))

    # make optuna optimization
    study.optimize(objective, n_trials=100, timeout=None)

    # save the dataframe results
    study.trials_dataframe().to_excel(args.results_path + "/" + str(args.lens) + "_" + str(args.clustering_method) + "_" + str(args.rn) + ".xlsx",index=False)
