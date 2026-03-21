import pandas as pd
import math
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
import statistics
from collections import Counter, defaultdict
from scipy.stats import gaussian_kde
import sys


def entropy_count(scomplex, G, initial_class, initial_class_type):
    """
    Compute graph entropy from a KeplerMapper simplicial complex,
    and attach node-wise weighted entropy as a 'entropy' attribute
    on the corresponding NetworkX graph.

    Parameters
    ----------
    scomplex : dict
        KeplerMapper simplicial complex (output of mapper.map).
    initial_class : pandas.Series
        Phenotype / label for each original data point (index aligned with scomplex nodes).

    Returns
    -------
    graph_entropy : float
        Size-weighted mean of node entropies (as before).
    node_entropy : dict
        Mapping node_name -> size-weighted node entropy.
    G : nx.Graph
        NetworkX graph with node attribute 'entropy' set.
    """
    initial_class_ = initial_class.copy(True)

    # For convenience, get node names consistently with scomplex["nodes"]
    # Here we assume G nodes are named with the same keys as scomplex["nodes"].
    node_names = list(scomplex["nodes"].keys())

    entropy_node = {}

    if initial_class_type == "categorical":

        label_values = list(initial_class_.value_counts().index)
        label_values.sort()

        # Per-node entropy, as in the original function
        for j, node_name in enumerate(node_names):
            # samples in this node
            idx = scomplex["nodes"][node_name]
            data_bin = initial_class_.iloc[idx]

            dimCluster = len(data_bin)

            dimData_dict = {}
            for label in label_values:
                dimData_dict[label] = np.sum(data_bin == label)

            H_node = 0.0
            if dimCluster > 0:
                for _, v in dimData_dict.items():
                    if v != 0:
                        p = v / dimCluster
                        H_node += -(p) * math.log2(p)

            entropy_node[node_name] = (dimCluster, H_node)
            G.nodes[node_name]["entropy"] = H_node
        

    elif initial_class_type == "continuous":
        # Get the actual array of continuous values
        label_values = np.asarray(initial_class_.values)
        
        # Create GLOBAL bins so every node is measured on the exact same scale.
        # 'auto' uses a robust statistical estimator to determine the ideal number of buckets 
        # based on your entire dataset's spread.
        global_bins = np.histogram_bin_edges(label_values, bins='auto')

        for j, node_name in enumerate(node_names):
            idx = scomplex["nodes"][node_name]
            values_in_node = label_values[idx]
            dimCluster = len(values_in_node)

            if dimCluster == 0:
                H_node = 0.0
            else:
                # Count how many node samples fall into each global bucket
                counts, _ = np.histogram(values_in_node, bins=global_bins)
                
                # Filter out empty buckets to avoid taking log2(0)
                non_zero_counts = counts[counts > 0]
                
                # Compute standard discrete Shannon Entropy
                probabilities = non_zero_counts / dimCluster
                H_node = -np.sum(probabilities * np.log2(probabilities))

            entropy_node[node_name] = (dimCluster, H_node)
            G.nodes[node_name]["entropy"] = H_node

    else:
        print("Insert a valid type of initial class [categorical/continuous]")
        sys.exit()

    sumBinEntropies = 0
    numberData = 0
    for node_name, (dimCluster, H_node) in entropy_node.items():
        sumBinEntropies += dimCluster * H_node
        numberData += dimCluster
    graph_entropy = sumBinEntropies / numberData if numberData > 0 else float("nan")

    return graph_entropy, G

def spread_measure( G: nx.Graph, group_ind, group_ind_type):
    """
    Spread measure on a connected Mapper graph already converted to NetworkX.

    Assumes each node in G has an attribute 'membership' containing
    the indices of samples in that node (as used by KeplerMapper adapters).

    Parameters
    ----------
    G : networkx.Graph
    group_ind : array-like, shape (n_samples,)
        Discrete label for each original data point (0-based indexing).

    Returns
    -------
    graph_spread : float
        Global spread measure over the whole graph (weighted average over labels).
    node_spread : dict
        Mapping node_id -> spread value for that node
        (label-frequency–weighted average of label_spread inside the node).
    """
    group_ind = np.asarray(group_ind)

    # ----- nodes, memberships, adjacency -----
    node_list = list(G.nodes())
    n_vertices = len(node_list)
    node_index = {n: i for i, n in enumerate(node_list)}

    points_in_vertex = []
    for n in node_list:
        members = G.nodes[n].get('membership', None)
        #G.nodes[n]['size'] = len(members)
        points_in_vertex.append(np.asarray(members, dtype=int))

    # ----- all-pairs shortest-path distances -----
    dist_dict = dict(nx.all_pairs_dijkstra_path_length(G,weight='weight'))

    D = np.zeros((n_vertices, n_vertices), dtype=float)
    for n_i, lengths in dist_dict.items():
        i = node_index[n_i]
        for n_j, d in lengths.items():
            j = node_index[n_j]
            D[i, j] = float(d)

    if group_ind_type == "categorical":

        # ----- label-by-vertex count matrix -----
        _, first_idx = np.unique(group_ind, return_index=True)
        unique_labels = group_ind[np.sort(first_idx)]
        n_labels = len(unique_labels)
        label_to_row = {lab: i for i, lab in enumerate(unique_labels)}

        groups_vertices = np.zeros((n_labels, n_vertices), dtype=int)
        for j, pts in enumerate(points_in_vertex):
            labs = group_ind[pts]
            counts = Counter(labs)
            for lab, c in counts.items():
                groups_vertices[label_to_row[lab], j] = c


        # ----- mean distance per label (R-style quadratic form) -----
        mean_dist = np.zeros(n_labels, dtype=float)
        for i in range(n_labels):
            v = groups_vertices[i, :].astype(float)
            if v.sum() == 0:
                mean_dist[i] = np.nan
                continue
            denom = np.sum(v)**2
            if denom == 0:
                mean_dist[i] = np.nan
                continue
            numerator = v @ D @ v
            mean_dist[i] = numerator / denom

        # ----- global weighting by class frequencies -----
        label_counts = Counter(group_ind)
        weight = np.array([label_counts[lab] for lab in unique_labels], dtype=float)

        valid = ~np.isnan(mean_dist)
        mean_dist[~valid] = 0.0
        weight[~valid] = 0.0

        if weight.sum() == 0:
            graph_spread = np.nan
        else:
            graph_spread = float(np.sum(weight * mean_dist) / np.sum(weight))

        # per-label spread
        label_spread = {
            lab: (float(md) if not np.isnan(md) else np.nan)
            for lab, md in zip(unique_labels, mean_dist)
        }

        # ----- per-node spread: label-weighted average of label_spread -----
        node_spread: Dict[Any, float] = {}
        for n, pts in zip(node_list, points_in_vertex):
            labs = group_ind[pts]
            if labs.size == 0:
                node_spread[n] = np.nan
                continue
            cnts = Counter(labs)
            total = sum(cnts.values())
            val = 0.0
            wsum = 0.0
            for lab, c in cnts.items():
                ls = label_spread.get(lab, np.nan)
                if np.isnan(ls):
                    continue
                val += c * ls
                wsum += c
            node_spread[n] = float(val / wsum) if wsum > 0 else np.nan
            G.nodes[n]["spread"] = node_spread[n] 

    elif group_ind_type == "continuous":
        
        v_cont = np.zeros(n_vertices, dtype=float)
        node_values = {}  # Store actual values in each node for later
        
        for j, pts in enumerate(points_in_vertex):
            if len(pts) > 0:
                v_cont[j] = np.mean(group_ind[pts])
                node_values[node_list[j]] = group_ind[pts]
            else:
                v_cont[j] = 0.0
                node_values[node_list[j]] = np.array([])
        
        # Compute global spread using quadratic form
        numerator = v_cont @ D @ v_cont
        denom = np.sum(v_cont) ** 2
        
        if denom == 0:
            graph_spread = np.nan
        else:
            graph_spread = float(numerator / denom)
        
        # Per-node continuous spread: weighted average of covariance-based spread
        # For each node, compute how much its values deviate from its neighbors (graph-weighted)
        node_spread_cont: Dict[Any, float] = {}

        # Normalize by global variance
        global_var = np.var(group_ind)
        global_var = max(global_var, 1e-10)
        
        for n_idx, (n, pts) in enumerate(zip(node_list, points_in_vertex)):
            values_in_node = group_ind[pts]
            
            if len(values_in_node) <= 1:
                node_spread_cont[n] = 0.0
                continue
            
            # Compute variance within this node
            var_node = np.var(values_in_node)
            node_spread_cont[n] = float(var_node)
            G.nodes[n]['spread'] = float(node_spread_cont[n]/ global_var)
       
    else:
        print("Insert a valid type of initial class [categorical/continuous]")
        sys.exit()
        
    return graph_spread, G


def estimate_dbscan_params(data, smooth_n=1000):
    """
    Estimate optimal DBSCAN parameters (`eps` and `minPts`) using the elbow method and log rule.
    
    Parameters:
    - datadf: (pandas DataFrame) dataset of interest
    - smooth_n: Number of interpolation points for smoother elbow curve (default: 1000)
    
    Returns:
    - eps_estimated: float, estimated `eps` value
    - minPts: int, estimated `minPts` value
    """

    n_samples = data.shape[0]
    minPts = int(np.round(np.log(n_samples)))

    # Use k+1 because kNN includes the point itself
    nbrs = NearestNeighbors(n_neighbors=minPts + 1).fit(data)
    distances, _ = nbrs.kneighbors(data)

    # Get the k-th nearest distance (excluding self-distance)
    k_distances = np.sort(distances[:, minPts])

    # Interpolate to smooth the curve
    x_vals = np.linspace(0, len(k_distances) - 1, smooth_n)
    interpolator = interp1d(np.arange(len(k_distances)), k_distances, kind='linear')
    y_vals = interpolator(x_vals)

    # First and second derivative for elbow detection
    dy = np.gradient(y_vals, x_vals)
    d2y = np.gradient(dy, x_vals)

    # Find the elbow as the maximum second derivative
    elbow_idx = np.argmax(np.abs(d2y))
    eps_estimated = y_vals[elbow_idx]

    return eps_estimated, minPts

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


def associate_sample_to_communities(
    G, 
    scomplex, 
    communities, 
    dataset_ids, 
    strategy
):
    """
    Fully parallelized version: ALL strategies use precomputed data.
    
    Args:
        G: NetworkX graph with edge weights = similarity
        scomplex: dict with "nodes" mapping node → list of dataset_ids (patients)
        communities: list of sets of node IDs
        dataset_ids: list of patient IDs
        strategy: 'size', 'centrality_ensemble'
    
    Returns:
        new_dataset_ids: DataFrame with 'dataset_id', 'communities'
    """

    assigned_communities = []
    
    # ========== PRECOMPUTE ALL DATA (PARALLELIZATION FOUNDATION) ==========
    
    # Map node -> community id (1..C)
    node_to_comm = {}
    for comm_id, nodes_community_c in enumerate(communities, start=1):
        for node_c in nodes_community_c:
            node_to_comm[node_c] = comm_id
    
    # Community → list of nodes (for ALL strategies)
    comm_to_nodes = defaultdict(list)
    for node, comm_id in node_to_comm.items():
        comm_to_nodes[comm_id].append(node)
    
    # Precompute centralities once (MOVED BEFORE comm_stats)
    deg = dict(G.degree())
    laplacian = nx.laplacian_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    
    # Precompute community statistics (for size/centrality strategies)
    comm_stats = {}
    for comm_id, nodes in comm_to_nodes.items():
        # Size: sum of node sizes
        total_size = sum(len(scomplex['nodes'][node]) for node in nodes)
        
        # Centrality ensemble: average of all 4 metrics
        avg_centrality = statistics.mean([
            statistics.mean(deg[node] for node in nodes),
            statistics.mean(laplacian[node] for node in nodes),
            statistics.mean(betweenness[node] for node in nodes),
            statistics.mean(pagerank[node] for node in nodes)
        ])
        
        comm_stats[comm_id] = {
            'size': total_size,
            'centrality': avg_centrality,
            'nodes': nodes
        }
    
    # Tie group cache (for new_community_for_ties)
    tie_map = {}
    next_new_comm_id = max(node_to_comm.values()) + 1 if node_to_comm else 1
    
    # ========== PROCESS EACH PATIENT (PARALLEL) ==========
    for patient in dataset_ids:
        # Find all nodes containing the patient
        patient_nodes = {
            node: node_to_comm[node]
            for node, values in scomplex["nodes"].items() 
            if patient in values
        }
        
        if not patient_nodes:
            assigned_communities.append(None)
            continue
        
        # Count community frequencies
        freq = Counter(patient_nodes.values())
        freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
        candidate_communities = list(freq.keys())
        candidate_frequencies = list(freq.values())
        
        # CASE 1: only one community (all strategies)
        if len(candidate_communities) == 1:
            assigned_communities.append(candidate_communities[0])
            continue
        
        # CASE 2: clear majority (all strategies)
        if candidate_frequencies[0] > candidate_frequencies[1]:
            assigned_communities.append(candidate_communities[0])
            continue
        
        # ========== TIED CASE: ALL STRATEGIES PARALLELIZED ==========
        
        # Identify tied communities
        max_frequency = max(candidate_frequencies)
        tied_communities = [
            candidate_communities[i] 
            for i, freq_val in enumerate(candidate_frequencies) 
            if freq_val == max_frequency
        ]
        
        # ========== PARALLEL STRATEGIES (ALL USE PRECOMPUTED DATA) ==========
            
        if strategy == "size":
            # Use precomputed community sizes
            best_comm = max(tied_communities, key=lambda c: comm_stats[c]['size'])
            
        elif strategy == "centrality_ensemble":
            # Use precomputed centrality scores
            best_comm = max(
                tied_communities, 
                key=lambda c: comm_stats[c]['centrality']
            )
        
        else:
            # Fallback for "majority" or unspecified: take the first tied community
            best_comm = tied_communities[0]
        
        assigned_communities.append(best_comm)
    
    # Create output DataFrame
    new_dataset_ids = pd.DataFrame({
        'dataset_id': dataset_ids,
        'communities': assigned_communities,

    })
    
    return new_dataset_ids