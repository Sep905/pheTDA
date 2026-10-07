"""Utilities for annotating and stratifying Mapper graphs."""

from collections import Counter

import kmapper as km
import networkx as nx
import numpy as np
import pandas as pd

VALID_CLASS_TYPES = {"categorical", "continuous"}


def _labels_for_measure(values, value_type):
    """Return categorical labels used by the entropy and spread measures.

    Continuous values are binned using one set of Freedman-Diaconis edges
    computed from the complete outcome vector. Using shared edges makes node
    entropies comparable and gives spread the same interpretation as in the
    categorical case.
    """
    if value_type not in VALID_CLASS_TYPES:
        raise ValueError("phenotype type must be either 'categorical' or 'continuous'")

    values = np.asarray(values)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("phenotype cannot be empty")
    if pd.isna(values).any():
        raise ValueError("phenotype cannot contain missing values")

    if value_type == "categorical":
        return values

    try:
        numeric_values = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("continuous phenotype values must be numeric") from exc
    if not np.isfinite(numeric_values).all():
        raise ValueError("continuous phenotype values must be finite")
    if np.all(numeric_values == numeric_values[0]):
        return np.zeros(numeric_values.size, dtype=int)

    bin_edges = np.histogram_bin_edges(numeric_values, bins="fd")
    if bin_edges.size <= 2:
        return np.zeros(numeric_values.size, dtype=int)
    return np.digitize(numeric_values, bin_edges[1:-1], right=False)


def _node_memberships(G):
    memberships = []
    for node in G.nodes:
        members = G.nodes[node].get("membership")
        if members is None:
            raise ValueError(f"Mapper node {node!r} has no 'membership' attribute")
        members = np.asarray(members, dtype=int)
        G.nodes[node]["size"] = int(members.size)
        memberships.append(members)
    return memberships


def _validate_memberships(memberships, n_samples):
    for members in memberships:
        if members.size and (members.min() < 0 or members.max() >= n_samples):
            raise IndexError("a Mapper membership index is outside the phenotype")


def entropy_count(
    scomplex,
    phenotype,
    phenotype_type,
    G=None,
    node_attribute="entropy",
):
    """Compute the membership-weighted mean Shannon entropy of Mapper nodes.

    For a continuous outcome, values are first assigned to global
    Freedman-Diaconis bins. The same bin edges are therefore used for every
    node. The returned graph has per-node (unweighted) entropy values in bits.

    Mapper memberships can overlap. Consequently, the graph-level weighting
    counts each node membership, which is intentional for a node-level metric.
    """
    labels = _labels_for_measure(phenotype, phenotype_type)
    if G is None:
        G = km.adapter.to_nx(scomplex)

    memberships = _node_memberships(G)
    _validate_memberships(memberships, labels.size)

    weighted_entropy = 0.0
    total_memberships = 0
    for node, members in zip(G.nodes, memberships):
        counts = np.asarray(list(Counter(labels[members]).values()), dtype=float)
        if counts.size == 0:
            node_entropy = 0.0
        else:
            probabilities = counts / counts.sum()
            node_entropy = float(-np.sum(probabilities * np.log2(probabilities)))

        G.nodes[node][node_attribute] = node_entropy
        weighted_entropy += members.size * node_entropy
        total_memberships += members.size

    graph_entropy = (
        weighted_entropy / total_memberships if total_memberships else float("nan")
    )
    return float(graph_entropy), G


def spread_measure(G, initial_class, initial_class_type):
    """Measure how widely outcome strata are distributed over a Mapper graph.

    For each categorical label (or continuous-value bin), the measure is the
    expected pairwise shortest-path distance between its Mapper memberships.
    Hop distances are divided by graph diameter, making the result comparable
    across graphs and bounded to ``[0, 1]``. Label-specific values are averaged
    using their frequencies in the original sample.
    """
    labels = _labels_for_measure(initial_class, initial_class_type)
    node_list = list(G.nodes)
    if not node_list:
        return float("nan"), G
    if not nx.is_connected(G):
        raise ValueError("spread_measure requires a connected graph")

    memberships = _node_memberships(G)
    _validate_memberships(memberships, labels.size)

    node_index = {node: index for index, node in enumerate(node_list)}
    distances = np.zeros((len(node_list), len(node_list)), dtype=float)
    for source, lengths in nx.all_pairs_shortest_path_length(G):
        source_index = node_index[source]
        for target, distance in lengths.items():
            distances[source_index, node_index[target]] = distance

    diameter = float(distances.max())
    if diameter > 0:
        distances /= diameter

    # Preserve first-seen ordering so mixed, non-sortable categorical labels
    # are supported as well.
    unique_labels = list(dict.fromkeys(labels.tolist()))
    label_to_row = {label: row for row, label in enumerate(unique_labels)}
    counts_by_node = np.zeros((len(unique_labels), len(node_list)), dtype=float)
    for column, members in enumerate(memberships):
        for label, count in Counter(labels[members]).items():
            counts_by_node[label_to_row[label], column] = count

    label_spread = {}
    for label, row in label_to_row.items():
        counts = counts_by_node[row]
        denominator = counts.sum() ** 2
        label_spread[label] = (
            float(counts @ distances @ counts / denominator)
            if denominator > 0
            else float("nan")
        )

    original_counts = Counter(labels)
    valid_labels = [
        label for label in unique_labels if np.isfinite(label_spread[label])
    ]
    total_weight = sum(original_counts[label] for label in valid_labels)
    graph_spread = (
        sum(original_counts[label] * label_spread[label] for label in valid_labels)
        / total_weight
        if total_weight
        else float("nan")
    )

    for node, members in zip(node_list, memberships):
        node_counts = Counter(labels[members])
        node_total = sum(node_counts.values())
        node_spread = (
            sum(count * label_spread[label] for label, count in node_counts.items())
            / node_total
            if node_total
            else float("nan")
        )
        G.nodes[node]["spread"] = float(node_spread)

    return float(graph_spread), G


def set_node_community(G, communities):
    """Attach all one-based community identifiers to graph nodes."""
    nx.set_node_attributes(G, {node: [] for node in G.nodes}, "communities")
    for community_id, nodes in enumerate(communities, start=1):
        for node in nodes:
            G.nodes[node]["communities"].append(community_id)
    for node in G.nodes:
        memberships = G.nodes[node]["communities"]
        G.nodes[node]["community"] = memberships[0] if memberships else None


def set_edge_community(G):
    """Attach communities shared by both endpoints, or zero if there are none."""
    for source, target in G.edges:
        shared = sorted(
            set(G.nodes[source]["communities"]) & set(G.nodes[target]["communities"])
        )
        G.edges[source, target]["communities"] = shared
        G.edges[source, target]["community"] = shared[0] if shared else 0


def _normalise_scores(scores):
    values = np.asarray(list(scores.values()), dtype=float)
    minimum = values.min()
    span = values.max() - minimum
    if span == 0:
        return {node: 0.0 for node in scores}
    return {node: (value - minimum) / span for node, value in scores.items()}


def associate_sample_to_communities(G, scomplex, communities, dataset_ids, strategy):
    """Resolve overlapping Mapper memberships into one community per sample."""
    valid_strategies = {"new_community_for_ties", "size", "centrality_ensemble"}
    if strategy not in valid_strategies:
        raise ValueError(f"unknown tie resolution strategy: {strategy!r}")

    node_to_communities = {node: [] for node in G.nodes}
    for community_id, nodes in enumerate(communities, start=1):
        for node in nodes:
            node_to_communities[node].append(community_id)
    next_community_id = len(communities) + 1
    tie_map = {}

    centralities = None
    if strategy == "centrality_ensemble":
        centralities = [
            _normalise_scores(dict(G.degree())),
            _normalise_scores(nx.laplacian_centrality(G)),
            _normalise_scores(nx.betweenness_centrality(G)),
            _normalise_scores(nx.pagerank(G)),
        ]

    assigned_communities = []
    for sample_id in dataset_ids:
        sample_nodes = {
            node: node_to_communities[node]
            for node, members in scomplex["nodes"].items()
            if sample_id in members
        }
        if not sample_nodes:
            assigned_communities.append(None)
            continue

        frequencies = Counter(
            community_id
            for node_communities in sample_nodes.values()
            for community_id in node_communities
        )
        maximum_frequency = max(frequencies.values())
        tied_communities = [
            community_id
            for community_id, frequency in frequencies.items()
            if frequency == maximum_frequency
        ]
        if len(tied_communities) == 1:
            assigned_communities.append(tied_communities[0])
            continue

        if strategy == "new_community_for_ties":
            tie_key = frozenset(tied_communities)
            if tie_key not in tie_map:
                tie_map[tie_key] = next_community_id
                next_community_id += 1
            assigned_communities.append(tie_map[tie_key])
            continue

        community_nodes = {
            community_id: [
                node
                for node, node_communities in sample_nodes.items()
                if community_id in node_communities
            ]
            for community_id in tied_communities
        }
        scores = []
        for community_id in tied_communities:
            nodes = community_nodes[community_id]
            if strategy == "size":
                score = sum(len(scomplex["nodes"][node]) for node in nodes)
            else:
                score = sum(
                    np.mean([centrality[node] for centrality in centralities])
                    for node in nodes
                )
            scores.append(score)

        assigned_communities.append(tied_communities[int(np.argmax(scores))])

    return pd.DataFrame(
        {"dataset_id": list(dataset_ids), "communities": assigned_communities}
    )
