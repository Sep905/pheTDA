import networkx as nx
import numpy as np
import pandas as pd

from optuna_pipeline.pipelines import (
    _valid_silhouette_labels,
    graph_creation_checks_fail,
    node_in_communities_check_fail,
)


def test_disconnected_graph_is_accepted_when_all_samples_are_present():
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("c", "d")])
    simplicial_complex = {"nodes": {"a": [0], "b": [1], "c": [2], "d": [3]}}

    assert not nx.is_connected(graph)
    assert not graph_creation_checks_fail(np.zeros((4, 2)), simplicial_complex, graph)


def test_graph_without_edges_is_rejected():
    graph = nx.Graph()
    graph.add_node("a", membership=[0])

    assert graph_creation_checks_fail(np.zeros((1, 2)), {"nodes": {"a": [0]}}, graph)


def test_graph_missing_a_sample_is_rejected():
    graph = nx.Graph()
    graph.add_edge("a", "b")
    simplicial_complex = {"nodes": {"a": [0], "b": [1]}}

    assert graph_creation_checks_fail(np.zeros((3, 2)), simplicial_complex, graph)


def test_community_check_requires_exact_graph_node_coverage():
    graph = nx.path_graph(["a", "b", "c"])
    complete = [["a", "b"], ["c"]]
    incomplete = [["a", "b"]]
    unexpected = [["a", "b"], ["c", "d"]]

    assert not node_in_communities_check_fail(graph, complete)
    assert node_in_communities_check_fail(graph, incomplete)
    assert node_in_communities_check_fail(graph, unexpected)


def test_silhouette_labels_require_between_two_and_n_minus_one_groups():
    assert _valid_silhouette_labels(pd.Series([1, 1, 2, 2]))
    assert not _valid_silhouette_labels(pd.Series([1, 1, 1, 1]))
    assert not _valid_silhouette_labels(pd.Series([1, 2, 3, 4]))
    assert not _valid_silhouette_labels(pd.Series([1, None, 2, 2]))
