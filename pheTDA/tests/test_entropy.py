import math

import networkx as nx
import numpy as np
import pytest

from utils.graph_utils import entropy_count


def _two_node_graph(left_members, right_members):
    graph = nx.Graph()
    graph.add_node("left", membership=left_members)
    graph.add_node("right", membership=right_members)
    graph.add_edge("left", "right")
    simplicial_complex = {"nodes": {"left": left_members, "right": right_members}}
    return graph, simplicial_complex


def test_multiclass_entropy_uses_all_category_levels():
    graph, simplicial_complex = _two_node_graph([0, 1, 2], [3, 4, 5])
    phenotype = ["a", "b", "c", "a", "a", "a"]

    graph_entropy, graph = entropy_count(
        simplicial_complex, phenotype, "categorical", G=graph
    )

    assert graph.nodes["left"]["entropy"] == pytest.approx(math.log2(3))
    assert graph.nodes["right"]["entropy"] == pytest.approx(0.0)
    assert graph_entropy == pytest.approx(math.log2(3) / 2)


def test_two_phenotype_entropies_use_distinct_node_attributes():
    graph, simplicial_complex = _two_node_graph([0, 1], [2, 3])

    _, graph = entropy_count(
        simplicial_complex,
        ["a", "a", "b", "b"],
        "categorical",
        G=graph,
        node_attribute="initial_entropy",
    )
    _, graph = entropy_count(
        simplicial_complex,
        [0.0, 0.1, 0.9, 1.0],
        "continuous",
        G=graph,
        node_attribute="final_entropy",
    )

    for node in graph:
        assert "initial_entropy" in graph.nodes[node]
        assert "final_entropy" in graph.nodes[node]


def test_continuous_entropy_is_invariant_to_positive_affine_rescaling():
    graph, simplicial_complex = _two_node_graph([0, 2], [1, 3])
    phenotype = np.array([0.0, 1.0, 10.0, 11.0])

    original, _ = entropy_count(
        simplicial_complex, phenotype, "continuous", G=graph.copy()
    )
    rescaled, _ = entropy_count(
        simplicial_complex, phenotype * 7 - 30, "continuous", G=graph.copy()
    )

    assert original == pytest.approx(rescaled)


@pytest.mark.parametrize(
    ("phenotype", "phenotype_type", "message"),
    [
        ([1, None, 2], "categorical", "missing"),
        (["low", "high"], "continuous", "numeric"),
        ([1, 2], "unsupported", "phenotype type"),
    ],
)
def test_invalid_phenotypes_are_rejected(phenotype, phenotype_type, message):
    graph, simplicial_complex = _two_node_graph([0], [1])

    with pytest.raises(ValueError, match=message):
        entropy_count(simplicial_complex, phenotype, phenotype_type, G=graph)
