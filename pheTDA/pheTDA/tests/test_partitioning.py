import importlib

import networkx as nx

partitioning_module = importlib.import_module("pipeline_objects.Partitioning")


def _mapper_graph():
    graph = nx.Graph()
    graph.add_node("a", membership=[0, 1])
    graph.add_node("b", membership=[0, 2])
    graph.add_edge("a", "b", weight=1)
    simplicial_complex = {"nodes": {"a": [0, 1], "b": [0, 2]}}
    return graph, simplicial_complex


def test_partitioning_uses_only_louvain_and_passes_resolution(monkeypatch):
    calls = []

    def fake_louvain(graph, weight, resolution, seed):
        calls.append((graph, weight, resolution, seed))
        return [{"a"}, {"b"}]

    monkeypatch.setattr(partitioning_module, "louvain_communities", fake_louvain)
    graph, simplicial_complex = _mapper_graph()

    partition = partitioning_module.Partitioning(
        {
            "louvain_resolution": 0.5,
            "ties_resolving_strategy": "new_community_for_ties",
        },
        graph,
        simplicial_complex,
        seed=17,
        sample_ids=[0, 1, 2],
    )

    assert calls == [(graph, "weight", 0.5, 17)]
    assert not hasattr(partition, "ensemble_partitions")


def test_tie_resolution_is_applied_after_louvain(monkeypatch):
    monkeypatch.setattr(
        partitioning_module,
        "louvain_communities",
        lambda *args, **kwargs: [{"a"}, {"b"}],
    )
    graph, simplicial_complex = _mapper_graph()
    partition = partitioning_module.Partitioning(
        {
            "louvain_resolution": 1.0,
            "ties_resolving_strategy": "new_community_for_ties",
        },
        graph,
        simplicial_complex,
        seed=2,
        sample_ids=[0, 1, 2],
    )

    communities = partition.introduce_stratification()["communities"].tolist()

    assert communities == [3, 1, 2]


def test_networkx_louvain_covers_every_graph_node():
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 3.0), ("b", "c", 0.1), ("c", "d", 3.0)])
    simplicial_complex = {
        "nodes": {node: [index] for index, node in enumerate(graph.nodes)}
    }

    partition = partitioning_module.Partitioning(
        {
            "louvain_resolution": 1.0,
            "ties_resolving_strategy": "new_community_for_ties",
        },
        graph,
        simplicial_complex,
        seed=11,
        sample_ids=list(range(4)),
    )

    assigned_nodes = {node for community in partition.communities for node in community}
    assert assigned_nodes == set(graph.nodes)
