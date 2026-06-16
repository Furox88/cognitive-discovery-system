"""Tests for graph theory algorithms."""
from typing import Any

import pytest

from cds.graph import Graph, bfs, dfs, dijkstra, has_cycle, kruskal_mst, topological_sort


def _make_undirected_graph() -> Any:
    """Simple undirected graph:
    0 --1-- 1
    |       |
    4       2
    |       |
    3 --3-- 2
    """
    g = Graph(n_vertices=4, directed=False)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 2.0)
    g.add_edge(2, 3, 3.0)
    g.add_edge(0, 3, 4.0)
    return g


def _make_directed_dag() -> Any:
    """DAG: 0->1, 0->2, 1->3, 2->3"""
    g = Graph(n_vertices=4, directed=True)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    return g


class TestBFS:
    def test_bfs_visits_all_nodes(self) -> None:
        g = _make_undirected_graph()
        order = bfs(g, 0)
        assert set(order) == {0, 1, 2, 3}

    def test_bfs_starts_from_source(self) -> None:
        g = _make_undirected_graph()
        order = bfs(g, 0)
        assert order[0] == 0

    def test_bfs_level_order(self) -> None:
        g = _make_undirected_graph()
        order = bfs(g, 0)
        assert order.index(0) < order.index(1)
        assert order.index(0) < order.index(3)


class TestDFS:
    def test_dfs_visits_all_nodes(self) -> None:
        g = _make_undirected_graph()
        order = dfs(g, 0)
        assert set(order) == {0, 1, 2, 3}

    def test_dfs_starts_from_source(self) -> None:
        g = _make_undirected_graph()
        order = dfs(g, 0)
        assert order[0] == 0

    def test_dfs_on_dag(self) -> None:
        g = _make_directed_dag()
        order = dfs(g, 0)
        assert set(order) == {0, 1, 2, 3}


class TestDijkstra:
    def test_shortest_paths(self) -> None:
        g = _make_undirected_graph()
        dist, prev = dijkstra(g, 0)
        assert dist[0] == 0.0
        assert dist[1] == 1.0
        assert dist[2] == 3.0  # 0->1->2
        assert dist[3] == 4.0  # 0->3

    def test_unreachable_node(self) -> None:
        g = Graph(n_vertices=3, directed=True)
        g.add_edge(0, 1)
        dist, _ = dijkstra(g, 0)
        assert 2 not in dist

    def test_negative_weight_raises(self) -> None:
        g = Graph(n_vertices=2, directed=True)
        g.add_edge(0, 1, -1.0)
        with pytest.raises(ValueError, match="negative"):
            dijkstra(g, 0)

    def test_single_node(self) -> None:
        g = Graph(n_vertices=1)
        dist, prev = dijkstra(g, 0)
        assert dist[0] == 0.0


class TestKruskalMST:
    def test_mst_weight(self) -> None:
        g = _make_undirected_graph()
        edges, total = kruskal_mst(g)
        assert total == 6.0  # 1 + 2 + 3

    def test_mst_edge_count(self) -> None:
        g = _make_undirected_graph()
        edges, _ = kruskal_mst(g)
        assert len(edges) == 3  # V-1 edges

    def test_mst_larger_graph(self) -> None:
        g = Graph(n_vertices=5, directed=False)
        g.add_edge(0, 1, 2)
        g.add_edge(0, 3, 6)
        g.add_edge(1, 2, 3)
        g.add_edge(1, 3, 8)
        g.add_edge(1, 4, 5)
        g.add_edge(2, 4, 7)
        g.add_edge(3, 4, 9)
        edges, total = kruskal_mst(g)
        assert total == 16.0  # 2+3+5+6


class TestTopologicalSort:
    def test_topological_order(self) -> None:
        g = _make_directed_dag()
        order = topological_sort(g)
        assert len(order) == 4
        assert order.index(0) < order.index(1)
        assert order.index(0) < order.index(2)
        assert order.index(1) < order.index(3)
        assert order.index(2) < order.index(3)

    def test_cycle_raises(self) -> None:
        g = Graph(n_vertices=3, directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(g)


class TestHasCycle:
    def test_dag_has_no_cycle(self) -> None:
        g = _make_directed_dag()
        assert has_cycle(g) is False

    def test_cycle_detected(self) -> None:
        g = Graph(n_vertices=3, directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        assert has_cycle(g) is True

    def test_self_loop(self) -> None:
        g = Graph(n_vertices=2, directed=True)
        g.add_edge(0, 0)
        assert has_cycle(g) is True
