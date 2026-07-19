"""Tests for graph theory algorithms."""

import pytest

from cds.graph import Graph, bfs, dfs, dijkstra, has_cycle, kruskal_mst, topological_sort


def _make_undirected_graph() -> Graph:
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


def _make_directed_dag() -> Graph:
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

    def test_disconnected_graph_partial_mst(self) -> None:
        # Two connected components with no edge between them: the sorted-edge
        # loop exhausts without ever reaching len(mst) == n_vertices - 1, so
        # the `break` is skipped and the partial (per-component) forest returns.
        g = Graph(n_vertices=4, directed=False)
        g.add_edge(0, 1, 1.0)  # component {0,1}
        g.add_edge(2, 3, 2.0)  # component {2,3}
        edges, total = kruskal_mst(g)
        # A spanning forest of 2 components → 4 - 2 = 2 edges.
        assert len(edges) == 2
        assert total == 3.0


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


class TestDegreeComponents:
    def test_degree(self) -> None:
        from cds.graph import degree

        g = _make_undirected_graph()
        assert degree(g, 0) == 2
        with pytest.raises(ValueError):
            degree(g, 99)

    def test_connected_components(self) -> None:
        from cds.graph import Graph, connected_components

        g = Graph(n_vertices=5, directed=False)
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        comps = connected_components(g)
        assert len(comps) == 3  # {0,1}, {2,3}, {4}


class TestBellmanFordFloydPrim:
    def test_bellman_ford(self) -> None:
        from cds.graph import Graph, bellman_ford

        g = Graph(n_vertices=3, directed=True)
        g.add_edge(0, 1, 2.0)
        g.add_edge(1, 2, -1.0)
        g.add_edge(0, 2, 4.0)
        dist, _ = bellman_ford(g, 0)
        assert dist[2] == 1.0

    def test_bellman_ford_negative_cycle(self) -> None:
        from cds.graph import Graph, bellman_ford

        g = Graph(n_vertices=2, directed=True)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 0, -2.0)
        with pytest.raises(ValueError, match="negative cycle"):
            bellman_ford(g, 0)

    def test_floyd_warshall(self) -> None:
        import math

        from cds.graph import floyd_warshall

        g = _make_undirected_graph()
        d = floyd_warshall(g)
        assert d[0][0] == 0.0
        assert d[0][1] == 1.0
        assert d[0][2] == 3.0
        assert math.isinf(floyd_warshall(Graph(n_vertices=2, directed=True))[0][1])

    def test_prim_matches_kruskal_weight(self) -> None:
        from cds.graph import kruskal_mst, prim_mst

        g = _make_undirected_graph()
        _, w_k = kruskal_mst(g)
        _, w_p = prim_mst(g, start=0)
        assert abs(w_k - w_p) < 1e-12

    def test_prim_directed_error(self) -> None:
        from cds.graph import Graph, prim_mst

        g = Graph(n_vertices=2, directed=True)
        g.add_edge(0, 1)
        with pytest.raises(ValueError):
            prim_mst(g)


def test_bellman_ford_undirected_and_errors() -> None:
    from cds.graph import bellman_ford

    g = _make_undirected_graph()
    dist, _ = bellman_ford(g, 0)
    assert dist[0] == 0.0
    assert dist[1] == 1.0
    with pytest.raises(ValueError):
        bellman_ford(g, -1)


def test_bellman_ford_neg_cycle_undirected() -> None:
    from cds.graph import Graph, bellman_ford

    g = Graph(n_vertices=2, directed=False)
    g.add_edge(0, 1, -1.0)
    # self-loop style cycle via two edges? single undirected negative edge isn't a cycle alone
    # create triangle negative cycle
    g2 = Graph(n_vertices=3, directed=True)
    g2.add_edge(0, 1, 1.0)
    g2.add_edge(1, 2, 1.0)
    g2.add_edge(2, 0, -3.0)
    with pytest.raises(ValueError, match="negative cycle"):
        bellman_ford(g2, 0)


def test_floyd_negative_cycle() -> None:
    from cds.graph import Graph, floyd_warshall

    g = Graph(n_vertices=2, directed=True)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 0, -2.0)
    with pytest.raises(ValueError, match="negative cycle"):
        floyd_warshall(g)


def test_prim_empty_and_bad_start() -> None:
    from cds.graph import Graph, prim_mst

    g0 = Graph(n_vertices=0, directed=False)
    edges, w = prim_mst(g0)
    assert edges == [] and w == 0.0
    g = _make_undirected_graph()
    with pytest.raises(ValueError):
        prim_mst(g, start=99)


def test_connected_directed_weak() -> None:
    from cds.graph import Graph, connected_components

    g = Graph(n_vertices=3, directed=True)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    comps = connected_components(g)
    assert len(comps) == 1


def test_bellman_ford_undirected_branches() -> None:
    from cds.graph import Graph, bellman_ford

    # Disconnected undirected: edge among unreached vertices
    g = Graph(n_vertices=4, directed=False)
    g.add_edge(0, 1, 1.0)
    g.add_edge(2, 3, 1.0)
    dist, _ = bellman_ford(g, 0)
    assert 0 in dist and 1 in dist
    assert 2 not in dist

    # Negative cycle undirected
    g2 = Graph(n_vertices=2, directed=False)
    g2.add_edge(0, 1, -1.0)
    with pytest.raises(ValueError, match="negative cycle"):
        bellman_ford(g2, 0)


def test_floyd_parallel_heavier_edge() -> None:
    from cds.graph import Graph, floyd_warshall

    g = Graph(n_vertices=2, directed=True)
    g.add_edge(0, 1, 1.0)
    g.add_edge(0, 1, 5.0)  # heavier parallel — ignored
    d = floyd_warshall(g)
    assert d[0][1] == 1.0


def test_dijkstra_duplicate_heap_paths() -> None:
    from cds.graph import Graph, dijkstra

    g = Graph(n_vertices=3, directed=True)
    g.add_edge(0, 1, 1.0)
    g.add_edge(0, 2, 5.0)
    g.add_edge(1, 2, 1.0)
    g.add_edge(0, 2, 10.0)  # worse path still may sit in heap
    dist, _ = dijkstra(g, 0)
    assert dist[2] == 2.0


def test_union_rank_branches_via_kruskal() -> None:
    from cds.graph import Graph, kruskal_mst

    # denser graph to exercise union-by-rank paths
    g = Graph(n_vertices=6, directed=False)
    for u, v, w in [
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (4, 5, 1),
        (0, 2, 2),
        (1, 3, 2),
        (2, 4, 2),
        (0, 5, 10),
    ]:
        g.add_edge(u, v, float(w))
    edges, total = kruskal_mst(g)
    assert len(edges) == 5
    assert total > 0


def test_union_find_rank_paths() -> None:
    from cds.graph.algorithms import _find, _union

    parent = {0: 0, 1: 1, 2: 2}
    rank = {0: 0, 1: 0, 2: 0}
    assert _union(parent, rank, 0, 1) is True
    assert rank[0] == 1 or rank[1] == 1
    assert _union(parent, rank, 0, 1) is False  # already same
    # attach lower rank under higher
    parent2 = {0: 0, 1: 1}
    rank2 = {0: 0, 1: 2}
    assert _union(parent2, rank2, 0, 1) is True
    assert _find(parent2, 0) == _find(parent2, 1)


def test_kruskal_complete_tree_break() -> None:
    from cds.graph import Graph, kruskal_mst

    g = Graph(n_vertices=3, directed=False)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 1.0)
    g.add_edge(0, 2, 5.0)  # extra edge not needed after tree complete
    edges, total = kruskal_mst(g)
    assert len(edges) == 2
    assert total == 2.0
