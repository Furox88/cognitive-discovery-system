"""Graph algorithms demo — BFS/DFS, Dijkstra, Kruskal MST, topo sort, cycle detection."""

from cds.graph import (
    Graph,
    bfs,
    dfs,
    dijkstra,
    has_cycle,
    kruskal_mst,
    topological_sort,
)


def main() -> None:
    # Undirected weighted graph (integer nodes).
    g = Graph(n_vertices=6, directed=False)
    edges = [(0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5), (2, 3, 8), (2, 4, 10), (3, 4, 2)]
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)

    print("=== Traversal ===")
    print(f"BFS from 0: {bfs(g, 0)}")
    print(f"DFS from 0: {dfs(g, 0)}")

    print("\n=== Shortest Paths (Dijkstra) ===")
    dists, prev = dijkstra(g, 0)
    for node in sorted(dists):
        print(f"  dist(0 -> {node}) = {dists[node]:.1f}")

    print("\n=== Minimum Spanning Tree (Kruskal) ===")
    mst_edges, total = kruskal_mst(g)
    for e in mst_edges:
        print(f"  {e.src} -- {e.dst}  (w={e.weight})")
    print(f"  MST total weight = {total}")

    print("\n=== Cycle Detection (undirected) ===")
    print(f"has_cycle: {has_cycle(g)}")

    print("\n=== Topological Sort (DAG) ===")
    dag = Graph(n_vertices=3, directed=True)
    dag.add_edge(0, 1, weight=1)
    dag.add_edge(1, 2, weight=1)
    dag.add_edge(0, 2, weight=1)
    order = topological_sort(dag)
    print(f"order: {order}")


if __name__ == "__main__":
    main()
