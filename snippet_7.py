from cds.graph import Graph, dijkstra, kruskal_mst, bfs

g = Graph(n_vertices=4, directed=False)
g.add_edge(0, 1, 1.0)
g.add_edge(1, 2, 2.0)
g.add_edge(2, 3, 3.0)
g.add_edge(0, 3, 10.0)

dist, prev = dijkstra(g, 0)
print(dist)  # {0: 0.0, 1: 1.0, 2: 3.0, 3: 6.0}

edges, total = kruskal_mst(g)
print(f"MST weight: {total}")  # 6.0
