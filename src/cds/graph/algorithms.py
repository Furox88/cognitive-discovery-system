"""Graph algorithms — standard implementations from Cormen et al. (CLRS).

References:
    - Cormen, Leiserson, Rivest, Stein. Introduction to Algorithms (4th ed.)
    - Dijkstra, E.W. (1959). A note on two problems in connexion with graphs.
    - Kruskal, J.B. (1956). On the shortest spanning subtree of a graph.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Edge:
    """Weighted edge in a graph."""

    src: int
    dst: int
    weight: float = 1.0


@dataclass
class Graph:
    """Adjacency-list graph representation.

    Supports both directed and undirected graphs with weighted edges.
    """

    n_vertices: int
    directed: bool = False
    adj: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """Add an edge from u to v (and v to u if undirected)."""
        self.adj.setdefault(u, []).append((v, weight))
        if not self.directed:
            self.adj.setdefault(v, []).append((u, weight))
        self.edges.append(Edge(u, v, weight))


def bfs(graph: Graph, start: int) -> list[int]:
    """Breadth-first search traversal.

    Returns vertices in BFS order starting from `start`.
    Time complexity: O(V + E)  [CLRS §22.2]
    """
    visited: set[int] = set()
    order: list[int] = []
    queue: deque[int] = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor, _ in graph.adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order


def dfs(graph: Graph, start: int) -> list[int]:
    """Depth-first search traversal (iterative).

    Returns vertices in DFS order starting from `start`.
    Time complexity: O(V + E)  [CLRS §22.3]
    """
    visited: set[int] = set()
    order: list[int] = []
    stack: list[int] = [start]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor, _ in reversed(graph.adj.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)
    return order


def dijkstra(
    graph: Graph,
    start: int,
) -> tuple[dict[int, float], dict[int, int | None]]:
    """Dijkstra's shortest path algorithm.

    Returns (distances, predecessors) from `start` to all reachable vertices.
    Time complexity: O((V + E) log V) with binary heap  [Dijkstra 1959]

    Args:
        graph: weighted graph (non-negative weights)
        start: source vertex

    Returns:
        distances: dict mapping vertex -> shortest distance from start
        predecessors: dict mapping vertex -> previous vertex on shortest path

    Raises:
        ValueError: if a negative weight is encountered
    """
    dist: dict[int, float] = {start: 0.0}
    prev: dict[int, int | None] = {start: None}
    heap: list[tuple[float, int]] = [(0.0, start)]
    visited: set[int] = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        for v, w in graph.adj.get(u, []):
            if w < 0:
                raise ValueError("negative edge weights not supported")
            new_dist = d + w
            if v not in dist or new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return dist, prev


def _find(parent: dict[int, int], x: int) -> int:
    """Union-Find: find with path compression."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(
    parent: dict[int, int],
    rank: dict[int, int],
    a: int,
    b: int,
) -> bool:
    """Union-Find: union by rank. Returns True if merged."""
    ra, rb = _find(parent, a), _find(parent, b)
    if ra == rb:
        return False
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1
    return True


def kruskal_mst(graph: Graph) -> tuple[list[Edge], float]:
    """Kruskal's minimum spanning tree algorithm.

    Time complexity: O(E log E)  [Kruskal 1956]

    Args:
        graph: undirected weighted graph

    Returns:
        mst_edges: list of edges in the MST
        total_weight: sum of edge weights in the MST
    """
    sorted_edges = sorted(graph.edges, key=lambda e: e.weight)
    parent = {i: i for i in range(graph.n_vertices)}
    rank = {i: 0 for i in range(graph.n_vertices)}

    mst: list[Edge] = []
    total = 0.0

    for edge in sorted_edges:
        if _union(parent, rank, edge.src, edge.dst):
            mst.append(edge)
            total += edge.weight
            if len(mst) == graph.n_vertices - 1:
                break

    return mst, total


def topological_sort(graph: Graph) -> list[int]:
    """Kahn's algorithm for topological sort of a DAG.

    Time complexity: O(V + E)  [CLRS §22.4]

    Args:
        graph: directed acyclic graph

    Returns:
        list of vertices in topological order

    Raises:
        ValueError: if graph contains a cycle
    """
    in_degree: dict[int, int] = {i: 0 for i in range(graph.n_vertices)}
    for u in graph.adj:
        for v, _ in graph.adj[u]:
            in_degree[v] = in_degree.get(v, 0) + 1

    queue: deque[int] = deque(v for v in range(graph.n_vertices) if in_degree.get(v, 0) == 0)
    order: list[int] = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v, _ in graph.adj.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != graph.n_vertices:
        raise ValueError("graph contains a cycle")
    return order


def has_cycle(graph: Graph) -> bool:
    """Detect if a directed graph has a cycle using DFS coloring.

    Time complexity: O(V + E)

    WHITE=0 (unvisited), GRAY=1 (in current DFS path), BLACK=2 (finished).
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[int, int] = {i: WHITE for i in range(graph.n_vertices)}

    def _dfs(u: int) -> bool:
        color[u] = GRAY
        for v, _ in graph.adj.get(u, []):
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and _dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(_dfs(v) for v in range(graph.n_vertices) if color[v] == WHITE)
