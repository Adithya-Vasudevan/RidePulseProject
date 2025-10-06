"""
Small unit test to validate A* returns the expected path on a tiny synthetic graph.
"""
import unittest
import heapq
from typing import Dict, List, Tuple, Callable, Optional


def astar_generic(
    adjacency: Dict[str, List[Tuple[str, float]]],
    src: str,
    dst: str,
    weight: Callable[[str, str, float], float],
    heuristic: Callable[[str], float],
) -> Tuple[List[str], float]:
    """A* algorithm implementation for testing."""
    g = {src: 0.0}
    prev: Dict[str, Optional[str]] = {src: None}
    openq = [(heuristic(src), 0.0, src)]  # (f, g, node)
    visited = set()

    while openq:
        f, curg, u = heapq.heappop(openq)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        for v, base_w in adjacency.get(u, []):
            w = weight(u, v, base_w)
            ng = curg + w
            if ng < g.get(v, float("inf")):
                g[v] = ng
                prev[v] = u
                heapq.heappush(openq, (ng + heuristic(v), ng, v))

    if dst not in g:
        return [], float("inf")

    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path, g[dst]


def dijkstra_generic(
    adjacency: Dict[str, List[Tuple[str, float]]],
    src: str,
    dst: str,
    weight: Callable[[str, str, float], float],
) -> Tuple[List[str], float]:
    """Dijkstra algorithm implementation for testing."""
    dist = {src: 0.0}
    prev: Dict[str, Optional[str]] = {src: None}
    pq = [(0.0, src)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        for v, base_w in adjacency.get(u, []):
            w = weight(u, v, base_w)
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dst not in dist:
        return [], float("inf")

    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path, dist[dst]


class TestRoutingSmall(unittest.TestCase):
    """Test basic A* routing on a simple synthetic graph."""

    def test_astar_simple_path(self):
        """Test A* on a simple 4-node graph."""
        # Create a simple graph: A -> B -> C -> D (linear)
        #   A(0,0) -> B(1,0) -> C(2,0) -> D(3,0)
        # All edges have distance 1.0
        adjacency = {
            "A": [("B", 1.0)],
            "B": [("A", 1.0), ("C", 1.0)],
            "C": [("B", 1.0), ("D", 1.0)],
            "D": [("C", 1.0)]
        }
        
        # Simple weight function that just returns the base distance
        def weight(u: str, v: str, base_km: float) -> float:
            return base_km
        
        # Coordinates for heuristic
        coords = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0)}
        
        # Heuristic function (straight-line distance to destination)
        def heuristic(u: str) -> float:
            x1, y1 = coords[u]
            x2, y2 = coords["D"]  # Target is D
            return abs(x2 - x1) + abs(y2 - y1)  # Manhattan distance
        
        # Test A* from A to D
        path, cost = astar_generic(adjacency, "A", "D", weight, heuristic)
        
        # Should find path A -> B -> C -> D with cost 3.0
        expected_path = ["A", "B", "C", "D"]
        expected_cost = 3.0
        
        self.assertEqual(path, expected_path, f"Expected path {expected_path}, got {path}")
        self.assertEqual(cost, expected_cost, f"Expected cost {expected_cost}, got {cost}")
    
    def test_astar_no_path(self):
        """Test A* when no path exists."""
        # Create disconnected graph: A -> B, C -> D
        adjacency = {
            "A": [("B", 1.0)],
            "B": [("A", 1.0)],
            "C": [("D", 1.0)],
            "D": [("C", 1.0)]
        }
        
        def weight(u: str, v: str, base_km: float) -> float:
            return base_km
        
        def heuristic(u: str) -> float:
            return 0.0  # Simple zero heuristic
        
        # Try to find path from A to C (impossible)
        path, cost = astar_generic(adjacency, "A", "C", weight, heuristic)
        
        # Should return empty path and infinite cost
        self.assertEqual(path, [], "Expected empty path for disconnected graph")
        self.assertEqual(cost, float("inf"), "Expected infinite cost for impossible path")

    def test_dijkstra_simple_path(self):
        """Test Dijkstra on the same simple graph."""
        adjacency = {
            "A": [("B", 1.0)],
            "B": [("A", 1.0), ("C", 1.0)],
            "C": [("B", 1.0), ("D", 1.0)],
            "D": [("C", 1.0)]
        }
        
        def weight(u: str, v: str, base_km: float) -> float:
            return base_km
        
        # Test Dijkstra from A to D
        path, cost = dijkstra_generic(adjacency, "A", "D", weight)
        
        expected_path = ["A", "B", "C", "D"]
        expected_cost = 3.0
        
        self.assertEqual(path, expected_path, f"Expected path {expected_path}, got {path}")
        self.assertEqual(cost, expected_cost, f"Expected cost {expected_cost}, got {cost}")


if __name__ == "__main__":
    unittest.main()