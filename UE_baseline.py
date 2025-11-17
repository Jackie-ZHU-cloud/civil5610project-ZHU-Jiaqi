
"""
Baseline User-Equilibrium (UE) solver for CIVL 5610 course project
------------------------------------------------------------------
- Reads the original project data file: "Project Network Input Data.txt"
- Builds the directed network (24 nodes, 76 links)
- Reads the OD matrix for the 14 centroid nodes
- Solves the static UE using a simple MSA (Method of Successive Averages)
  with BPR-type link travel time function

"""

import numpy as np
from collections import defaultdict
import heapq

DATA_FILE = "Project Network Input Data.txt"

# ------------------------------------------------------------
# 1. Read project input file
# ------------------------------------------------------------

with open(DATA_FILE, "r") as f:
    lines = f.read().splitlines()

# First 76 lines: link data
link_lines = lines[:76]


centroids_line = lines[76]
centroid_nodes = list(map(int, centroids_line.split()))


# Remaining lines: OD entries (origin, destination, demand)
od_lines = lines[78:]

num_links = len(link_lines)


tail = np.zeros(num_links + 1, dtype=int)
head = np.zeros(num_links + 1, dtype=int)
t0 = np.zeros(num_links + 1, dtype=float)
cap = np.zeros(num_links + 1, dtype=float)

for i, line in enumerate(link_lines, start=1):
    parts = line.split()
    link_id = int(parts[0])
    assert link_id == i, "Link IDs in the data file must be 1..76 in order."
    tail[i] = int(parts[1])
    head[i] = int(parts[2])
    t0[i] = float(parts[3])                       # free-flow travel time (minutes)
    cap[i] = float(parts[4]) / 1000.0            # convert veh/hr to (10^3 veh/hr)

num_nodes = max(max(tail[1:]), max(head[1:]))

# Build OD matrix for centroid nodes
zones = centroid_nodes             # e.g. [1,2,4,5,10,11,13,14,15,19,20,21,22,24]
nz = len(zones)
zone_index = {z: i for i, z in enumerate(zones)}    # node -> 0..nz-1

demand = np.zeros((nz, nz))
for line in od_lines:
    parts = line.split()
    o = int(parts[0])
    d = int(parts[1])
    q = float(parts[2]) / 1000.0    # convert veh/hr to (10^3 veh/hr)
    demand[zone_index[o], zone_index[d]] = q

print("Network loaded.")
print(f"  Nodes: {num_nodes}")
print(f"  Links: {num_links}")
print(f"  Centroid nodes (zones): {zones}")
print(f"  Total demand (10^3 veh/hr): {demand.sum():.2f}")

# ------------------------------------------------------------
# 2. Shortest path and all-or-nothing assignment
# ------------------------------------------------------------

def build_adjacency(times):
    """Build adjacency list with current link travel times."""
    adj = [[] for _ in range(num_nodes + 1)]
    for a in range(1, num_links + 1):
        adj[tail[a]].append((head[a], a, times[a]))
    return adj


def shortest_paths_from_origin(origin, times):
    adj = build_adjacency(times)
    INF = 1e18
    dist = [INF] * (num_nodes + 1)
    pred_link = [-1] * (num_nodes + 1)

    dist[origin] = 0.0
    heap = [(0.0, origin)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, a, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred_link[v] = a
                heapq.heappush(heap, (nd, v))

    return dist, pred_link


def all_or_nothing(times):
    flows = np.zeros(num_links + 1)
    for oi, o_node in enumerate(zones):
        dist, pred = shortest_paths_from_origin(o_node, times)
        for dj, d_node in enumerate(zones):
            q = demand[oi, dj]
            if q <= 0 or o_node == d_node:
                continue
            # Backtrack from destination to origin using the predecessor tree
            path_links = []
            cur = d_node
            while cur != o_node and cur != -1:
                a = pred[cur]
                if a == -1:
                    raise RuntimeError(f"No path from {o_node} to {d_node}")
                path_links.append(a)
                cur = tail[a]
            if cur != o_node:
                raise RuntimeError(f"No path from {o_node} to {d_node}")
            # Add flow q on each link in the path
            for a in path_links:
                flows[a] += q
    return flows

# ------------------------------------------------------------
# 3. Link performance function and performance measures
# ------------------------------------------------------------

def link_travel_times(flow):
    """
    BPR-type travel time function:
        t_a(x) = t0_a * (1 + 0.15 * (x_a / c_a)^4 )
    """
    times = np.zeros(num_links + 1)
    for a in range(1, num_links + 1):
        xa = flow[a]
        ca = cap[a]
        times[a] = t0[a] * (1.0 + 0.15 * (xa / ca) ** 4)
    return times


def total_system_travel_time(flow):
    times = link_travel_times(flow)
    return float((times[1:] * flow[1:]).sum())


def vc_ratios(flow):
    return flow[1:] / cap[1:]

# ------------------------------------------------------------
# 4. MSA-based User Equilibrium solver
# ------------------------------------------------------------

def solve_ue_msa(max_iter=100, tol=1e-4):
    """
    Solve static UE by Method of Successive Averages (MSA).

    Algorithm:
        x^0 = all-or-nothing assignment using free-flow times
        For k = 1..max_iter:
            1) t(x^{k-1})
            2) y^k = all-or-nothing using t(x^{k-1})
            3) alpha_k = 1 / (k + 1)
            4) x^k = x^{k-1} + alpha_k * (y^k - x^{k-1})
            5) check relative gap; stop if small
    """
    # Initial all-or-nothing using free-flow times
    x = all_or_nothing(t0)
    print("Initial all-or-nothing done.")
    print(f"  Initial total system TT = {total_system_travel_time(x):.2f} veh·min")

    for k in range(1, max_iter + 1):
        times = link_travel_times(x)
        y = all_or_nothing(times)
        alpha = 1.0 / (k + 1)
        x_new = x + alpha * (y - x)

        # L1 relative gap
        gap = np.linalg.norm(x_new - x, 1) / (x_new[1:].sum() + 1e-6)
        tstt = total_system_travel_time(x_new)
        print(f"Iter {k:3d}: alpha={alpha:.4f}, gap={gap:.6f}, TSTT={tstt:.2f}")

        x = x_new
        if gap < tol:
            print("Convergence achieved.")
            break

    return x

# ------------------------------------------------------------
# 5. Run UE solver and print bottlenecks
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Solving user equilibrium...")
    flow_ue = solve_ue_msa(max_iter=100, tol=1e-3)

    times_ue = link_travel_times(flow_ue)
    vc = vc_ratios(flow_ue)

    print("\n=== Final UE summary ===")
    print(f"Total system travel time = {total_system_travel_time(flow_ue):.2f} veh·min")
    print(f"Total demand            = {demand.sum():.2f} (10^3 veh/hr)")

    # Print top-10 most congested links by v/c
    indices = np.argsort(-vc)  # descending
    print("\nTop 10 most congested links (by v/c):")
    print("rank  link   from  to    flow   cap    v/c   t(min)")
    for rank in range(10):
        a = indices[rank] + 1   # because vc is 0..num_links-1
        print(
            f"{rank+1:>4d}  {a:>4d}  {tail[a]:>4d}  {head[a]:>3d} "
            f"{flow_ue[a]:6.2f} {cap[a]:6.2f} {vc[a-1]:5.2f} {times_ue[a]:7.2f}"
        )


import numpy as np
np.save("flow_baseline.npy", flow_ue)      # baseline UE

