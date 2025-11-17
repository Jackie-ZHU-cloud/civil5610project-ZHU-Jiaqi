
"""
Scheme 1 for CIVL5610 Project:
- Expand link 2 (+4000 veh/hr)
- Expand link 57 (+4000 veh/hr)
- Add new links C#1 (13→21) and C#2 (24→21)
- Solve UE via MSA
"""

import numpy as np
from collections import defaultdict
import heapq

DATA_FILE = "Project Network Input Data.txt"

# ===============================
# 1. Load original network
# ===============================

with open(DATA_FILE, "r") as f:
    lines = f.read().splitlines()

# first 76 links
link_lines = lines[:76]
centroids = list(map(int, lines[76].split()))
od_lines = lines[78:]

num_links0 = len(link_lines)
tail = []
head = []
t0 = []
cap = []

for line in link_lines:
    p = line.split()
    tail.append(int(p[1]))
    head.append(int(p[2]))
    t0.append(float(p[3]))
    cap.append(float(p[4]) / 1000.0)  # convert to 10^3 veh/hr

tail = [0] + tail
head = [0] + head
t0 = [0] + t0
cap = [0] + cap

num_links = num_links0
num_nodes = max(max(tail), max(head))

# OD matrix for 14 centroid nodes
zones = centroids
nz = len(zones)
zone_idx = {z: i for i, z in enumerate(zones)}
demand = np.zeros((nz, nz))

for line in od_lines:
    p = line.split()
    o, d, q = int(p[0]), int(p[1]), float(p[2])/1000.0
    demand[zone_idx[o], zone_idx[d]] = q

print("=== Scheme 1: Expand + Add Links ===")
print("Original network loaded.")

# ===============================
# 2. Apply Scheme 1 modifications
# ===============================

# Expand link 2 and 57 (increase capacity by +4000 veh/hr = 4.0 in 10^3 units)
cap[2] += 4.0
cap[57] += 4.0

# Add new links C#1 and C#2
# C#1: 13 -> 21, t0=2.4 min, cap=4.0 or 6.0 → choose 6.0
# C#2: 24 -> 21, t0=1.8 min, cap=4.0 or 6.0 → choose 6.0

new_links = [
    (13, 21, 2.4, 6.0),   # C#1
    (24, 21, 1.8, 6.0)    # C#2
]

for fr, to, t0_new, cap_new in new_links:
    tail.append(fr)
    head.append(to)
    t0.append(t0_new)
    cap.append(cap_new)
    num_links += 1

print(f"New total links = {num_links}")

# ===============================
# 3. Shortest paths
# ===============================

def build_adj(times):
    adj = [[] for _ in range(num_nodes + 1)]
    for a in range(1, num_links + 1):
        adj[tail[a]].append((head[a], a, times[a]))
    return adj

def dijkstra(origin, times):
    INF = 1e18
    dist = [INF]*(num_nodes+1)
    pred = [-1]*(num_nodes+1)
    dist[origin] = 0
    h = [(0,origin)]
    adj = build_adj(times)

    while h:
        d,u = heapq.heappop(h)
        if d>dist[u]: continue
        for v,a,w in adj[u]:
            nd = d+w
            if nd<dist[v]:
                dist[v]=nd
                pred[v]=a
                heapq.heappush(h,(nd,v))
    return dist,pred

def all_or_nothing(times):
    flow = np.zeros(num_links+1)
    for oi,o in enumerate(zones):
        dist,pred = dijkstra(o,times)
        for dj,d in enumerate(zones):
            q = demand[oi,dj]
            if q<=0 or o==d: continue
            # backtrack
            cur=d
            while cur!=o:
                a = pred[cur]
                if a==-1: raise RuntimeError(f"No path {o}->{d}")
                flow[a]+=q
                cur = tail[a]
    return flow

# ===============================
# 4. UE via MSA
# ===============================

def tt(flow):
    times = np.zeros(num_links+1)
    for a in range(1,num_links+1):
        times[a] = t0[a]*(1+0.15*(flow[a]/cap[a])**4)
    return times

def TSTT(flow):
    return float((flow[1:]*tt(flow)[1:]).sum())

def solve_ue(max_iter=80, tol=1e-3):
    x = all_or_nothing(t0)
    print("Initial AON done.")
    for k in range(1,max_iter+1):
        times = tt(x)
        y = all_or_nothing(times)
        alpha = 1/(k+1)
        x_new = x + alpha*(y-x)
        gap = np.linalg.norm(x_new-x,1)/(x_new.sum()+1e-6)
        print(f"Iter {k:2d}: alpha={alpha:.3f}, gap={gap:.5f}, TSTT={TSTT(x_new):.2f}")
        x = x_new
        if gap<tol: break
    return x

flow_ue = solve_ue()

print("\n=== Scheme 1 Completed ===")
print(f"TSTT = {TSTT(flow_ue):.2f}")
vc = flow_ue[1:] / cap[1:]
idx = np.argsort(-vc)
print("Top congested after Scheme 1:")
for i in range(10):
    a = idx[i]+1
    print(f"{i+1:2d} Link {a:2d}: {tail[a]}->{head[a]}, flow={flow_ue[a]:.2f}, cap={cap[a]:.2f}, v/c={vc[a-1]:.2f}")


import numpy as np
np.save("flow_scheme1.npy", flow_ue)       # scheme 1 UE
