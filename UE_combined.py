
"""
Combined Policy:
- Expand link 2 & 57
- Add C#1, C#2
- Apply tolls on top 9 congested links
"""

import numpy as np
import heapq

DATA_FILE="Project Network Input Data.txt"

# ===========================
# Load original + expand + add
# ===========================

with open(DATA_FILE,"r") as f:
    lines=f.read().splitlines()

link_lines=lines[:76]
centroids=list(map(int,lines[76].split()))
od_lines=lines[78:]

tail=[0]; head=[0]; t0=[0]; cap=[0]

for line in link_lines:
    p=line.split()
    tail.append(int(p[1]))
    head.append(int(p[2]))
    t0.append(float(p[3]))
    cap.append(float(p[4])/1000.0)

num_links=76
num_nodes=max(max(tail),max(head))
zones=centroids
nz=len(zones)
zone_idx={z:i for i,z in enumerate(zones)}
demand=np.zeros((nz,nz))
for line in od_lines:
    p=line.split()
    o,d,q=int(p[0]),int(p[1]),float(p[2])/1000.0
    demand[zone_idx[o],zone_idx[d]]=q

# Expand
cap[2]+=4.0
cap[57]+=4.0

# Add links C#1 and C#2
new_links=[(13,21,2.4,6.0),(24,21,1.8,6.0)]
for fr,to,t0n,capn in new_links:
    tail.append(fr); head.append(to); t0.append(t0n); cap.append(capn)
    num_links+=1

# Tolls
toll_links=[2,66,75,34,40,14,57,39,19]
toll_time=0.5

print("=== Combined Scheme ===")
print(f"Total links after add = {num_links}")

# ============ UE functions ============
def build_adj(times):
    adj=[[] for _ in range(num_nodes+1)]
    for a in range(1,num_links+1):
        adj[tail[a]].append((head[a],a,times[a]))
    return adj

def dijkstra(origin,times):
    INF=1e18
    dist=[INF]*(num_nodes+1)
    pred=[-1]*(num_nodes+1)
    dist[origin]=0
    h=[(0,origin)]
    adj=build_adj(times)
    while h:
        d,u=heapq.heappop(h)
        if d>dist[u]: continue
        for v,a,w in adj[u]:
            nd=d+w
            if nd<dist[v]:
                dist[v]=nd; pred[v]=a
                heapq.heappush(h,(nd,v))
    return dist,pred

def all_or_nothing(times):
    flow=np.zeros(num_links+1)
    for oi,o in enumerate(zones):
        dist,pred=dijkstra(o,times)
        for dj,d in enumerate(zones):
            q=demand[oi,dj]
            if q<=0 or o==d: continue
            cur=d
            while cur!=o:
                a=pred[cur]
                if a==-1: raise RuntimeError("No path")
                flow[a]+=q
                cur=tail[a]
    return flow

def travel_time(flow):
    times=np.zeros(num_links+1)
    for a in range(1,num_links+1):
        base=t0[a]*(1+0.15*(flow[a]/cap[a])**4)
        if a in toll_links:
            base+=toll_time
        times[a]=base
    return times

def TSTT(flow):
    times=travel_time(flow)
    return float((times[1:]*flow[1:]).sum())

def solve_ue(max_iter=80,tol=1e-3):
    x=all_or_nothing(t0)
    for k in range(1,max_iter+1):
        y=all_or_nothing(travel_time(x))
        alpha=1/(k+1)
        x_new=x+alpha*(y-x)
        gap=np.linalg.norm(x_new-x,1)/(x_new.sum()+1e-6)
        print(f"Iter {k:2d}: gap={gap:.5f}, TSTT={TSTT(x_new):.2f}")
        x=x_new
        if gap<tol: break
    return x

flow_ue=solve_ue()

print("\n=== Combined Policy Completed ===")
print(f"TSTT = {TSTT(flow_ue):.2f}")
vc=flow_ue[1:]/cap[1:]
idx=np.argsort(-vc)
print("Top congested links after Combined scheme:")
for i in range(10):
    a=idx[i]+1
    print(f"{i+1}. Link {a}: v/c={vc[a-1]:.2f}, flow={flow_ue[a]:.2f}")


import numpy as np
np.save("flow_combined.npy", flow_ue)      # combined UE
