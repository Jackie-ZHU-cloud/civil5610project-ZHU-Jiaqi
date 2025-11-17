
"""
UE_plots.py

"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Step 0. Load original network (76 links from project txt)
# ============================================================

def load_network(filename="Project Network Input Data.txt"):
    """Load the original 76-link network."""
    with open(filename,"r") as f:
        lines=f.read().splitlines()

    link_lines = lines[:76]

    tail=[0]; head=[0]; t0=[0]; cap=[0]
    for line in link_lines:
        p=line.split()
        tail.append(int(p[1]))
        head.append(int(p[2]))
        t0.append(float(p[3]))
        cap.append(float(p[4]) / 1000.0)  # convert to 10^3 veh/hr

    return tail, head, np.array(t0), np.array(cap)

tail_base, head_base, t0_base76, cap_base76 = load_network()
num_links_base = 76


# ============================================================
# Step 1. Load UE flow results
# ============================================================

flow_base = np.load("flow_baseline.npy")      # 76 links
flow_s1   = np.load("flow_scheme1.npy")       # 78 links
flow_s2   = np.load("flow_scheme2.npy")       # 76 links
flow_s3   = np.load("flow_combined.npy")      # 78 links

flows = [flow_base, flow_s1, flow_s2, flow_s3]
scenario_names = ["Baseline","Scheme 1","Scheme 2","Combined"]


# ============================================================
# Step 2. Build t0/cap for each scenario (with expansion & new links)
# ============================================================

# 扩容信息
EXPANDED_LINKS = [2, 57]
EXPAND_AMOUNT = 4.0   # +4 (10^3 veh/hr)

# 新建 C#1, C#2 的参数
NEW_LINK_T0 = [2.4, 1.8]
NEW_LINK_CAP = [6.0, 6.0]

# 收费信息
TOLL_LINKS = [2, 66, 75, 34, 40, 14, 57, 39, 19]
TOLL_TIME = 0.5   # 10 HK$ = 0.5 min

def build_t0_cap_for_scenario(flow, expanded=False):
    """
    根据场景构造对应的 t0 / cap:
      - expanded=True: 对 link 2 和 57 进行扩容
      - 如果 flow 带有新建 link (长度 78)，就在末尾追加 C#1, C#2 的 t0/cap
    """
    t0_use = np.copy(t0_base76)
    cap_use = np.copy(cap_base76)

    # 扩容
    if expanded:
        for a in EXPANDED_LINKS:
            cap_use[a] += EXPAND_AMOUNT

    # 新建链接 (Scheme1 / Combined，flow 长度为 79，前面有一个 0)
    L = len(flow) - 1
    if L > 76:
        t0_use = np.append(t0_use, NEW_LINK_T0)
        cap_use = np.append(cap_use, NEW_LINK_CAP)

    return t0_use, cap_use


# ============================================================
# Step 3. Travel time and TSTT functions (with pricing option)
# ============================================================

def travel_time(flow, t0, cap, tolled=False):
    """
    BPR travel time function; 若 tolled=True，则在收费路段额外加 0.5 分钟。
    """
    L = len(flow) - 1
    times = np.zeros(L+1)
    for a in range(1, L+1):
        base = t0[a] * (1 + 0.15 * (flow[a] / cap[a])**4)
        if tolled and a in TOLL_LINKS:
            base += TOLL_TIME
        times[a] = base
    return times

def TSTT(flow, t0, cap, tolled=False):
    """Total System Travel Time (veh·min)."""
    tt = travel_time(flow, t0, cap, tolled=tolled)
    return float((tt[1:] * flow[1:]).sum())



scenario_cfg = [
    ("Baseline", False, False),
    ("Scheme 1", False, True),   # 扩容 + 新路
    ("Scheme 2", True,  False),  # 收费
    ("Combined", True,  True),   # 扩容 + 新路 + 收费
]

TSTT_vals = []
vc_ratios = []

for (name, tolled, expanded), flow in zip(scenario_cfg, flows):
    t0_use, cap_use = build_t0_cap_for_scenario(flow, expanded=expanded)
    TSTT_vals.append(TSTT(flow, t0_use, cap_use, tolled=tolled))
    # v/c 用真实 capacity（含扩容、新路）
    L = len(flow) - 1
    vc_ratios.append(flow[1:] / cap_use[1:L+1])

# 打印
print("T_baseline =", TSTT_vals[0])
print("T_scheme1 =", TSTT_vals[1])
print("T_scheme2 =", TSTT_vals[2])
print("T_combined =", TSTT_vals[3])


# ============================================================
# Step 4. Generate plots
# ============================================================


# 1. Bar chart — TSTT comparison (delta vs Baseline)
T_base = TSTT_vals[0]
delta_TSTT = [0.0]  # baseline is reference
for val in TSTT_vals[1:]:
    delta_TSTT.append(val - T_base)

plt.figure(figsize=(9,6))
plt.bar(scenario_names, delta_TSTT,
        color=["black","blue","red","green"])
plt.ylabel("ΔTSTT relative to Baseline (veh·min)")
plt.title("Scenario Comparison — Change in Total System Travel Time")
plt.axhline(0, color="gray", linewidth=1)
plt.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig("plot_TSTT_delta.png", dpi=300)
plt.show()

plt.figure(figsize=(9,6))
plt.bar(scenario_names, TSTT_vals,
        color=["black","blue","red","green"])
plt.ylabel("Total System Travel Time (veh·min)")
plt.title("Scenario Comparison — TSTT (absolute)")
plt.ylim(min(TSTT_vals)-30, max(TSTT_vals)+30)  # zoom in around 1100
plt.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig("plot_TSTT_absolute.png", dpi=300)
plt.show()


# 2. Sorted v/c curves
plt.figure(figsize=(10,7))
for vc, label in zip(vc_ratios, scenario_names):
    plt.plot(sorted(vc), label=label, linewidth=2)

plt.xlabel("Link Rank (sorted)")
plt.ylabel("v/c ratio")
plt.title("Network-wide v/c Distribution")
plt.legend()
plt.grid(ls="--", alpha=0.5)
plt.savefig("plot_vc_sorted.png", dpi=300)
plt.show()


# 3. Baseline top 10 bottlenecks
vc_base = vc_ratios[0]
idx_top = np.argsort(-vc_base)[:10]

plt.figure(figsize=(10,6))
plt.bar([f"L{a}" for a in (idx_top+1)], vc_base[idx_top], color="darkred")
plt.title("Top 10 Bottlenecks — Baseline")
plt.ylabel("v/c ratio")
plt.grid(axis="y", ls="--")
plt.savefig("plot_top10_bottlenecks.png", dpi=300)
plt.show()


# 4. Flow difference plots (only on original 76 links)

flow_base_cut = flow_base[1:77]   # 76 original links

# For Scheme 1 and Combined (78 links), only compare first 76 links
flow_diff_s1 = flow_s1[1:77] - flow_base_cut
flow_diff_s2 = flow_s2[1:77] - flow_base_cut    # same length 76
flow_diff_s3 = flow_s3[1:77] - flow_base_cut

plt.figure(figsize=(12,6))
plt.plot(range(1,77), flow_diff_s1, label="Scheme 1")
plt.plot(range(1,77), flow_diff_s2, label="Scheme 2")
plt.plot(range(1,77), flow_diff_s3, label="Combined")
plt.axhline(0, color="black", lw=1)
plt.title("Flow Difference vs Baseline (Original 76 Links)")
plt.xlabel("Link ID")
plt.ylabel("Flow Difference (10^3 veh/hr)")
plt.legend()
plt.grid(ls="--")
plt.tight_layout()
plt.savefig("plot_flow_difference.png", dpi=300)
plt.show()

#  show flows on the two new links in Scheme 1 and Combined
new_links_ids = [77, 78]
new_s1 = flow_s1[77:79]
new_s3 = flow_s3[77:79]

plt.figure(figsize=(6,5))
x = np.arange(len(new_links_ids))
width = 0.35
plt.bar(x - width/2, new_s1, width, label="Scheme 1")
plt.bar(x + width/2, new_s3, width, label="Combined")
plt.xticks(x, [f"L{lid}" for lid in new_links_ids])
plt.ylabel("Flow (10^3 veh/hr)")
plt.title("Flows on New Links C#1 and C#2")
plt.legend()
plt.grid(axis="y", ls="--")
plt.tight_layout()
plt.savefig("plot_new_links_flow.png", dpi=300)
plt.show()

