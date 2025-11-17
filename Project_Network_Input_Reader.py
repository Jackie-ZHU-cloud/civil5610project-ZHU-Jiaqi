import numpy as np

def load_network(filename="Project Network Input Data.txt"):
    with open(filename,"r") as f:
        lines=f.read().splitlines()
    link_lines=lines[:76]
    tail=[0]; head=[0]; t0=[0]; cap=[0]
    for line in link_lines:
        p=line.split()
        tail.append(int(p[1]))
        head.append(int(p[2]))
        t0.append(float(p[3]))
        cap.append(float(p[4])/1000.0)
    return tail, head, t0, cap
