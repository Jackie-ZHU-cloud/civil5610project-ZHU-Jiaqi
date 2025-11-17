
"""
run_the_model.py
----------------------------------------
One-click runner for CIVL5610 transport project:
It will automatically run:

1. Project_Network_Input_Reader.py
2. UE_baseline.py
3. UE_scheme1.py
4. UE_scheme2.py
5. UE_combined.py
6. UE_plots.py

----------------------------------------
"""

import subprocess
import time

scripts = [
    "Project_Network_Input_Reader.py",   # only if needed
    "UE_baseline.py",
    "UE_scheme1.py",
    "UE_scheme2.py",
    "UE_combined.py",
    "UE_plots.py"
]

print("=======================================")
print(" Running Full CIVL5610 Project Pipeline ")
print("=======================================")

for script in scripts:
    print(f"\n>>> Running {script} ...")
    try:
        subprocess.run(["python", script], check=True)
        print(f">>> Finished {script}\n")
    except subprocess.CalledProcessError:
        print(f"!!! ERROR: Script {script} failed.")
        break

print("=======================================")
print("   All tasks completed successfully.    ")
print("=======================================")
time.sleep(1)
