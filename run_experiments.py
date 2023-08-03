"""
experiments must be defined in configs/experiments.json in the following format:

{
    "experiment1": {
        "flag1 (e.g. fixed_start or init_health)": 2
    },
    "experiment2": {
        "flag1": 1,
        "flag2": 2
    },
    ...
}
"""

import json
import multiprocessing
import subprocess
import shlex
from multiprocessing.pool import ThreadPool
import os


N_PROCS = 5  # multiprocessing.cpu_count() // 2
N_SEEDS = 10
START_SEED = 0
TRAIN_TIME = 75  # equivalent to 60k eps. scout env models tend to plateau at ~50k and skirmish at ~30-40k (?)

# read experiments.json
with open("configs/experiments/experiments_policy.json", "r") as f:
    experiments = json.load(f)

# turn experiments in json into cmdline commands
experiment_cmds = []
for experiment_name in experiments:
    flags = experiments[experiment_name]
    for i in range(N_SEEDS):
        flags["name"] = experiment_name + f"_SEED{i+START_SEED}"
        experiment_cmds.append(
            "python train.py"
            + "".join([f" --{flag} {flags[flag]}" for flag in flags])
            + f" --seed {i+START_SEED}"
            + f" --train_time {TRAIN_TIME}"
        )


# https://stackoverflow.com/questions/25120363/multiprocessing-execute-external-command-and-wait-before-proceeding
def call_proc(cmd):
    p = subprocess.Popen(
        shlex.split(
            cmd
        )  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True # for some reason these break on windows if stdout/err is redirected??
    )
    # out, err = p.communicate()
    # return (out, err)
    return p.wait()


# start ray cluster
os.system("ray start --head --port=6379")

print(f"running on {multiprocessing.cpu_count()} cpus")
print(f"choosing to run {N_PROCS} processes")
pool = ThreadPool(N_PROCS)
results = []
for cmd in experiment_cmds:
    print(f"starting exp parametrized by: {cmd}")
    results.append(pool.apply_async(call_proc, (cmd,)))
pool.close()
pool.join()
