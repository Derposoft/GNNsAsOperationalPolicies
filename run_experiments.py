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
import argparse
import ray._private.services as services

# STABLE: 10 procs, 30 cpu

N_PROCS = 20  # multiprocessing.cpu_count() // 2
N_CPUS = 40
N_SEEDS = 10  # TODO change back to 10 after tuning
START_SEED = 0
# 60k eps required; n_eps = TRAIN_TIME * batch_size, so this number has to go up if we use a smaller batch size and down if we use a larger batch size
# TRAIN_TIME = 150  # equivalent to 60k steps. scout env models tend to plateau at ~50k and skirmish at ~30-40k (?)
TRAIN_TIME = 2500  # equivalent to 1m steps. using to test change in std dev.


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", "-p", type=int, required=True, help="port to run ray cluster on"
    )
    config = parser.parse_args()

    # start ray cluster
    port = config.port
    os.system(
        f"CUDA_VISIBLE_DEVICES=4,5,6,7 ray start --head --port={port} --num-cpus={N_CPUS} --num-gpus=4"
    )

    # read experiments.json
    with open("configs/experiments/experiments.json", "r") as f:
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

    # run experiments
    print(f"running on {multiprocessing.cpu_count()} cpus")
    print(f"choosing to run {N_PROCS} processes")
    address = f"localhost:{port}"
    address = services.canonicalize_bootstrap_address_or_die(address)
    os.system(f"echo {address} > /tmp/ray/ray_current_cluster")  # bad hacc go brrr
    pool = ThreadPool(N_PROCS)
    results = []
    for cmd in experiment_cmds:
        print(f"starting exp parametrized by: {cmd}")
        results.append(pool.apply_async(call_proc, (cmd,)))
    pool.close()
    pool.join()
