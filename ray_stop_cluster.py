import os
import psutil
import ray
import subprocess
import sys
from typing import List, Tuple
import argparse

import ray._private.services as services
from ray.autoscaler._private.constants import RAY_PROCESSES
from ray.autoscaler._private.cli_logger import add_click_logging_options, cf, cli_logger

RAY_STOP_MULTIPLE_CLUSTERS_WARNING = (
    "WARNING: Multiple ray clusters found. Are you sure you want to proceed? "
    "Will automatically proceed in 10 seconds to avoid blocking."
)


def stop(address: str, force: bool, grace_period: int):
    """Stop Ray processes manually on the local machine."""
    is_linux = sys.platform.startswith("linux")
    total_procs_found = 0
    total_procs_stopped = 0
    procs_not_gracefully_killed = []

    def kill_procs(
        force: bool,
        grace_period: int,
        processes_to_kill: List[str],
        proc_arg_filters: List[str] = None,
    ) -> Tuple[int, int, List[psutil.Process]]:
        """Find all processes from `processes_to_kill` that match the list of optional
        process argument filters given and terminate them.

        Unless `force` is specified, it gracefully kills processes. If
        processes are not cleaned within `grace_period`, it force kill all
        remaining processes.

        Returns:
            total_procs_found: Total number of processes found from
                `processes_to_kill` is added.
            total_procs_stopped: Total number of processes gracefully
                stopped from `processes_to_kill` is added.
            procs_not_gracefully_killed: If processes are not killed
                gracefully, they are added here.
        """
        process_infos = []
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                process_infos.append((proc, proc.name(), proc.cmdline()))
            except psutil.Error:
                pass

        stopped = []
        for keyword, filter_by_cmd in processes_to_kill:
            if filter_by_cmd and is_linux and len(keyword) > 15:
                # getting here is an internal bug, so we do not use cli_logger
                msg = (
                    "The filter string should not be more than {} "
                    "characters. Actual length: {}. Filter: {}"
                ).format(15, len(keyword), keyword)
                raise ValueError(msg)

            found = []
            for candidate in process_infos:
                proc, proc_cmd, proc_args = candidate
                proc_args_cmdline = subprocess.list2cmdline(proc_args)
                corpus = proc_cmd if filter_by_cmd else proc_args_cmdline
                matches_arg_filter = (
                    True
                    if proc_arg_filters is None
                    else all(
                        [
                            arg_filter in proc_args_cmdline
                            for arg_filter in proc_arg_filters
                        ]
                    )
                )
                if keyword in corpus and matches_arg_filter:
                    found.append(candidate)
            for proc, proc_cmd, proc_args in found:
                proc_string = str(subprocess.list2cmdline(proc_args))
                try:
                    if force:
                        proc.kill()
                    else:
                        # TODO(mehrdadn): On Windows, this is forceful termination.
                        # We don't want CTRL_BREAK_EVENT, because that would
                        # terminate the entire process group. What to do?
                        proc.terminate()

                    if force:
                        cli_logger.verbose(
                            "Killed `{}` {} ",
                            cf.bold(proc_string),
                            cf.dimmed("(via SIGKILL)"),
                        )
                    else:
                        cli_logger.verbose(
                            "Send termination request to `{}` {}",
                            cf.bold(proc_string),
                            cf.dimmed("(via SIGTERM)"),
                        )

                    stopped.append(proc)
                except psutil.NoSuchProcess:
                    cli_logger.verbose(
                        "Attempted to stop `{}`, but process was already dead.",
                        cf.bold(proc_string),
                    )
                except (psutil.Error, OSError) as ex:
                    cli_logger.error(
                        "Could not terminate `{}` due to {}",
                        cf.bold(proc_string),
                        str(ex),
                    )

        # Wait for the processes to actually stop.
        # Dedup processes.
        stopped, alive = psutil.wait_procs(stopped, timeout=0)
        procs_to_kill = stopped + alive
        total_found = len(procs_to_kill)

        # Wait for grace period to terminate processes.
        gone_procs = set()

        def on_terminate(proc):
            gone_procs.add(proc)
            cli_logger.print(f"{len(gone_procs)}/{total_found} stopped.", end="\r")

        stopped, alive = psutil.wait_procs(
            procs_to_kill, timeout=grace_period, callback=on_terminate
        )
        total_stopped = len(stopped)

        # For processes that are not killed within the grace period,
        # we send force termination signals.
        for proc in alive:
            proc.kill()
        # Wait a little bit to make sure processes are killed forcefully.
        psutil.wait_procs(alive, timeout=2)
        return total_found, total_stopped, alive

    if len(services.find_gcs_addresses()) > 1 and address is None:
        cli_logger.interactive = True
        if cli_logger.interactive:
            confirm_kill_all_clusters = cli_logger.confirm(
                False,
                RAY_STOP_MULTIPLE_CLUSTERS_WARNING,
                _default=True,
                _timeout_s=10,
            )
        else:
            cli_logger.print(RAY_STOP_MULTIPLE_CLUSTERS_WARNING)
        if not confirm_kill_all_clusters:
            sys.exit(0)

    # GCS should exit after all other processes exit.
    # Otherwise, some of processes may exit with an unexpected
    # exit code which breaks ray start --block.
    processes_to_kill = RAY_PROCESSES
    gcs = processes_to_kill[0]
    assert gcs[0] == "gcs_server"

    grace_period_to_kill_gcs = int(grace_period / 2)
    grace_period_to_kill_components = grace_period - grace_period_to_kill_gcs

    # Kill everything except GCS.
    non_gcs_proc_arg_filters = (
        [f"--gcs-address={address}"] if address is not None else None
    )
    found, stopped, alive = kill_procs(
        force,
        grace_period_to_kill_components,
        processes_to_kill[1:],
        non_gcs_proc_arg_filters,
    )
    total_procs_found += found
    total_procs_stopped += stopped
    procs_not_gracefully_killed.extend(alive)

    # Kill GCS.
    if address is not None:
        address = services.canonicalize_bootstrap_address_or_die(address)
        url, port = address.split(":")
        gcs_proc_arg_filters = [f"--node-ip-address={url}", f"--gcs_server_port={port}"]
    else:
        gcs_proc_arg_filters = None
    found, stopped, alive = kill_procs(
        force,
        grace_period_to_kill_gcs,
        [gcs],
        gcs_proc_arg_filters,
    )
    total_procs_found += found
    total_procs_stopped += stopped
    procs_not_gracefully_killed.extend(alive)

    # Print the termination result.
    if total_procs_found == 0:
        cli_logger.print("Did not find any active Ray processes.")
    else:
        if total_procs_stopped == total_procs_found:
            cli_logger.success("Stopped all {} Ray processes.", total_procs_stopped)
        else:
            cli_logger.warning(
                f"Stopped only {total_procs_stopped} out of {total_procs_found} "
                f"Ray processes within the grace period {grace_period} seconds. "
                f"Set `{cf.bold('-v')}` to see more details. "
                f"Remaining processes {procs_not_gracefully_killed} "
                "will be forcefully terminated.",
            )
            cli_logger.warning(
                f"You can also use `{cf.bold('--force')}` to forcefully terminate "
                "processes or set higher `--grace-period` to wait longer time for "
                "proper termination."
            )

    # NOTE(swang): This will not reset the cluster address for a user-defined
    # temp_dir. This is fine since it will get overwritten the next time we
    # call `ray start`.
    ray._private.utils.reset_ray_address()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to stop a specific locally-running ray cluster by its port"
    )
    parser.add_argument(
        "--port", "-p", type=int, required=True, help="Port of cluster to stop"
    )
    config = parser.parse_args()
    port = config.port
    grace_period = 120
    force_if_not_killed_within_grace_period = True
    stop(
        f"localhost:{port}",
        force_if_not_killed_within_grace_period,
        grace_period,
    )
