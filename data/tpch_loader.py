import random
from pathlib import Path

from typing import Any, Dict, List, Optional
from pathlib import Path

import absl
import numpy as np
import yaml

from more_itertools import before_and_after

from utils import EventTime
from workload import (
    Workload,
    WorkProfile,
    Job,
    JobGraph,
    ExecutionStrategy,
    ExecutionStrategies,
    Resource,
    Resources,
)

from .base_workload_loader import BaseWorkloadLoader


"""
- [ ] Release policy based on workload
- [ ] Fix current time setting
- [ ] Configure deadline variance 
- [ ] Configure release policy
"""


class TpchLoader(BaseWorkloadLoader):
    """Loads the TPCH trace from the provided file

    Args:
        path (`str`): Path to a YAML file specifying the TPC-H query DAGs
        flags (`absl.flags`): The flags used to initialize the app, if any
    """

    def __init__(self, path: str, flags: "absl.flags") -> None:
        self._flags = flags
        self._rng_seed = flags.random_seed
        self._rng = random.Random(self._rng_seed)
        self._loop_timeout = flags.loop_timeout
        self._num_queries = flags.tpch_num_queries
        self._dataset_size = flags.tpch_dataset_size
        if flags.workload_profile_path:
            self._workload_profile_path = str(
                Path(flags.workload_profile_path) / f"{self._dataset_size}g"
            )
        else:
            self._workload_profile_path = "./profiles/workload/tpch/decima/2g"
        self._workload_update_interval = EventTime(10, EventTime.Unit.US)
        release_policy = self._get_release_policy()
        self._release_times = release_policy.get_release_times(
            completion_time=EventTime(self._flags.loop_timeout, EventTime.Unit.US)
        )

        with open(path, "r") as f:
            workload_data = yaml.safe_load(f)

        job_graphs = {}
        for query in workload_data["graphs"]:
            query_name = query["name"]
            graph = query["graph"]
            job_graph = TpchLoader.make_job_graph(
                query_name=query_name,
                graph=graph,
                profile_path=self._workload_profile_path,
            )
            job_graphs[query_name] = job_graph

        self._job_graphs = job_graphs
        self._workload = Workload.empty(flags)

    def _get_release_policy(self):
        release_policy_args = {}
        if self._flags.override_release_policy == "periodic":
            release_policy_args = {
                "period": EventTime(
                    self._flags.override_arrival_period, EventTime.Unit.US
                ),
            }
        elif self._flags.override_release_policy == "fixed":
            release_policy_args = {
                "period": EventTime(
                    self._flags.override_arrival_period, EventTime.Unit.US
                ),
                "num_invocations": self._flags.override_num_invocation,
            }
        elif self._flags.override_release_policy == "poisson":
            release_policy_args = {
                "rate": self._flags.override_poisson_arrival_rate,
                "num_invocations": self._flags.override_num_invocation,
            }
        elif self._flags.override_release_policy == "gamma":
            release_policy_args = {
                "rate": self._flags.override_poisson_arrival_rate,
                "num_invocations": self._flags.override_num_invocation,
                "coefficient": self._flags.override_gamma_coefficient,
            }
        elif self._flags.override_release_policy == "fixed_gamma":
            release_policy_args = {
                "variable_arrival_rate": self._flags.override_poisson_arrival_rate,
                "base_arrival_rate": self._flags.override_base_arrival_rate,
                "num_invocations": self._flags.override_num_invocation,
                "coefficient": self._flags.override_gamma_coefficient,
            }
        else:
            raise NotImplementedError(
                f"Release policy {self._flags.override_release_policy} not implemented."
            )

        # Check that none of the arg values are None
        assert all([val is not None for val in release_policy_args.values()])

        # Construct the release policy
        start_time = EventTime(
            time=self._rng.randint(
                self._flags.randomize_start_time_min,
                self._flags.randomize_start_time_max,
            ),
            unit=EventTime.Unit.US,
        )
        release_policy = getattr(
            JobGraph.ReleasePolicy, self._flags.override_release_policy
        )(start=start_time, rng_seed=self._rng_seed, **release_policy_args)

        return release_policy

    @staticmethod
    def make_job_graph(
        query_name: str, graph: List[Dict[str, Any]], profile_path: str
    ) -> JobGraph:
        job_graph = JobGraph(
            name=query_name,
            # TODO: make configurable
            deadline_variance=(10, 50),
        )

        query_num = int(query_name[1:])
        profiler_data = TpchLoader.get_profiler_data_for_query(profile_path, query_num)

        name_to_job = {}
        for node in graph:
            worker_profile = TpchLoader.load_query_profile(
                profiler_data=profiler_data,
                query_name=query_name,
                node_name=node["name"],
            )
            job = Job(name=node["name"], profile=worker_profile)
            name_to_job[node["name"]] = job
            job_graph.add_job(job=job)

        for node in graph:
            job = name_to_job[node["name"]]
            if "children" in node:
                for child in node["children"]:
                    if child not in name_to_job:
                        raise ValueError(
                            f"Child {child} of {node['name']} was "
                            f"not present in the graph."
                        )
                    child_job = name_to_job[child]
                    job_graph.add_child(job, child_job)

        return job_graph

    @staticmethod
    def load_query_profile(
        profiler_data: Dict[int, Dict[str, Any]], query_name: str, node_name: str
    ) -> WorkProfile:
        profile = profiler_data[int(node_name)]
        resources = Resources(
            resource_vector={
                Resource(name="Slot", _id="any"): profile["num_tasks"],
            },
        )
        execution_strategies = ExecutionStrategies()
        execution_strategies.add_strategy(
            strategy=ExecutionStrategy(
                resources=resources,
                batch_size=1,
                runtime=EventTime(profile["avg_task_duration"], EventTime.Unit.US),
            ),
        )
        return WorkProfile(
            name=f"{query_name}_{node_name}_execution_profile",
            execution_strategies=execution_strategies,
        )

    @staticmethod
    def get_profiler_data_for_query(
        profile_path: str, query_num: int
    ) -> Dict[int, Dict[str, Any]]:
        def pre_process_task_duration(task_duration):
            # remove fresh durations from first wave
            clean_first_wave = {}
            for e in task_duration["first_wave"]:
                clean_first_wave[e] = []
                fresh_durations = SetWithCount()
                for d in task_duration["fresh_durations"][e]:
                    fresh_durations.add(d)
                for d in task_duration["first_wave"][e]:
                    if d not in fresh_durations:
                        clean_first_wave[e].append(d)
                    else:
                        # prevent duplicated fresh duration blocking first wave
                        fresh_durations.remove(d)

        task_durations = np.load(
            Path(profile_path) / f"task_duration_{query_num}.npy",
            allow_pickle=True,
        ).item()

        num_nodes = len(task_durations)

        stage_info = {}

        for n in range(num_nodes):
            task_duration = task_durations[n]
            e = next(iter(task_duration["first_wave"]))

            num_tasks = len(task_duration["first_wave"][e]) + len(
                task_duration["rest_wave"][e]
            )

            # remove fresh duration from first wave duration
            # drag nearest neighbor first wave duration to empty spots
            pre_process_task_duration(task_duration)
            rough_duration = np.mean(
                [i for t in task_duration["first_wave"].values() for i in t]
                + [i for t in task_duration["rest_wave"].values() for i in t]
                + [i for t in task_duration["fresh_durations"].values() for i in t]
            )

            curr_stage = {
                "stage_id": n,
                "num_tasks": num_tasks,
                "avg_task_duration": round(rough_duration),
            }
            stage_info[n] = curr_stage

        return stage_info

    def get_next_workload(self, current_time: EventTime) -> Optional[Workload]:
        if len(self._release_times) == 0:
            return None
        to_release, self._release_times = before_and_after(lambda t: t <= current_time + self._workload_update_interval, self._release_times)
        for t in to_release:
            query_num = self._rng.randint(1, len(self._job_graphs))
            query_name = f"Q{query_num}"
            job_graph = self._job_graphs[query_name]
            task_graph = job_graph.get_next_task_graph(
                start_time=t,
                _flags=self._flags,
            )
            self._workload.add_task_graph(task_graph)
        self._release_times = list(self._release_times)
        return self._workload


class SetWithCount(object):
    """
    allow duplication in set
    """

    def __init__(self):
        self.set = {}

    def __contains__(self, item):
        return item in self.set

    def add(self, item):
        if item in self.set:
            self.set[item] += 1
        else:
            self.set[item] = 1

    def clear(self):
        self.set.clear()

    def remove(self, item):
        self.set[item] -= 1
        if self.set[item] == 0:
            del self.set[item]
