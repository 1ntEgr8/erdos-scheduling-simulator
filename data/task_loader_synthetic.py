import logging
import sys
from typing import Mapping, Optional, Sequence, Tuple

import absl  # noqa: F401

from data import TaskLoader, TaskLoaderPylot
from utils import EventTime, fuzz_time, setup_logging
from workload import Job, JobGraph, Resource, Resources, Task, TaskGraph


class TaskLoaderSynthetic(TaskLoader):
    """Generates a synthetic task workload.

    Args:
        num_perception_sensors (`int`): Number of (camera, lidar) sensor pairs.
        num_traffic_light_cameras (`int`): Number of traffic light sensor cameras.
        _flags (`absl.flags`): The flags used to initialize the app, if any.
    """

    def __init__(
        self,
        num_perception_sensors: int = 1,
        num_traffic_light_cameras: int = 1,
        _flags: Optional["absl.flags"] = None,
    ):
        # Set up the logger.
        if _flags:
            self._logger = setup_logging(
                name=self.__class__.__name__,
                log_dir=_flags.log_dir,
                log_file=_flags.log_file_name,
                log_level=_flags.log_level,
            )
            task_logger = setup_logging(
                name="Task", log_file=_flags.log_file_name, log_level=_flags.log_level
            )
        else:
            self._logger = setup_logging(name=self.__class__.__name__)
            task_logger = setup_logging(name="Task")

        # Create the synthetic JobGraph.
        (
            self._jobs,
            self._job_graph,
            deadlines,
            resources,
        ) = TaskLoaderSynthetic._TaskLoaderSynthetic__create_job_graph(
            num_perception_sensors, num_traffic_light_cameras
        )

        # Create the Tasks and the TaskGraph from the Jobs.
        max_timestamp = (
            _flags.max_timestamp if _flags.max_timestamp is not None else sys.max_size
        )
        self.create_tasks(
            max_timestamp,
            EventTime(_flags.timestamp_difference, EventTime.Unit.US),
            _flags.runtime_variance,
            (_flags.min_deadline_variance, _flags.max_deadline_variance),
            deadlines,
            resources,
            _flags.use_end_to_end_deadlines,
            task_logger,
        )
        self._logger.debug(f"Created {len(self._tasks)} Tasks")
        (
            self._grouped_tasks,
            self._task_graph,
        ) = TaskLoader._TaskLoader__create_task_graph(self._tasks, self._job_graph)
        self._logger.debug("Finished creating TaskGraph from loaded tasks.")

    @staticmethod
    def __create_job_graph(
        num_perception_sensors: int,
        num_traffic_light_cameras: int,
        deadline_slack_factor: float = 1.2,
        periodic_deadline_slack_factor: float = 3.0,
    ):
        """Creates a synthetic Pylot JobGraph.

        Args:
            num_perception_sensors (`int`): Number of cameras the pipeline has.
            num_traffic_light_cameras (`int`): Number of traffic light cameras the
                pipeline has.
            deadline_slack_factor (`float`): Factor multiplied with the task runtimes
                in order to compute task deadlines.
            periodic_deadline_slack_factor (`float`): Factor multiplied with the task
                runtimes in order to compute task deadlines for periodic operators with
                small runtimes.

        Returns:
            A `JobGraph` instance depicting the relation between the different
            `Job`s.
        """
        deadlines = {}
        resources = {}
        gnss = Job(name="gnss", runtime=EventTime(1000, EventTime.Unit.US))
        deadlines[gnss.name] = gnss.runtime.time * periodic_deadline_slack_factor
        resources[gnss.name] = Resources(
            resource_vector={Resource("CPU", _id="any"): 1}
        )
        imu = Job(name="imu", runtime=EventTime(1000, EventTime.Unit.US))
        deadlines[imu.name] = imu.runtime.time * periodic_deadline_slack_factor
        resources[imu.name] = Resources(resource_vector={Resource("CPU", _id="any"): 1})
        localization = Job(
            name="localization", runtime=EventTime(20000, EventTime.Unit.US)
        )
        deadlines[localization.name] = (
            localization.runtime.time * periodic_deadline_slack_factor
        )
        resources[localization.name] = Resources(
            resource_vector={Resource("CPU", _id="any"): 1}
        )
        cameras = []
        lidars = []
        detectors = []
        trackers = []
        object_localization = []
        lane_detectors = []
        for i in range(num_perception_sensors):
            cameras.append(
                Job(
                    name=f"camera_{i}",
                    runtime=EventTime(10000, EventTime.Unit.US),
                    pipelined=True,
                )
            )
            deadlines[cameras[-1].name] = (
                cameras[-1].runtime.time * periodic_deadline_slack_factor
            )
            resources[cameras[-1].name] = Resources(
                resource_vector={Resource("CPU", _id="any"): 1}
            )
            lidars.append(
                Job(
                    name=f"lidar_{i}",
                    runtime=EventTime(8000, EventTime.Unit.US),
                    pipelined=True,
                )
            )
            deadlines[lidars[-1].name] = (
                lidars[-1].runtime.time * periodic_deadline_slack_factor
            )
            resources[lidars[-1].name] = Resources(
                resource_vector={Resource("CPU", _id="any"): 1}
            )
            detectors.append(
                Job(
                    name=f"detection_{i}",
                    runtime=EventTime(130000, EventTime.Unit.US),
                    pipelined=True,
                )
            )
            deadlines[detectors[-1].name] = (
                detectors[-1].runtime.time * deadline_slack_factor
            )
            resources[detectors[-1].name] = Resources(
                resource_vector={
                    Resource("GPU", _id="any"): 1,
                    Resource("CPU", _id="any"): 1,
                }
            )
            trackers.append(
                Job(
                    name=f"tracker_{i}",
                    runtime=EventTime(50000, EventTime.Unit.US),
                    pipelined=False,
                )
            )
            deadlines[trackers[-1].name] = (
                trackers[-1].runtime.time * deadline_slack_factor
            )
            resources[trackers[-1].name] = Resources(
                resource_vector={
                    Resource("GPU", _id="any"): 1,
                    Resource("CPU", _id="any"): 2,
                }
            )
            object_localization.append(
                Job(
                    name=f"obj_localization_{i}",
                    runtime=EventTime(20000, EventTime.Unit.US),
                    pipelined=True,
                )
            )
            deadlines[object_localization[-1].name] = (
                object_localization[-1].runtime.time * deadline_slack_factor
            )
            resources[object_localization[-1].name] = Resources(
                resource_vector={Resource("CPU", _id="any"): 4}
            )
            lane_detectors.append(
                Job(
                    name=f"lane_detection_{i}",
                    runtime=EventTime(90000, EventTime.Unit.US),
                    pipelined=True,
                )
            )
            deadlines[lane_detectors[-1].name] = (
                lane_detectors[-1].runtime.time * deadline_slack_factor
            )
            resources[lane_detectors[-1].name] = Resources(
                resource_vector={
                    Resource("GPU", _id="any"): 1,
                    Resource("CPU", _id="any"): 1,
                }
            )

        tl_cameras = []
        tl_detectors = []
        tl_object_localization = []
        for i in range(num_traffic_light_cameras):
            tl_cameras.append(
                Job(
                    name=f"traffic_light_camera_{i}",
                    runtime=EventTime(10000, EventTime.Unit.US),
                    pipelined=True,
                )
            )
            deadlines[tl_cameras[-1].name] = (
                tl_cameras[-1].runtime.time * periodic_deadline_slack_factor
            )
            resources[tl_cameras[-1].name] = Resources(
                resource_vector={Resource("CPU", _id="any"): 1}
            )
            tl_detectors.append(
                Job(
                    name=f"tl_detection_{i}",
                    runtime=EventTime(95000, EventTime.Unit.US),
                    pipelined=True,
                )
            )
            deadlines[tl_detectors[-1].name] = (
                tl_detectors[-1].runtime.time * deadline_slack_factor
            )
            resources[tl_detectors[-1].name] = Resources(
                resource_vector={
                    Resource("GPU", _id="any"): 1,
                    Resource("CPU", _id="any"): 1,
                }
            )
            tl_object_localization.append(
                Job(
                    name=f"tl_obj_localization_{i}",
                    runtime=EventTime(10000, EventTime.Unit.US),
                    pipelined=True,
                )
            )
            deadlines[tl_object_localization[-1].name] = (
                tl_object_localization[-1].runtime.time * deadline_slack_factor
            )
            resources[tl_object_localization[-1].name] = Resources(
                resource_vector={Resource("CPU", _id="any"): 1}
            )

        prediction = Job(
            name="prediction",
            runtime=EventTime(30000, EventTime.Unit.US),
            pipelined=False,
        )
        deadlines[prediction.name] = prediction.runtime.time * deadline_slack_factor
        resources[prediction.name] = Resources(
            resource_vector={
                Resource("GPU", _id="any"): 1,
                Resource("CPU", _id="any"): 1,
            }
        )
        planning = Job(
            name="planning",
            runtime=EventTime(50000, EventTime.Unit.US),
            pipelined=False,
        )
        deadlines[planning.name] = planning.runtime.time * deadline_slack_factor
        resources[planning.name] = Resources(
            resource_vector={Resource("CPU", _id="any"): 8}
        )
        control = Job(
            name="control", runtime=EventTime(1000, EventTime.Unit.US), pipelined=False
        )
        deadlines[control.name] = control.runtime.time * periodic_deadline_slack_factor
        resources[control.name] = Resources(
            resource_vector={Resource("CPU", _id="any"): 1}
        )

        job_graph = JobGraph()

        job_graph.add_job(gnss, [localization])
        job_graph.add_job(imu, [localization])
        job_graph.add_job(
            localization,
            [planning, control] + object_localization + tl_object_localization,
        )
        # Add camera, lidars, and their perception operators.
        for i in range(num_perception_sensors):
            job_graph.add_job(
                cameras[i], [detectors[i], lane_detectors[i], trackers[i]]
            )
            if i < num_traffic_light_cameras:
                job_graph.add_job(
                    lidars[i], [object_localization[i], tl_object_localization[i]]
                )
            else:
                job_graph.add_job(lidars[i], [object_localization[i]])
            job_graph.add_job(detectors[i], [trackers[i]])
            job_graph.add_job(trackers[i], [object_localization[i]])
            job_graph.add_job(lane_detectors[i], [planning])
            job_graph.add_job(object_localization[i], [prediction])

        # Add traffic light camera operators.
        for i in range(num_traffic_light_cameras):
            job_graph.add_job(tl_cameras[i], [tl_detectors[i]])
            job_graph.add_job(tl_detectors[i], [tl_object_localization[i]])
            job_graph.add_job(tl_object_localization[i], [prediction])

        job_graph.add_job(prediction, [planning])
        job_graph.add_job(planning, [control])
        jobs = (
            [gnss, imu, localization, prediction, planning, control]
            + cameras
            + lidars
            + detectors
            + lane_detectors
            + trackers
            + object_localization
            + tl_cameras
            + tl_detectors
            + tl_object_localization
        )
        return jobs, job_graph, deadlines, resources

    def create_tasks(
        self,
        max_timestamp: int,
        timestamp_difference: EventTime,
        runtime_variance: int,
        deadline_variance: Tuple[int, int],
        deadlines: Mapping[str, int],
        resources: Mapping[str, Sequence[Resources]],
        use_end_to_end_deadlines: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        def get_deadline(job, timestamp):
            if use_end_to_end_deadlines:
                job_name = f"e2e-{timestamp}"
                if job_name not in deadlines:
                    deadlines[job_name] = fuzz_time(
                        self._job_graph.completion_time, deadline_variance
                    )
                return deadlines[job_name]
            else:
                return fuzz_time(
                    EventTime(int(deadlines[job.name]), EventTime.Unit.US),
                    deadline_variance,
                )

        tasks = {}
        sensor_release_time = EventTime.zero()
        for timestamp in range(max_timestamp + 1):
            for job in self._job_graph:
                # All times are in microseconds.
                if self._job_graph.is_source(job) or use_end_to_end_deadlines:
                    # Source jobs are released at a pre-specified interval.
                    # If we use end-to-end deadlines, then all tasks must have same
                    # deadline.
                    release_time = sensor_release_time
                    deadline = sensor_release_time + get_deadline(job, timestamp)
                else:
                    # Non-Source jobs are released as soon as all of their dependencies
                    # are estimated to be satisfied.
                    max_estimated_parent_completion_time = max(
                        tasks[(parent.name, timestamp)].release_time + parent.runtime
                        for parent in self._job_graph.get_parents(job)
                    )
                    release_time = (
                        max_estimated_parent_completion_time
                        if job.pipelined or timestamp == 0
                        else max(
                            max_estimated_parent_completion_time,
                            tasks[(job.name, timestamp - 1)].release_time + job.runtime,
                        )
                    )
                    deadline = release_time + get_deadline(job, timestamp)

                # Create the task.
                task = Task(
                    job.name,
                    job,
                    resource_requirements=resources[job.name],
                    runtime=fuzz_time(job.runtime, (0, runtime_variance)),
                    deadline=deadline,
                    timestamp=timestamp,
                    release_time=release_time,
                    _logger=logger,
                )
                tasks[(job.name, timestamp)] = task

            sensor_release_time += timestamp_difference
        self._tasks = tasks.values()

    def get_jobs(self) -> Sequence[Job]:
        """Retrieve the set of `Job`s loaded.

        Returns:
            The set of `Job`s loaded.
        """
        return self._jobs

    def get_job_graph(self) -> JobGraph:
        """Retrieve the constructed `JobGraph`.

        Returns:
            The `JobGraph` constructed by the TaskLoader.
        """
        return self._job_graph

    def get_tasks(self) -> Sequence[Task]:
        """Retrieve the set of `Task`s loaded by the TaskLoader.

        Returns:
            The set of `Task`s loaded by the TaskLoader.
        """
        return self._tasks

    def get_task_graph(self) -> TaskGraph:
        """Retrieve the `TaskGraph` constructed by the TaskLoader.

        Returns:
            The `TaskGraph` constructed by the TaskLoader.
        """
        return self._task_graph
