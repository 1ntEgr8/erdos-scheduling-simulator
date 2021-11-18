import uuid
from typing import Optional, Sequence, Type

from workload import Resources, Task
from schedulers import BaseScheduler


class Worker(object):
    """A `Worker` is a virtual abstraction over a single machine.

    A `Worker` "owns" a certain set of `Resource`s, and contains a set of
    runtime threads that together execute a set of `Task`s assigned to it.

    Args:
        name (`str`): A name assigned to the particular instance of the Worker.
        resources (`Resource`): The set of `Resource`s owned by this worker.
        id (`UUID`): The ID of this particular Worker.
        num_threads (`int`): The number of threads in this Worker.
    """
    def __init__(self, name: str, resources: Resources, num_threads: int):
        self._name = name
        self._id = uuid.uuid4()
        self._resources = resources
        self._num_threads = num_threads
        self._placed_tasks = {}  # Tasks along with their execution status.

    def place_task(self, task: Task):
        """Places the task on this `Worker`.

        The caller must check that the `Worker` can accomodate this task by
        invoking `can_accomodate_task`.

        Args:
            task (`Task`): The task to be placed in this `WorkerPool`.
        """
        self._resources.allocate_multiple(task.resource_requirements)
        self._placed_tasks[task] = task.state

    def can_accomodate_task(self, task: Task) -> bool:
        """Checks if this `Worker` can accomodate the given `Task` based on
        its resource availability.

        Args:
            task (`Task`): The task to be placed on this `Worker`.

        Returns:
            `True` if the task can be placed, `False` otherwise.
        """
        return self._resources > task.resource_requirements

    def get_placed_tasks(self) -> Sequence[Task]:
        """Retrieves the `Task` that is currently placed on this `Worker`.

        Returns:
            A sequence of `Task`s that are currently placed on this `Worker`.
        """
        placed_tasks = []
        for task, _ in self._placed_tasks.items():
            placed_tasks.append(task)
        return placed_tasks

    def step(self, current_time: float, step_size: float = 1.0) -> \
            Sequence[Task]:
        """Steps all the tasks of this `Worker` by the given `step_size`.

        Args:
            current_time (`float`): The current time of the simulator loop.
            step_size (`float`): The amount of time for which to step the
                tasks.

        Returns:
            A set of tasks that have been completed.
        """
        completed_tasks = []
        # Invoke the step() method on all the tasks.
        for task in self._placed_tasks:
            if task.step(current_time, step_size):
                completed_tasks.append(task)
                # Resave the task as now completed / evicted.
                self._placed_tasks[task] = task.state
        return completed_tasks

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return str(self._id)

    @property
    def resources(self):
        return self._resources

    @property
    def num_threads(self):
        return self._num_threads

    def __str__(self):
        return "Worker(name={}, id={}, resources={}, num_threads={})".format(
                self.name, self.id, self.resources, self.num_threads)

    def __repr__(self):
        return str(self)


class WorkerPool(object):
    """A `WorkerPool` acts as a collection of `Worker`s, and may perform its
    own scheduling (for a two-level scheduling architecture) across its
    workers.

    A Scheduler only works at the level of a WorkerPool, and a single Worker
    must be abstracted into a WorkerPool before it can be used by the
    Scheduler.

    However, to schedule a task across its `Worker`s, developers can provide
    a `Scheduler` instance that is in charge of scheduling the placed task.

    Args:
        name (`str`): A name assigned to this WorkerPool.
        id (`UUID`): The ID of the particular WorkerPool.
    """
    def __init__(self, name: str, workers: Optional[Sequence[Worker]] = [],
                 scheduler: Optional[Type[BaseScheduler]] = None):
        self._name = name
        self._workers = {worker.id: worker for worker in workers}
        self._scheduler = scheduler
        self._id = uuid.uuid4()

    def add_workers(self, workers: Sequence[Worker]):
        """Adds the given set of `Worker`s to this `WorkerPool`.

        Args:
            workers (`Sequence[Worker]`): The workers to be added.
        """
        for worker in workers:
            if worker.id in self._workers:
                print("Skipping adding Worker with ID: {} because it already\
                        exists.".format(worker.id))
            else:
                self._workers[worker.id] = worker

    def place_task(self, task: Task):
        """Places the task on this `WorkerPool`.

        The caller must ensure that the `WorkerPool` has enough resources to
        execute this task before invoking this method. Further, the
        `WorkerPool` is in charge of executing the task across its workers.

        Args:
            task (`Task`): The task to be placed in this `WorkerPool`.

        Raises:
            `ValueError` if the task could not be placed due to insufficient
            resources.
        """
        placement = None
        if self._scheduler is not None:
            # If a scheduler was provided, get a task placement from it.
            runtime, placement = self._scheduler.schedule(
                                        [task],  # Only this task is available.
                                        None,    # No task graph.
                                        self._workers)
            # Add the runtime to the task start time.
            task._start_time += runtime
        else:
            # If there was no scheduler, find the first worker that can
            # accomodate the task given its resource requirements.
            for _id, _worker in self._workers.items():
                if _worker.can_accomodate_task(task):
                    placement = _id
                    break

        if placement is None:
            raise ValueError("The task ({}) could not be placed.".format(task))
        else:
            self._workers[placement].place_task(task)

    def get_placed_tasks(self) -> Sequence[Task]:
        """Retrieves the `Task`s that are currently placed on this `WorkerPool`.

        Returns:
            A sequence of `Task`s that are currently placed on this `Worker`.
        """
        placed_tasks = []
        for _, _worker in self._workers.items():
            placed_tasks.extend(_worker.get_placed_tasks())
        return placed_tasks

    def step(self, current_time: float, step_size: float = 1.0) ->\
            Sequence[Task]:
        """Steps all the tasks of this `WorkerPool` by the given `step_size`.

        Args:
            current_time (`float`): The current time of the simulator loop.
            step_size (`float`): The amount of time for which to step the
                workers.

        Returns:
            The set of tasks that have finished execution.
        """
        completed_tasks = []
        # Invoke the step() method on all the workers.
        for _, worker in self._workers.items():
            completed_tasks.extend(worker.step(current_time, step_size))
        return completed_tasks

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return str(self._id)

    @property
    def workers(self):
        return self._workers

    def __str__(self):
        return "WorkerPool(name={}, id={})".format(self.name, self.id)

    def __repr__(self):
        return str(self)
