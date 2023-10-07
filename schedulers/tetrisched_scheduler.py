import time
from typing import List, Mapping, Optional, Set, Tuple

import absl  # noqa: F401
import tetrisched_py as tetrisched

from schedulers import BaseScheduler
from utils import EventTime
from workers import Worker, WorkerPool, WorkerPools
from workload import Placement, Placements, Resource, Task, TaskGraph, Workload


class TetriSchedScheduler(BaseScheduler):
    """Implements a STRL-based, DAG-aware formulation for the Tetrisched backend.

    Args:
        preemptive (`bool`): If `True`, the ILP scheduler can preempt the tasks
            that are currently running.
        runtime (`EventTime`): The runtime to return to the Simulator (in us).
            If -1, the scheduler returns the actual runtime.
        enforce_deadlines (`bool`): If True then deadlines must be met or else the
            `schedule()` will return None.
        retract_schedules (`bool`): If the scheduler schedules future tasks, then
            setting this to `True` enables the scheduler to retract prior scheduling
            decisions before they are actually placed on the WorkerPools.
    """

    def __init__(
        self,
        preemptive: bool = False,
        runtime: EventTime = EventTime.invalid(),
        lookahead: EventTime = EventTime.zero(),
        enforce_deadlines: bool = False,
        retract_schedules: bool = False,
        release_taskgraphs: bool = False,
        time_discretization: EventTime = EventTime(1, EventTime.Unit.US),
        _flags: Optional["absl.flags"] = None,
    ):
        if preemptive:
            raise ValueError("TetrischedScheduler does not support preemption.")
        super(TetriSchedScheduler, self).__init__(
            preemptive=preemptive,
            runtime=runtime,
            lookahead=lookahead,
            enforce_deadlines=enforce_deadlines,
            retract_schedules=retract_schedules,
            release_taskgraphs=release_taskgraphs,
            _flags=_flags,
        )
        self._time_discretization = time_discretization.to(EventTime.Unit.US)
        self._scheduler = tetrisched.Scheduler(self._time_discretization.time)

    def schedule(
        self, sim_time: EventTime, workload: Workload, worker_pools: WorkerPools
    ) -> Placements:
        # Retrieve the schedulable tasks from the Workload.
        tasks_to_be_scheduled: List[Task] = workload.get_schedulable_tasks(
            time=sim_time,
            lookahead=self.lookahead,
            preemption=self.preemptive,
            retract_schedules=self.retract_schedules,
            worker_pools=worker_pools,
            policy=self.policy,
            branch_prediction_accuracy=self.branch_prediction_accuracy,
            release_taskgraphs=self.release_taskgraphs,
        )
        self._logger.debug(
            f"[{sim_time.time}] The scheduler received {len(tasks_to_be_scheduled)} "
            f"tasks to be scheduled. These tasks along with their deadlines were: "
            f"{[f'{t.unique_name} ({t.deadline})' for t in tasks_to_be_scheduled]}"
        )

        # Construct the STRL expression.
        scheduler_start_time = time.time()
        placements = []
        if len(tasks_to_be_scheduled) > 0:
            # Construct the partitions from the Workers in the WorkerPool.
            workers, worker_to_worker_pools, partitions = self.construct_partitions(
                worker_pools=worker_pools
            )

            # Construct the STRL expressions for each TaskGraph.
            task_graph_names: Set[TaskGraph] = {
                task.task_graph for task in tasks_to_be_scheduled
            }
            task_strls: Mapping[str, tetrisched.strl.Expression] = {}
            task_graph_strls: List[tetrisched.strl.Expression] = []
            for task_graph_name in task_graph_names:
                # Retrieve the TaskGraph and construct its STRL.
                task_graph = workload.get_task_graph(task_graph_name)
                task_graph_strl = self.construct_task_graph_strl(
                    current_time=sim_time,
                    task_graph=task_graph,
                    partitions=partitions,
                    task_strls=task_strls,
                )
                if task_graph_strl is not None:
                    task_graph_strls.append(task_graph_strl)

            objective_strl = tetrisched.strl.ObjectiveExpression()
            for task_graph_strl in task_graph_strls:
                objective_strl.addChild(task_graph_strl)

            # Register the STRL expression with the scheduler and solve it.
            self._scheduler.registerSTRL(objective_strl, partitions, sim_time.time)
            self._scheduler.schedule()

            # Retrieve the Placements for each task.
            for task in tasks_to_be_scheduled:
                if task.id not in task_strls:
                    self._logger.error(
                        f"[{sim_time.time}] No STRL was generated for "
                        f"Task {task.unique_name}."
                    )
                task_strl = task_strls[task.id]
                task_strl_solution = task_strl.getSolution()
                if task_strl_solution.utility > 0:
                    # The task was placed, retrieve the Partition where the task
                    # was placed.
                    task_placement = task_strl_solution.getPlacement(task.unique_name)
                    worker_index = task_placement.getPartitionAssignments()[0][0]
                    worker_id = workers[worker_index].id
                    task_placement = Placement.create_task_placement(
                        task=task,
                        placement_time=EventTime(
                            task_strl_solution.startTime, EventTime.Unit.US
                        ),
                        worker_id=worker_id,
                        worker_pool_id=worker_to_worker_pools[worker_id],
                        execution_strategy=task.available_execution_strategies[0],
                    )
                    placements.append(task_placement)
                else:
                    # The task was not placed, log the error.
                    self._logger.debug(
                        f"[{sim_time.time}] Task {task.unique_name} was not placed "
                        f"by the Tetrisched scheduler."
                    )
                    placements.append(Placement.create_task_placement(task=task))

        scheduler_end_time = time.time()
        scheduler_runtime = EventTime(
            int((scheduler_end_time - scheduler_start_time) * 1e6), EventTime.Unit.US
        )
        runtime = (
            scheduler_runtime if self.runtime == EventTime.invalid() else self.runtime
        )
        return Placements(
            runtime=runtime, true_runtime=scheduler_runtime, placements=placements
        )

    def construct_partitions(
        self, worker_pools: WorkerPools
    ) -> Tuple[Mapping[int, Worker], Mapping[str, WorkerPool], tetrisched.Partitions]:
        """Partitions the Workers in the WorkerPools into a granular partition set.

        The Partitions are used to reduce the number of variables in the compiled ILP
        model. All the resources in the Partition are expected to belong to an
        equivalence set and are therefore interchangeable.

        Args:
            worker_pools (`WorkerPools`): The WorkerPools to be partitioned.

        Returns:
            A `Partitions` object that contains the partitions.
        """
        partitions = tetrisched.Partitions()
        # TODO (Sukrit): This method constructs a separate partition for all the slots
        # in a Worker. This might not be the best strategy for dealing with heterogenous
        # resources. Fix.
        worker_index = 1
        workers: Mapping[int, Worker] = {}
        worker_to_worker_pool: Mapping[str, WorkerPool] = {}
        for worker_pool in worker_pools.worker_pools:
            for worker in worker_pool.workers:
                # Check that the Worker only has Slot resources.
                for resource, _ in worker.resources.resources:
                    if resource.name != "Slot":
                        raise ValueError(
                            "TetrischedScheduler currently supports Slot resources."
                        )

                # Create a tetrisched Worker.
                tetrisched_worker = tetrisched.Worker(worker_index, worker.name)
                slot_quantity = worker.resources.get_total_quantity(
                    resource=Resource(name="Slot", _id="any")
                )
                partition = tetrisched.Partition()
                partition.addWorker(tetrisched_worker, slot_quantity)
                partitions.addPartition(partition)

                # Add the Partition to the Map.
                workers[partition.id] = worker
                worker_to_worker_pool[worker.id] = worker_pool
                worker_index += 1
        return workers, worker_to_worker_pool, partitions

    def construct_task_strl(
        self,
        current_time: EventTime,
        task: Task,
        partitions: tetrisched.Partitions,
    ) -> tetrisched.strl.Expression:
        """Constructs the STRL expression subtree for a given Task.

        Args:
            current_time (`EventTime`): The current time.
            task (`Task`): The Task for which the STRL expression is to be constructed.
            task_id (`int`): The index of this Task in the Workload.

        Returns:
            A reference to a STRL subtree that encapsulates the entire set of placement
            choices as ChooseExpressions and a MaxExpression that selects the best
            placement choice.
        """
        if len(task.available_execution_strategies) > 1:
            raise NotImplementedError(
                "TetrischedScheduler does not support multiple execution strategies."
            )

        # Check that the Task works only on Slots for now.
        # TODO (Sukrit): We should expand this to general resource usage.
        # But, this works for now.
        execution_strategy = task.available_execution_strategies[0]
        for resource, _ in execution_strategy.resources.resources:
            if resource.name != "Slot":
                raise ValueError(
                    "TetrischedScheduler currently only supports Slot resources."
                )

        # Construct the STRL MAX expression for this Task.
        # This enforces the choice of only one placement for this Task.
        self._logger.debug(
            f"[{current_time.time}] Constructing a STRL expression tree for "
            f"{task.name} (runtime={execution_strategy.runtime}, "
            f"deadline={task.deadline}) with name: {task.unique_name}_placement."
        )
        chooseOneFromSet = tetrisched.strl.MaxExpression(
            f"{task.unique_name}_placement"
        )

        # Construct the STRL ChooseExpressions for this Task.
        # This expression represents a particular placement choice for this Task.
        num_slots_required = execution_strategy.resources.get_total_quantity(
            resource=Resource(name="Slot", _id="any")
        )

        for placement_time in range(
            current_time.to(EventTime.Unit.US).time,
            task.deadline.to(EventTime.Unit.US).time
            - execution_strategy.runtime.to(EventTime.Unit.US).time
            + 1,
            self._time_discretization.time,
        ):
            self._logger.debug(
                f"[{current_time.time}] Generating a Choose Expression "
                f"for {task.unique_name} at time {placement_time} for "
                f"{num_slots_required} slots for {execution_strategy.runtime}."
            )
            # Construct a ChooseExpression for placement at this time.
            # TODO (Sukrit): We just assume for now that all Slots are the same and
            # thus the task can be placed on any Slot. This is not true in general.
            chooseAtTime = tetrisched.strl.ChooseExpression(
                task.unique_name,
                partitions,
                num_slots_required,
                placement_time,
                execution_strategy.runtime.to(EventTime.Unit.US).time,
            )

            # Register this expression with the MAX expression.
            chooseOneFromSet.addChild(chooseAtTime)
        return chooseOneFromSet

    def _construct_task_graph_strl(
        self,
        current_time: EventTime,
        task: Task,
        task_graph: TaskGraph,
        partitions: tetrisched.Partitions,
        task_strls: Mapping[str, tetrisched.strl.Expression],
    ) -> tetrisched.strl.Expression:
        """Constructs the STRL expression subtree for a given TaskGraph starting at
        the specified Task.

        Args:
            current_time (`EventTime`): The current time.
            task (`Task`): The task in the TaskGraph for which the STRL expression is
                to be rooted at.
            task_graph (`TaskGraph`): The TaskGraph for which the STRL expression is
                to be constructed.

        Returns:
            A reference to a STRL subtree that encapsulates the entire set of placement
            choices for all the Tasks in the TaskGraph and enforces ordering amongst
            them.
        """
        # Check if we have already constructed the STRL for this Task, and return
        # the expression if we have.
        if task.id in task_strls:
            return task_strls[task.id]

        # Construct the STRL expression for this Task.
        self._logger.debug(
            f"[{current_time.time}] Constructing the TaskGraph STRL for the "
            f"graph {task_graph.name} rooted at {task.unique_name}."
        )
        task_expression = self.construct_task_strl(current_time, task, partitions)
        task_strls[task.id] = task_expression

        # Retrieve the STRL expressions for all the children of this Task.
        child_expressions = []
        for child in task_graph.get_children(task):
            self._logger.debug(
                f"[{current_time.time}] Constructing the STRL for {child.unique_name} "
                f"while creating the STRL for TaskGraph {task_graph.name} rooted at "
                f"{task.unique_name}."
            )
            child_expressions.append(
                self._construct_task_graph_strl(
                    current_time, child, task_graph, partitions, task_strls
                )
            )

        # If there are no children, return the expression for this Task.
        if len(child_expressions) == 0:
            return task_expression

        # Construct the subtree for the children of this Task.
        if len(child_expressions) > 1:
            # If there are more than one children, then we need to ensure that all
            # of them are placed by collating them under a MinExpression.
            self._logger.debug(
                f"[{current_time.time}] Collating the children of {task.unique_name} "
                f"under a MinExpression {task.unique_name}_children for STRL of the "
                f"TaskGraph {task_graph.name} rooted at {task.unique_name}."
            )
            child_expression = tetrisched.strl.MinExpression(
                f"{task.unique_name}_children"
            )
            for child in child_expressions:
                child_expression.addChild(child)
        else:
            # If there is just one child, then we can just use that subtree.
            child_expression = child_expressions[0]

        # Construct a LessThanExpression to order the two trees.
        self._logger.debug(
            f"[{current_time.time}] Ordering the STRL for {task.unique_name} and its "
            f"children under a LessThanExpression {task.unique_name}_less_than for "
            f"STRL of the TaskGraph {task_graph.name} rooted at {task.unique_name}."
        )
        task_graph_expression = tetrisched.strl.LessThanExpression(
            f"{task.unique_name}_less_than"
        )
        task_graph_expression.addChild(task_expression)
        task_graph_expression.addChild(child_expression)

        return task_graph_expression

    def construct_task_graph_strl(
        self,
        current_time: EventTime,
        task_graph: TaskGraph,
        partitions: tetrisched.Partitions,
        task_strls: Mapping[str, tetrisched.strl.Expression],
    ) -> tetrisched.strl.Expression:
        """Constructs the STRL expression subtree for a given TaskGraph.

        Args:
            current_time (`EventTime`): The time at which the scheduling is occurring.
            task_graph (`TaskGraph`): The TaskGraph for which the STRL expression is
                to be constructed.
            partitions (`Partitions`): The partitions that are available for scheduling.
            task_strls (`Mapping[str, tetrisched.strl.Expression]`): A mapping from Task
                IDs to their STRL expressions. Used for caching.
        """
        # Construct the STRL expression for all the roots of the TaskGraph.
        root_task_strls = []
        for root in task_graph.get_source_tasks():
            self._logger.debug(
                f"[{current_time.time}] Constructing the STRL for root "
                f"{root.unique_name} while creating the STRL for "
                f"TaskGraph {task_graph.name}."
            )
            root_task_strls.append(
                self._construct_task_graph_strl(
                    current_time, root, task_graph, partitions, task_strls
                )
            )

        if len(root_task_strls) == 0:
            # No roots, possibly empty TaskGraph, return None.
            return None
        elif len(root_task_strls) == 1:
            # Single root, reduce constraints and just bubble this up.
            return root_task_strls[0]
        else:
            # Construct a MinExpression to order the roots of the TaskGraph.
            min_expression_task_graph = tetrisched.strl.MinExpression(
                f"{task_graph.name}_min_expression"
            )
            for root_task_strl in root_task_strls:
                min_expression_task_graph.addChild(root_task_strl)
            return min_expression_task_graph