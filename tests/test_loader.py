from data import DataLoader
from workload import Job, Resource, Resources

from tests.test_tasks import __create_default_task


def test_create_jobs():
    """ Tests the __create_jobs method of the DataLoader. """
    jobs = DataLoader._DataLoader__create_jobs([
                                    {"pid": "perception_operator"},
                                    {"pid": "prediction_operator"},
                                    {"pid": "planning_operator"},
                                    ])
    assert len(jobs) == 3, "Incorrect number of Jobs returned."
    assert jobs["perception_operator"].name == "perception_operator",\
        "Incorrect Job returned."


def test_create_resources():
    """ Tests the __create_resources method of the DataLoader. """
    resources = DataLoader._DataLoader__create_resources([
            {
                "name": "perception_operator",
                "resource_requirements": [{"CPU:any": 1}],
            },
            {
                "name": "prediction_operator",
                "resource_requirements": [{"CPU:any": 1, "GPU:any": 1}],
            },
            {
                "name": "planning_operator",
                "resource_requirements": [
                                            {"CPU:any": 2},
                                            {"CPU:any": 1, "GPU:any": 1},
                                         ],
            },
        ])

    assert len(resources) == 3, "Incorrect number of Resources returned."
    assert len(resources['perception_operator']) == 1,\
        "Incorrect number of Resources returned."
    assert len(resources['prediction_operator']) == 1,\
        "Incorrect number of Resources returned."
    assert len(resources['planning_operator']) == 2,\
        "Incorrect number of Resources returned."


def test_create_tasks():
    """ Tests the __create_tasks method of the DataLoader. """
    json_entries = [
        {
            "name": "perception_operator.on_watermark",
            "pid": "perception_operator",
            "args": {
                "timestamp": 1,
            },
            "dur": 100,
        }
    ]
    jobs = {
        "perception_operator": Job(name="perception_operator"),
    }
    resources = {
        "perception_operator.on_watermark": [
            Resources(resource_vector={Resource(name="CPU", _id="any"): 1}),
        ]
    }
    tasks = DataLoader._DataLoader__create_tasks(json_entries, jobs, resources)

    assert len(tasks) == 1, "Incorrect number of Tasks returned."
    assert tasks[0].name == "perception_operator.on_watermark",\
        "Incorrect name returned for the Task."
    assert tasks[0].runtime == 100, "Incorrect runtime returned for the Task."
    assert tasks[0].timestamp == [1], "Incorrect timestamp for the Task."
    assert jobs["perception_operator"] == tasks[0].job,\
        "Incorrect Job returned for the Task."


def test_create_jobgraph():
    """ Tests the construction of a JobGraph by the Loader. """
    jobs = {
        "perception_operator": Job(name="perception_operator"),
        "prediction_operator": Job(name="prediction_operator"),
        "planning_operator": Job(name="planning_operator"),
    }
    edges = [
        ("perception_operator", "prediction_operator"),
        ("perception_operator", "planning_operator"),
        ("prediction_operator", "planning_operator"),
    ]
    job_graph = DataLoader._DataLoader__create_job_graph(jobs, edges)

    assert len(job_graph) == 3, "Incorrect length for JobGraph."
    assert len(job_graph.get_children(jobs["perception_operator"])) == 2,\
        "Incorrect length of children for Job."
    assert len(job_graph.get_children(jobs["prediction_operator"])) == 1,\
        "Incorrect length of children for Job."
    assert len(job_graph.get_children(jobs["planning_operator"])) == 0,\
        "Incorrect length of children for Job."


def test_create_taskgraph():
    """ Tests the construction of a TaskGraph by the Loader. """
    # Create the JobGraph first.
    jobs = {
        "perception_operator": Job(name="perception_operator"),
        "prediction_operator": Job(name="prediction_operator"),
        "planning_operator": Job(name="planning_operator"),
    }
    edges = [
        ("perception_operator", "prediction_operator"),
        ("perception_operator", "planning_operator"),
        ("prediction_operator", "planning_operator"),
    ]
    job_graph = DataLoader._DataLoader__create_job_graph(jobs, edges)

    # Create a list of Tasks to be put into a graph.
    tasks = [
        __create_default_task(job=jobs["perception_operator"], timestamp=[1]),
        __create_default_task(job=jobs["perception_operator"], timestamp=[2]),
        __create_default_task(job=jobs["perception_operator"], timestamp=[3]),
        __create_default_task(job=jobs["prediction_operator"], timestamp=[1]),
        __create_default_task(job=jobs["prediction_operator"], timestamp=[2]),
        __create_default_task(job=jobs["prediction_operator"], timestamp=[3]),
        __create_default_task(job=jobs["planning_operator"], timestamp=[1]),
        __create_default_task(job=jobs["planning_operator"], timestamp=[2]),
        __create_default_task(job=jobs["planning_operator"], timestamp=[3]),
    ]

    # Create a TaskGraph using the jobs and the list of tasks.
    task_graph = DataLoader._DataLoader__create_task_graph(tasks, job_graph)
    assert len(task_graph) == len(tasks), "Incorrect length of TaskGraph."

    # Check the parent-child relationships.
    parents_perception_task_1 = task_graph.get_parents(tasks[0])
    assert len(parents_perception_task_1) == 0, "Incorrect length for parents."

    parents_perception_task_3 = task_graph.get_parents(tasks[2])
    assert len(parents_perception_task_3) == 1, "Incorrect length for parents."
    assert set(parents_perception_task_3) == {tasks[1]},\
        "Incorrect parents for the task."

    parents_prediction_task = task_graph.get_parents(tasks[5])
    assert len(parents_prediction_task) == 2, "Incorrect length for parents."
    assert set(parents_prediction_task) == {tasks[2], tasks[4]},\
        "Incorrect parents for the task."

    parents_planning_task = task_graph.get_parents(tasks[8])
    assert len(parents_planning_task) == 3, "Incorrect length for parents."
    assert set(parents_planning_task) == {tasks[2], tasks[5], tasks[7]},\
        "Incorrect parents for the task."
