# llm_world/path_utils.py
from pathlib import Path

class PathManager:
    ROOT = Path(__file__).parent.parent.resolve()

    # Base directory that holds the task folders
    TASK_BASE = ROOT / "GPT-Planner" / "pddl"
    RESULT_BASE = ROOT / "results"


    # Allowed task IDs
    VALID_TASK_IDS = {1, 4, 6, 9, 10, 11}
    MAX_STEPS = {1: 20, 4: 20, 6: 30, 9: 15, 10: 20, 11: 40}


    @classmethod
    def task_dir(cls, task_id: int) -> Path:
        """Return the absolute path to the task directory for a given id.

        Valid ids: 1, 4, 6, 9, 10, 11.
        """
        assert task_id in cls.VALID_TASK_IDS
        return cls.TASK_BASE / f"task{task_id}"

    @classmethod
    def task_path(cls, task_id: int, *parts: str) -> Path:
        """Return an absolute path inside the given task's directory.

        Example: PathManager.task_path(1, 'domain.pddl')
        """
        return cls.task_dir(task_id).joinpath(*parts)
    
    @classmethod
    def results_path(cls, *parts: str) -> Path:
        """Return an absolute path inside the results directory.

        Example: PathManager.results_path("experiments.csv")
        """
        return cls.RESULT_BASE.joinpath(*parts)

if __name__ == "__main__":
    # from llm_world.path_utils import PathManager

    # /.../llm_world/GPT-Planner/pddl/task1
    print("task1_dir", PathManager.task_dir(1))

    # /.../llm_world/GPT-Planner/pddl/task4/domain.pddl
    print("domain4: ", PathManager.task_path(4, "domain.pddl"))

    # Raises ValueError: Invalid task_id 3. Must be one of [1, 4, 6, 9, 10, 11].
    print("bad_example", PathManager.task_dir(3))
