from tasks.easy import TASK_CONFIG as EASY_CONFIG, GRADER as EASY_GRADER
from tasks.medium import TASK_CONFIG as MEDIUM_CONFIG, GRADER as MEDIUM_GRADER
from tasks.hard import TASK_CONFIG as HARD_CONFIG, GRADER as HARD_GRADER

TASKS = {
    "easy":   {"config": EASY_CONFIG,   "grader": EASY_GRADER},
    "medium": {"config": MEDIUM_CONFIG, "grader": MEDIUM_GRADER},
    "hard":   {"config": HARD_CONFIG,   "grader": HARD_GRADER},
}

__all__ = ["TASKS"]