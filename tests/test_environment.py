"""
tests/test_environment.py
=========================
Run with: python -m pytest tests/ -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import EmergencyEnv
from tasks import TASKS


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    return EmergencyEnv(TASKS["easy"]["config"])

@pytest.fixture
def medium_env():
    return EmergencyEnv(TASKS["medium"]["config"])

@pytest.fixture
def hard_env():
    return EmergencyEnv(TASKS["hard"]["config"])


# ─── Basic env tests ──────────────────────────────────────────────────────────

class TestEnvironmentCore:

    def test_reset_returns_state(self, easy_env):
        observation = easy_env.reset()
        assert hasattr(observation, "hospitals")
        assert hasattr(observation, "ambulances")
        assert hasattr(observation, "current_patient")
        assert observation.current_patient is not None

    def test_state_has_required_keys(self, easy_env):
        state = easy_env.state()
        for h in state["hospitals"]:
            assert "id" in h
            assert "available_beds" in h
            assert "available_icu" in h
            assert "location" in h

    def test_step_returns_required_keys(self, easy_env):
        result = easy_env.step({"hospital_id": 1, "ambulance_id": 1})
        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert "info" in result

    def test_reward_is_float(self, easy_env):
        result = easy_env.step({"hospital_id": 1, "ambulance_id": 1})
        assert isinstance(result["reward"].value, float)

    def test_reward_range(self, easy_env):
        result = easy_env.step({"hospital_id": 1, "ambulance_id": 1})
        assert -1.0 <= result["reward"].value <= 1.0

    def test_done_after_all_patients(self, easy_env):
        # Easy task has 1 patient
        result = easy_env.step({"hospital_id": 1, "ambulance_id": 1})
        assert result["done"] is True

    def test_step_after_done_returns_zero_reward(self, easy_env):
        easy_env.step({"hospital_id": 1, "ambulance_id": 1})
        result = easy_env.step({"hospital_id": 1, "ambulance_id": 1})
        assert result["reward"] == 0.0
        assert result["done"] is True

    def test_reset_restores_capacity(self, easy_env):
        easy_env.step({"hospital_id": 1, "ambulance_id": 1})
        observation_after = easy_env.reset()
        h1 = next(h for h in observation_after.hospitals if h["id"] == 1)
        assert h1["available_beds"] == h1["total_beds"]


# ─── Reward logic tests ────────────────────────────────────────────────────────

class TestRewardLogic:

    def test_invalid_hospital_gives_negative_reward(self, easy_env):
        result = easy_env.step({"hospital_id": 999, "ambulance_id": 1})
        assert result["reward"].value < 0

    def test_invalid_ambulance_gives_negative_reward(self, easy_env):
        result = easy_env.step({"hospital_id": 1, "ambulance_id": 999})
        assert result["reward"].value < 0

    def test_p1_patient_to_no_icu_hospital_penalized(self, medium_env):
        # Hospital 1 in medium task has NO ICU
        result = medium_env.step({"hospital_id": 1, "ambulance_id": 1})
        assert result["reward"].value < 0
        assert result["info"]["valid"] is False

    def test_p1_patient_to_icu_hospital_gets_positive_reward(self, medium_env):
        # Hospital 2 has ICU in medium task
        result = medium_env.step({"hospital_id": 2, "ambulance_id": 1})
        assert result["reward"].value > 0

    def test_correct_easy_decision_gets_high_reward(self, easy_env):
        # Hospital 1 is very close to patient in easy task
        result = easy_env.step({"hospital_id": 1, "ambulance_id": 1})
        assert result["reward"].value >= 0.7

    def test_wrong_far_hospital_gets_lower_reward_than_near(self, easy_env):
        # Hospital 2 is far (80,80) vs Hospital 1 at (20,20) — patient at (25,25)
        # Far hospital should give significantly less reward than near hospital
        result_far = easy_env.step({"hospital_id": 2, "ambulance_id": 1})
        reward_far = result_far["reward"].value

        env2 = EmergencyEnv(TASKS["easy"]["config"])
        result_near = env2.step({"hospital_id": 1, "ambulance_id": 1})
        reward_near = result_near["reward"].value

        assert reward_far < reward_near, (
            f"Far hospital reward ({reward_far}) should be less than near hospital reward ({reward_near})"
        )


# ─── Task-level tests ─────────────────────────────────────────────────────────

class TestTasks:

    def test_easy_task_optimal_path(self, easy_env):
        result = easy_env.step({"hospital_id": 1, "ambulance_id": 1})
        grader = TASKS["easy"]["grader"]
        assert result["reward"] >= grader["min_passing_reward"]

    def test_medium_task_optimal_path(self, medium_env):
        result = medium_env.step({"hospital_id": 2, "ambulance_id": 1})
        grader = TASKS["medium"]["grader"]
        assert result["reward"] >= grader["min_passing_reward"]

    def test_hard_task_full_optimal_sequence(self, hard_env):
        grader = TASKS["hard"]["grader"]
        total_reward = 0.0
        for action in grader["optimal_sequence"]:
            result = hard_env.step(action)
            total_reward += result["reward"]
        assert total_reward >= grader["min_passing_total_reward"]

    def test_hard_task_runs_3_steps(self, hard_env):
        for action in TASKS["hard"]["grader"]["optimal_sequence"]:
            result = hard_env.step(action)
        assert result["done"] is True
        assert hard_env.step_count == 3

    def test_all_tasks_loadable(self):
        for name in ["easy", "medium", "hard"]:
            env = EmergencyEnv(TASKS[name]["config"])
            state = env.reset()
            assert state is not None


# ─── Hospital capacity tests ───────────────────────────────────────────────────

class TestCapacity:

    def test_capacity_reduces_after_admission(self, easy_env):
        state_before = easy_env.state()
        h1_before = next(h for h in state_before["hospitals"] if h["id"] == 1)
        beds_before = h1_before["available_beds"]

        easy_env.step({"hospital_id": 1, "ambulance_id": 1})

        # Easy task has 1 patient — done after step
        # Verify beds reduced (env tracks it internally)
        h1_after = next(h for h in easy_env.hospitals if h.id == 1)
        assert h1_after.available_beds == beds_before - 1

    def test_full_hospital_rejects_patient(self):
        config = {
            "hospitals": [{"id": 1, "location": (10, 10), "total_beds": 1, "total_icu": 0}],
            "ambulances": [{"id": 1, "location": (5, 5)}],
            "patients": [
                {"id": 1, "location": (8, 8), "severity": "P2"},
                {"id": 2, "location": (9, 9), "severity": "P2"},
            ],
        }
        env = EmergencyEnv(config)
        env.step({"hospital_id": 1, "ambulance_id": 1})   # fills the only bed
        result = env.step({"hospital_id": 1, "ambulance_id": 1})  # should be rejected
        assert result["reward"] < 0
        assert result["info"]["valid"] is False