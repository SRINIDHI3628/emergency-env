"""
Task 3 - HARD
=============
3 patients (P1, P2, P3), 3 hospitals with tight capacity, 2 ambulances.
Agent must triage correctly — P1 must not be sent to a full ICU.
Ambulance reuse is NOT modelled (each step uses the state post-previous-action).
"""

TASK_CONFIG = {
    "task_id": "task_hard",
    "description": (
        "Three patients (P1 critical, P2 urgent, P3 stable) arrive in sequence. "
        "ICU is very limited. Agent must prioritize correctly and avoid overloading hospitals."
    ),
    "seed": 3,
    "hospitals": [
        {"id": 1, "location": (20, 20), "total_beds": 5, "total_icu": 1},
        {"id": 2, "location": (55, 55), "total_beds": 8, "total_icu": 2},
        {"id": 3, "location": (85, 30), "total_beds": 6, "total_icu": 1},
    ],
    "ambulances": [
        {"id": 1, "location": (10, 10)},
        {"id": 2, "location": (60, 60)},
    ],
    "patients": [
        # P1 — critical, needs ICU
        {"id": 1, "location": (15, 15), "severity": "P1"},
        # P2 — urgent, needs bed
        {"id": 2, "location": (50, 50), "severity": "P2"},
        # P3 — stable, flexible
        {"id": 3, "location": (80, 25), "severity": "P3"},
    ],
}

GRADER = {
    "optimal_sequence": [
        {"hospital_id": 1, "ambulance_id": 1},  # P1 → closest ICU hospital
        {"hospital_id": 2, "ambulance_id": 2},  # P2 → middle hospital
        {"hospital_id": 3, "ambulance_id": 1},  # P3 → nearest remaining
    ],
    "min_passing_total_reward": 0.9,  # sum across 3 steps
    "explanation": (
        "P1 must go to a hospital with ICU. Hospital 1 is closest. "
        "P2 goes to Hospital 2. P3 goes to Hospital 3. "
        "All ambulances must be assigned smartly per proximity."
    ),
}