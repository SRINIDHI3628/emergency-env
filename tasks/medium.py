"""
Task 2 - MEDIUM
===============
1 critical patient (P1, needs ICU), 3 hospitals.
Nearest hospital has NO ICU available.
Agent must reason about capacity, not just distance.
"""

TASK_CONFIG = {
    "task_id": "task_medium",
    "description": (
        "Critical patient needs ICU. Nearest hospital has no ICU beds. "
        "Agent must balance distance vs. capacity."
    ),
    "seed": 2,
    "hospitals": [
        # Nearest but no ICU available
        {"id": 1, "location": (30, 30), "total_beds": 10, "total_icu": 0},
        # Medium distance, has ICU
        {"id": 2, "location": (50, 50), "total_beds": 8,  "total_icu": 3},
        # Farthest, has ICU but far
        {"id": 3, "location": (90, 90), "total_beds": 15, "total_icu": 5},
    ],
    "ambulances": [
        {"id": 1, "location": (35, 35)},
        {"id": 2, "location": (70, 70)},
    ],
    "patients": [
        {"id": 1, "location": (40, 40), "severity": "P1"},
    ],
}

GRADER = {
    "optimal_hospital_id": 2,
    "optimal_ambulance_id": 1,
    "min_passing_reward": 0.3,
    "explanation": (
        "Hospital 1 has no ICU, so it's invalid for P1 patient. "
        "Hospital 2 is closer than Hospital 3 AND has ICU. "
        "Ambulance 1 is closer to the patient."
    ),
}