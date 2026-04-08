"""
Task 1 - EASY
=============
1 patient, 2 hospitals, 1 ambulance.
Plenty of capacity. Agent just needs to pick the nearest hospital.
"""

TASK_CONFIG = {
    "task_id": "task_easy",
    "description": "Single patient, two nearby hospitals, sufficient capacity. Pick the nearest.",
    "seed": 1,
    "hospitals": [
        {"id": 1, "location": (20, 20), "total_beds": 10, "total_icu": 5},
        {"id": 2, "location": (80, 80), "total_beds": 10, "total_icu": 5},
    ],
    "ambulances": [
        {"id": 1, "location": (30, 30)},
    ],
    "patients": [
        {"id": 1, "location": (25, 25), "severity": "P2"},
    ],
}

GRADER = {
    "optimal_hospital_id": 1,     # Hospital 1 is clearly closer
    "optimal_ambulance_id": 1,
    "min_passing_reward": 0.5,
    "explanation": "Hospital 1 at (20,20) is much closer to patient at (25,25) than Hospital 2 at (80,80).",
}