import math
import random
from typing import Dict, List, Optional, Any

from env.models import Hospital, Ambulance, Patient, Action, Observation, Reward


GRID_SIZE = 100  # 100x100 grid units


def euclidean_distance(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class EmergencyEnv:
    """
    OpenEnv-compatible Emergency Resource Allocation Environment.

    The agent must:
    - Assign patients to hospitals
    - Dispatch the right ambulance
    - Minimize delay and avoid overloading hospitals
    """

    SPEED = 10.0  # grid units per minute

    def __init__(self, config: Dict):
        """
        config keys:
            hospitals (list of dicts)
            ambulances (list of dicts)
            patients   (list of dicts)  ← optional, else random
            seed       (int)            ← optional
        """
        self.config = config
        self.seed = config.get("seed", 42)
        self._rng = random.Random(self.seed)
        self.hospitals: List[Hospital] = []
        self.ambulances: List[Ambulance] = []
        self.patients: List[Patient] = []
        self.current_patient_idx: int = 0
        self.step_count: int = 0
        self.total_score: float = 0.0
        self.done: bool = False
        self.reset()

    # ──────────────────────────────────────────────
    # REQUIRED OpenEnv methods
    # ──────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment to initial state. Returns initial observation."""
        self._rng = random.Random(self.seed)
        self.hospitals = self._build_hospitals()
        self.ambulances = self._build_ambulances()
        self.patients = self._build_patients()
        self.current_patient_idx = 0
        self.step_count = 0
        self.total_score = 0.0
        self.done = False
        return Observation(**self.state())

    def step(self, action: Dict) -> Dict:
        """
        Process one action.

        action = {
            "hospital_id": int,
            "ambulance_id": int
        }

        Returns:
            {
                "observation": Observation,
                "reward": Reward,
                "done": bool,
                "info": dict
            }
        """
        if self.done:
            obs = Observation(**self.state())
            reward = Reward(value=0.0, info={"message": "Episode already done."})
            return {
                "observation": obs,
                "reward": reward,
                "done": True,
                "info": {"message": "Episode already done."},
            }

        patient = self.patients[self.current_patient_idx]
        hospital_id = action.get("hospital_id")
        ambulance_id = action.get("ambulance_id")

        reward_value, info = self._evaluate_action(patient, hospital_id, ambulance_id)

        # Apply valid action
        if info.get("valid"):
            hospital = self._get_hospital(hospital_id)
            ambulance = self._get_ambulance(ambulance_id)
            hospital.admit_patient(patient.needs_icu)
            ambulance.dispatch(hospital.location)

        self.total_score += reward_value
        self.step_count += 1
        self.current_patient_idx += 1

        if self.current_patient_idx >= len(self.patients):
            self.done = True

        obs = Observation(**self.state())
        reward = Reward(value=round(reward_value, 4), info=info)

        return {
            "observation": obs,
            "reward": reward,
            "done": self.done,
            "info": info,
        }

    def state(self) -> Dict:
        """Return current observable state."""
        current_patient = (
            self.patients[self.current_patient_idx].to_dict()
            if self.current_patient_idx < len(self.patients)
            else None
        )
        return {
            "hospitals": [h.to_dict() for h in self.hospitals],
            "ambulances": [a.to_dict() for a in self.ambulances],
            "current_patient": current_patient,
            "patients_remaining": len(self.patients) - self.current_patient_idx,
            "step": self.step_count,
        }

    # ──────────────────────────────────────────────
    # Reward logic
    # ──────────────────────────────────────────────

    def _evaluate_action(self, patient: Patient, hospital_id: int, ambulance_id: int):
        hospital = self._get_hospital(hospital_id)
        ambulance = self._get_ambulance(ambulance_id)

        # --- Validity checks ---
        if hospital is None:
            return -1.0, {"valid": False, "reason": f"Hospital {hospital_id} not found"}
        if ambulance is None:
            return -1.0, {"valid": False, "reason": f"Ambulance {ambulance_id} not found"}
        if not ambulance.available:
            return -0.8, {"valid": False, "reason": f"Ambulance {ambulance_id} is busy"}
        if not hospital.has_capacity(patient.needs_icu):
            resource = "ICU" if patient.needs_icu else "bed"
            return -1.0, {
                "valid": False,
                "reason": f"Hospital {hospital_id} has no available {resource}",
            }

        # --- Delay calculation ---
        amb_to_patient = euclidean_distance(ambulance.location, patient.location)
        patient_to_hosp = euclidean_distance(patient.location, hospital.location)
        total_distance = amb_to_patient + patient_to_hosp
        delay_minutes = total_distance / self.SPEED

        # --- Reward formula ---
        max_delay = patient.max_acceptable_delay
        delay_ratio = min(delay_minutes / max_delay, 2.0)  # cap at 2x

        if delay_ratio <= 1.0:
            # Within acceptable delay
            reward = 1.0 - (0.5 * delay_ratio)   # 1.0 → 0.5
        else:
            # Exceeded acceptable delay
            reward = -0.5 * (delay_ratio - 1.0)   # 0 → negative

        reward = max(round(reward, 4), -1.0)

        info = {
            "valid": True,
            "delay_minutes": round(delay_minutes, 2),
            "max_acceptable_delay": max_delay,
            "delay_ratio": round(delay_ratio, 3),
            "patient_severity": patient.severity,
            "hospital_id": hospital_id,
            "ambulance_id": ambulance_id,
        }
        return reward, info

    # ──────────────────────────────────────────────
    # Builders
    # ──────────────────────────────────────────────

    def _build_hospitals(self) -> List[Hospital]:
        hospitals = []
        for h in self.config["hospitals"]:
            hospitals.append(
                Hospital(
                    id=h["id"],
                    location=tuple(h["location"]),
                    total_beds=h["total_beds"],
                    total_icu=h["total_icu"],
                )
            )
        return hospitals

    def _build_ambulances(self) -> List[Ambulance]:
        ambulances = []
        for a in self.config["ambulances"]:
            ambulances.append(
                Ambulance(id=a["id"], location=tuple(a["location"]))
            )
        return ambulances

    def _build_patients(self) -> List[Patient]:
        if "patients" in self.config:
            return [
                Patient(id=p["id"], location=tuple(p["location"]), severity=p["severity"])
                for p in self.config["patients"]
            ]
        # Random generation fallback
        severities = ["P1", "P2", "P3"]
        n = self.config.get("num_patients", 1)
        patients = []
        for i in range(n):
            patients.append(
                Patient(
                    id=i + 1,
                    location=(
                        round(self._rng.uniform(0, GRID_SIZE), 1),
                        round(self._rng.uniform(0, GRID_SIZE), 1),
                    ),
                    severity=self._rng.choice(severities),
                )
            )
        return patients

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _get_hospital(self, hospital_id: int) -> Optional[Hospital]:
        for h in self.hospitals:
            if h.id == hospital_id:
                return h
        return None

    def _get_ambulance(self, ambulance_id: int) -> Optional[Ambulance]:
        for a in self.ambulances:
            if a.id == ambulance_id:
                return a
        return None