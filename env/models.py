from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
from pydantic import BaseModel


# OpenEnv Spec Models
class Action(BaseModel):
    hospital_id: int
    ambulance_id: int


class Observation(BaseModel):
    hospitals: List[Dict[str, Any]]
    ambulances: List[Dict[str, Any]]
    current_patient: Optional[Dict[str, Any]]
    patients_remaining: int
    step: int


class Reward(BaseModel):
    value: float
    info: Dict[str, Any]


@dataclass
class Hospital:
    id: int
    location: Tuple[float, float]
    total_beds: int
    total_icu: int
    available_beds: int = 0
    available_icu: int = 0

    def __post_init__(self):
        if self.available_beds == 0:
            self.available_beds = self.total_beds
        if self.available_icu == 0:
            self.available_icu = self.total_icu

    def has_capacity(self, needs_icu: bool = False) -> bool:
        if needs_icu:
            return self.available_icu > 0
        return self.available_beds > 0

    def admit_patient(self, needs_icu: bool = False):
        if needs_icu:
            self.available_icu -= 1
        else:
            self.available_beds -= 1

    def to_dict(self):
        return {
            "id": self.id,
            "location": self.location,
            "available_beds": self.available_beds,
            "available_icu": self.available_icu,
            "total_beds": self.total_beds,
            "total_icu": self.total_icu,
        }


@dataclass
class Ambulance:
    id: int
    location: Tuple[float, float]
    available: bool = True

    def dispatch(self, new_location: Tuple[float, float]):
        self.available = False
        self.location = new_location

    def to_dict(self):
        return {
            "id": self.id,
            "location": self.location,
            "available": self.available,
        }


@dataclass
class Patient:
    id: int
    location: Tuple[float, float]
    severity: str  # "P1" (critical), "P2" (urgent), "P3" (stable)

    @property
    def needs_icu(self) -> bool:
        return self.severity == "P1"

    @property
    def max_acceptable_delay(self) -> float:
        return {"P1": 5.0, "P2": 15.0, "P3": 30.0}.get(self.severity, 15.0)

    def to_dict(self):
        return {
            "id": self.id,
            "location": self.location,
            "severity": self.severity,
            "needs_icu": self.needs_icu,
        }