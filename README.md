---
title: Emergency Env
emoji: 🚨
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - healthcare
  - emergency
---

# 🚨 Emergency Resource Allocation — OpenEnv

> AI environment for intelligent emergency dispatch and hospital assignment.

## Problem

Hospitals and ambulance services make critical time-sensitive decisions daily.  
This environment simulates that problem — an AI agent must dispatch ambulances  
and assign patients to the right hospital, fast.

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/ -v

# Run API
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Environment API

| Method | Description |
|--------|-------------|
| `reset()` | Reset to initial state, returns state dict |
| `step(action)` | Takes `{"hospital_id": int, "ambulance_id": int}`, returns `{state, reward, done, info}` |
| `state()` | Returns current observable state |

## HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment (`{"task": "easy"}`) |
| `/step` | POST | Take action (`{"hospital_id": 1, "ambulance_id": 1}`) |
| `/state` | GET | Get current state |
| `/health` | GET | Health check |
| `/tasks` | GET | List all tasks |

## Tasks

| Task | Difficulty | Patients | Description |
|------|------------|----------|-------------|
| easy | 🟢 | 1 | Pick nearest hospital |
| medium | 🟡 | 1 | Balance distance + ICU availability |
| hard | 🔴 | 3 | Triage + priority scheduling |

## Reward Function

```
reward = 1 - (0.5 * delay_ratio)   # if within acceptable delay
reward = -0.5 * (delay_ratio - 1)  # if exceeded acceptable delay
-1.0 for invalid actions (wrong hospital, busy ambulance, full ICU)
```

## Patient Severity

| Severity | Max Delay | Needs ICU |
|----------|-----------|-----------|
| P1 (Critical) | 5 min | ✅ |
| P2 (Urgent)   | 15 min | ❌ |
| P3 (Stable)   | 30 min | ❌ |

## Docker

```bash
docker build -t emergency-env .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  emergency-env
```

## Project Structure

```
emergency-env/
├── env/
│   ├── __init__.py
│   ├── environment.py   ← Core env (reset, step, state)
│   └── models.py        ← Hospital, Ambulance, Patient
├── tasks/
│   ├── __init__.py
│   ├── easy.py
│   ├── medium.py
│   └── hard.py
├── tests/
│   └── test_environment.py
├── inference.py         ← LLM agent
├── app.py               ← FastAPI server
├── openenv.yaml         ← Hackathon spec
├── Dockerfile
├── requirements.txt
└── README.md
```
