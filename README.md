# Macro Placement Challenge Submission

This repository provides a macro placer for the Macro Placement Challenge.



## 1. Docker Option

A pre-built Docker image can also be used to avoid manually building DREAMPlace and setting up the Python environment.

Download the Docker snapshot from:



## 2. Configure `run_json`

Each benchmark needs a JSON file under:

```bash
run_json/
```

For example:

```bash
run_json/ibm01.json
```

For a new case, mainly modify these two fields:

```json
{
    "case_name": "ibm01",
    "benchmark_dir": "/Macro_challenge_2026/macro-place-challenge-2026/external/MacroPlacement/Testcases/ICCAD04/ibm01"
}
```


Other parameters can follow `run_json/ibm01.json`.

---


