# Macro Placement Challenge Submission

This repository provides a macro placer for the Macro Placement Challenge.

The submission entry is:

```bash
submissions/examples/place.py
```

Run with:

```bash
uv run evaluate submissions/examples/place.py -b ibm01
```

---

## 1. Required Dependencies

### 1.1 MacroPlacement

MacroPlacement should be placed under `external/`:

```bash
mkdir -p external
git clone https://github.com/TILOS-AI-Institute/MacroPlacement.git external/MacroPlacement
```

After installation, the expected path is:

```bash
external/MacroPlacement
```

The benchmark directory should look like:

```bash
external/MacroPlacement/Testcases/ICCAD04/ibm01
external/MacroPlacement/Testcases/ICCAD04/ibm02
external/MacroPlacement/Testcases/ICCAD04/ibm03
```

### 1.2 DREAMPlace

DREAMPlace should be placed under:

```bash
submissions/examples/icas_placer/DREAMPlace
```

The compiled DREAMPlace package should be available at:

```bash
submissions/examples/icas_placer/DREAMPlace/install/dreamplace
```

If DREAMPlace has not been compiled, run:

```bash
cd submissions/examples/icas_placer/DREAMPlace

rm -rf build
mkdir build
cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=../install \
  -DPYTHON_EXECUTABLE=/usr/bin/python3.8 \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
make install
```

### 1.3 Docker Option

A pre-built Docker image can also be used to avoid manually building DREAMPlace and setting up the Python environment.

Download the Docker snapshot from:

```bash
<your-docker-download-link>

---



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

## 3. Run

Run one benchmark:

```bash
/root/.local/bin/uv run --no-sync evaluate submissions/examples/place.py -b ibm01
```

---



