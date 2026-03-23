# Light Path Tracer

A small black-hole ray-tracing playground for experimenting with geodesics, lensing, shadow shapes, and asymmetry measurements.

## Setup

Create a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run scripts from the repo root with `python -m`:

```bash
python -m scripts.main
python -m scripts.black_hole_shadow
python -m scripts.cli_image_lens
python -m scripts.gen_asym_data
python -m scripts.plot_asym_run 1
```

## Tests

```bash
python -m unittest discover -s tests -v
```
