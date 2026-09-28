
# Setup Guide

This document explains how to set up a reproducible development environment for the course repository and where to start working.

## Prerequisites
- OS: Linux / macOS / Windows (WSL recommended)
- Python 3.10+ (use the version in `requirements.txt` if pinned)
- Optional: Git for developers or if you want to clone the repo
- Optional: GPU + CUDA for heavy training (see section below)

## Recommended Python environment
1. Create and activate a virtual environment (venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer `conda`:

```bash
conda create -n ai-lect python=3.10
conda activate ai-lect
pip install -r requirements.txt
```
## Quarto (site building)

If you want to build or preview the course website, install Quarto (see quarto.org) or use your OS package manager. Verify Quarto is available:

```bash
quarto --version
```

From the project root (the folder containing `_quarto.yml`) you can build the site:

```bash
quarto render
```

Preview locally with live reload:

```bash
quarto preview
```

## Where to start
- Read this repo's top-level `README.md` for the course overview.
- `cd notebooks/` and run 

~~~
make build-nb
~~~

This creates the jupyter notebooks.

- Open `notebooks/01-einleitung.ipynb` to follow the introductory examples. Notebooks can be opened with Jupyter or in Quarto.
- Run `src/02-perzeptron.py` to see a minimal perceptron implementation.

## Troubleshooting
- If imports fail, ensure the virtual environment is activated and `pip install -r requirements.txt` completed successfully.
- For missing dataset files, check `data/` and `notebooks/output/` for preprocessed artifacts.

## GPU / CUDA notes
If you have a CUDA-capable GPU and want to use `tensorflow` or `torch` with GPU support, install the matching CUDA builds as described in the respective library docs. This repository does not assume GPU by default.
