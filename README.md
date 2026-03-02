# TOM

## Visualize soft alignment matrices

This repo includes `vis_alignment.py` to visualize sDTW / uDTW soft alignment matrices for:

- `gamma = 1.0, 0.1, 0.01, 0.001, 0.0001`

### Windows quick start (no existing local env)

Run from project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_and_run_vis.ps1
```

Output image:

- `soft_alignment_matrices.png`

### Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-vis.txt
python vis_alignment.py
```

### Google Colab (if you don't want local setup)

```python
!pip install torch numpy numba matplotlib
%cd /content
!git clone <your-repo-url>
%cd <repo-folder>
!python vis_alignment.py
```