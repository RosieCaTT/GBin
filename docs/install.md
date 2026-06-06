# gbin — installation & testing

There are two environments. Set them up in this order:

1. **CPU test env** — fast, runs the full 81-test suite. Works on Windows or
   Linux. Use it to confirm the code is correct before fighting with CUDA.
2. **GPU production env (WSL2)** — PyTorch CUDA (+ optional RAPIDS) + read
   mappers, for real binning on the GPU.

Paths: on Windows the project is `D:\...\Binning Tools\gbin`; in WSL2 the same
folder is `/mnt/d/Rory/The University of Queensland/PhD Project/Commands/Binning Tools/gbin`
(quote it — it has spaces).

---

## Part 1 — CPU test environment (run all tests)

Dependencies: `numpy loguru pytest torch(cpu) pyfastx pyrodigal pyhmmer`, plus
gbin itself (editable). `pycoverm`, `strobealign`, `minimap2` are **not** needed
for tests (BAM path untested locally; mappers are mocked).

> Important: always run `pytest` and `python -m gbin` from **inside** the `gbin/`
> project directory. Running from the parent folder makes the `gbin/` directory
> shadow the installed `gbin` package and imports break. (The installed `gbin`
> console script works from anywhere.)

### Windows (a venv `.venv-test` already exists one level up, from development)

```powershell
cd "D:\Rory\The University of Queensland\PhD Project\Commands\Binning Tools\gbin"
..\.venv-test\Scripts\python.exe -m pytest tests -q
```

To recreate it from scratch (this time inside `gbin/`):

```powershell
cd "D:\Rory\The University of Queensland\PhD Project\Commands\Binning Tools\gbin"
py -3 -m venv .venv-test
.\.venv-test\Scripts\python.exe -m pip install -U pip
.\.venv-test\Scripts\python.exe -m pip install numpy loguru pytest pyfastx pyrodigal pyhmmer
.\.venv-test\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv-test\Scripts\python.exe -m pip install -e .
.\.venv-test\Scripts\python.exe -m pytest tests -q
# after this, the console script also works:  .\.venv-test\Scripts\gbin.exe --version
```

### Linux / WSL2 (fresh venv)

```bash
cd "/mnt/d/Rory/The University of Queensland/PhD Project/Commands/Binning Tools/gbin"
python3 -m venv .venv-test-linux
source .venv-test-linux/bin/activate
pip install -U pip
pip install numpy loguru pytest pyfastx pyrodigal pyhmmer
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pytest tests -q
```

Expected: `81 passed` in ~1 minute.

---

## Part 2 — GPU production environment (WSL2)

### 2.0 Prerequisites
- Up-to-date NVIDIA Windows driver (provides CUDA to WSL2; a 5070 Ti / Blackwell
  needs a driver supporting CUDA 12.8+). No CUDA toolkit install inside WSL2 is
  required for PyTorch wheels.
- Miniforge/Mambaforge in WSL2: https://github.com/conda-forge/miniforge

### 2.1 Easiest (recommended): single env from `environment.yml`
One **RAPIDS-free** env: PyTorch (GPU) + read mappers (strobealign, minimap2,
samtools) + marker tools (pyrodigal, pyhmmer) + **CheckM2**. gbin's default
`medoid` clusterer needs no RAPIDS, so leaving it out keeps the conda solve fast
and conflict-free (RAPIDS is an optional add-on for `--cluster leiden`, see 2.3).

```bash
cd "/mnt/d/.../Binning Tools/gbin"
# Use mamba/libmamba -- the classic conda solver is very slow on this stack:
mamba env create -f environment.yml      # or: conda env create -f environment.yml
conda activate gbin
# Blackwell GPUs: force the cu128 PyTorch wheel (ships sm_120 kernels)
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
checkm2 database --download              # one-time ~3 GB CheckM2 DB
```

> Slow or stuck solve? The classic conda solver struggles with this many
> bioconda + conda-forge packages. Install mamba, or switch conda to libmamba:
> `conda install -n base conda-libmamba-solver && conda config --set solver libmamba`,
> then retry. A healthy solve here is a few minutes, not tens of minutes.

### 2.2 Manual alternative (no `environment.yml`)
The same env built by hand, if you'd rather pick versions yourself:

```bash
conda create -n gbin python=3.12 -c conda-forge -y
conda activate gbin
# read mappers + marker tools + CheckM2 (bioconda/conda-forge)
mamba install -c bioconda -c conda-forge strobealign minimap2 samtools \
    pyrodigal pyhmmer pyfastx pycoverm checkm2 'tensorflow=*=cpu*' diamond -y
# PyTorch with CUDA 12.8 (Blackwell)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .   # run from the gbin/ dir
checkm2 database --download
```

(Drop `checkm2 'tensorflow=*=cpu*' diamond` if you'd rather keep CheckM2 in a
separate env — see 2.2b.)

### 2.2b CheckM2 — accurate final QC + CheckM2-guided refinement
CheckM2 gives an accurate, ML-based completeness/contamination estimate. gbin can
(a) report it on the final bins (`--checkm2`) and (b) **re-refine** bins with it —
split CheckM2-contaminated bins on the GPU and keep a split only if CheckM2
confirms it improves quality (`--checkm2-refine`).

**Recommended: same env as gbin.** `environment.yml` now installs CheckM2 (with a
**CPU** TensorFlow build) alongside gbin. The only awkward dependency is CheckM2's
TensorFlow (its neural-network model); gbin runs CheckM2 as a subprocess with
`CUDA_VISIBLE_DEVICES=""`, so TF never touches the GPU or fights torch for VRAM —
CheckM2's real bottleneck (DIAMOND) is CPU anyway. After creating the env, just
download the database once:

```bash
conda activate gbin
checkm2 database --download        # one-time ~3 GB DB
gbin bin ... --checkm2-refine      # CheckM2 QC + guided decontamination
gbin bin ... --checkm2             # CheckM2 QC only (no re-splitting)
gbin qc -o gbin_out --refine -i contigs.fna.gz   # add it to an existing output
```

> GPU note: only gbin's own clustering/splitting math runs on the GPU. CheckM2
> itself is CPU-bound (DIAMOND); `--checkm2-refine` adds CheckM2 runs (one per
> `--checkm2-refine-iters` round, plus an initial scoring pass), so it is slower
> than plain `--checkm2`. Tune `--checkm2-refine-min-contamination` /
> `--checkm2-refine-score-weight` to control how aggressively bins are split.

**Fallback: separate env.** If the single env won't solve on your platform (e.g. a
RAPIDS/TensorFlow clash), install CheckM2 on its own and point gbin at it — every
flag above still works, just add `--checkm2-bin`:

```bash
conda create -n checkm2 -c bioconda -c conda-forge checkm2 -y
conda run -n checkm2 checkm2 database --download
gbin bin ... --checkm2-refine --checkm2-bin "$(conda run -n checkm2 which checkm2)"
```

Custom DB location for either setup: `--checkm2-db /path/to/CheckM2_database/`.

### 2.3 Optional: RAPIDS (GPU Leiden clustering)
The default clusterer is `medoid` (pure-torch, robust on any CUDA GPU). RAPIDS is
only needed if you explicitly opt into `--cluster leiden` on large datasets, and
only helps if your GPU is well supported by RAPIDS/numba (on brand-new GPUs like
Blackwell, Leiden may fail and auto-fall-back to medoid — just use medoid there).
Use a RAPIDS release that supports your CUDA/GPU:

```bash
# example — check https://docs.rapids.ai/install for the current command/version
conda install -c rapidsai -c conda-forge -c nvidia cuml cugraph 'cuda-version=12.8' -y
```

### 2.4 Verify the GPU
```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); import torch as t; x=t.randn(1024,1024,device='cuda'); print('matmul ok', (x@x).sum().item()!=0)"
python -c "import cuml, cugraph; print('RAPIDS ok')"   # only if you did 2.3
gbin --version
```
If `matmul ok` prints, the GPU (incl. Blackwell sm_120) works. If you see
"no kernel image is available", your torch is too old for the GPU — reinstall the
cu128 wheel (or a nightly).

### 2.5 Reinstall gbin from scratch (teardown + rebuild)
Use this if you bumped the Python version (the env is now Python 3.12) or the
environment got into a bad state.

```bash
# 1. remove the old envs
conda deactivate
conda env remove -n gbin -y
conda env remove -n checkm2 -y          # only if you made a CheckM2 env

# 2. recreate gbin — RAPIDS-free single env, CheckM2 included (use mamba):
cd "/mnt/d/.../Binning Tools/gbin"
mamba env create -f environment.yml      # or the manual 2.2 recipe
conda activate gbin
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# 3. download CheckM2's DB (it's bundled in the env above)
checkm2 database --download
#    ...or, if you used the separate-env fallback (2.2b) instead:
#    conda create -n checkm2 -c bioconda -c conda-forge checkm2 -y
#    conda run -n checkm2 checkm2 database --download

# 4. sanity check
python -c "import sys; print('python', sys.version.split()[0])"   # 3.12.x
gbin --version
```

To also rebuild the Windows CPU test venv after a Python change, delete
`.venv-test` and follow Part 1 again (gbin runs on Python 3.10–3.13).

---

## Part 3 — Running the tests

From the `gbin/` directory (any of the envs above with pytest):

```bash
pytest tests -q                      # all 81 tests
pytest tests -v                      # verbose, per-test
pytest tests/test_composition.py -v  # one module
pytest tests/test_markers.py -v      # pyrodigal+pyhmmer on a real bundled genome
pytest tests -k "cluster or refine"  # by keyword
```

What the suite covers:

| File | Validates |
|------|-----------|
| `test_composition.py` | GPU k-mer counting vs a plain-Python oracle (N-skipping, contig boundaries, projection) |
| `test_normalize.py` | abundance/TNF/weight normalization vs NumPy reference |
| `test_abundance.py` | aemb/merged TSV parsing + alignment to contigs |
| `test_mapping.py` | reads→coverage wiring (mocked mapper), command builders, manifest parsing |
| `test_vae.py` | VAE forward, training, latent separates genomes, hybrid cannot-link |
| `test_cluster.py` | medoid + kNN-graph + label-prop; end-to-end genome recovery |
| `test_markers.py` | **real-genome** SCG detection, contamination, random-DNA negative control |
| `test_scg.py`/`test_constraints.py`/`test_refine.py` | completeness/contamination, cannot-link pairs, decontamination split |
| `test_pipeline.py` | `gbin bin` CLI end-to-end → pure bins; caching |
| `test_checkm2.py`/`test_checkm2_refine.py` | CheckM2 command/report plumbing, CPU-TF guard, and the guided propose→score→accept/reject refine loop (CheckM2 mocked) |
| `test_utils.py` | refhash, N50, chunking |

The GPU/RAPIDS code paths run on `device='cpu'` in tests, so correctness is
covered; only raw GPU speed/VRAM needs a real card to observe.

---

## Part 4 — End-to-end smoke test on synthetic data

Confirms the whole pipeline produces bins (no real reads needed):

```bash
cd "/mnt/d/.../Binning Tools/gbin"
# 8 genomes, 5 samples of synthetic contigs + aemb-style TSVs
python tests/make_synthetic.py /tmp/gbin_demo --genomes 8 --samples 5

gbin bin -i /tmp/gbin_demo/contigs.fna -a /tmp/gbin_demo/sample*.tsv \
    -o /tmp/gbin_demo_out --device cuda --cluster medoid \
    --epochs 100 --min-bin-size 5000

cat /tmp/gbin_demo_out/bins_info.tsv
ls  /tmp/gbin_demo_out/bins/        # expect ~8 bins, one per genome
```

(Use `--device cpu` if the GPU isn't ready yet.)

---

## Part 5 — Real data (you have FASTA + reads)

```bash
# single sample
gbin bin -i contigs.fna.gz --reads s1_R1.fq.gz,s1_R2.fq.gz -o out --device cuda

# multiple samples (best) — manifest: name<TAB>R1<TAB>R2 per line
gbin bin -i contigs.fna.gz --reads-tsv samples.tsv -o out --device cuda

# long reads
gbin bin -i contigs.fna.gz --reads ont.fq.gz --mapper minimap2 --mapper-preset map-ont -o out --device cuda
```

Outputs in `out/`: `bins/*.fna`, `bins_info.tsv`, `contig_bins.tsv`. Intermediate
features (mapping, composition, abundance, markers, latent) are cached under
`out/cache/`.

---

## Troubleshooting

- **`torch.cuda.is_available()` is False** — WSL2 not seeing the GPU. Update the
  Windows NVIDIA driver; check `nvidia-smi` works inside WSL2.
- **"no kernel image is available for execution"** — torch predates your GPU.
  Reinstall: `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`.
- **`'strobealign' not found on PATH`** — `conda install -c bioconda strobealign`
  (or use `--mapper minimap2`, or pass `-a *.tsv` / `--bamdir`).
- **RAPIDS import fails / version clash with Blackwell** — skip it; run
  `--cluster medoid` (still all-GPU). Add RAPIDS later.
- **Out of VRAM** — lower `--batch-size`, keep `--precision bf16`, or pass
  `--max-gpu-mem <GB>`; for huge assemblies raise `--min-contig-len`.
