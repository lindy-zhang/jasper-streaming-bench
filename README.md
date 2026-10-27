# Jasper streaming benchmark

A harness for running [Jasper](https://github.com/zikunw/jasper) (GPU Vamana index, VLDB '26) under
streaming insert/delete workloads, including **CANDOR-Bench's own datasets and runbooks**. It measures
insert throughput, search QPS, recall over time, GPU utilization and GPU memory.

It answers the questions from the Oct 22 meeting:
- How does Jasper perform on the CANDOR streaming workloads?
- How do batch size and insert/delete patterns change that?
- Does a small batch leave the GPU underused? What is the bottleneck?

| File | Purpose |
| --- | --- |
| `stream_bench.py` | The benchmark: build, stream batches, evaluate, write a CSV. Replays CANDOR runbooks or built-in patterns |
| `run_candor.sh` | Runs Jasper on a list of CANDOR runbooks + datasets |
| `run_sweep.sh` | Built-in patterns: every batch size x pattern for one dataset |
| `plot_runs.py` | Overlays runs: insert speed, GPU utilization, recall, memory over the stream |
| `plot_sweep.py` | Summary + plots for a `run_sweep.sh` batch-size sweep |
| `convert_vecs.py` | Converts `.fvecs`/`.bvecs` (TEXMEX SIFT, GIST) to Jasper's `.bin` format |
| `tests/` | CPU-only mock of Jasper, scaled-down CANDOR runbooks, and a test of the harness's bookkeeping |

---

## 0. Get GPU access

Jasper needs an NVIDIA GPU with compute capability 8.0+ (A100, A40, L40, H100/H200, RTX 30/40). A laptop
cannot run it. The lab servers:

```bash
ssh <csail-username>@fatality.csail.mit.edu     # A100s
ssh <csail-username>@nezha.csail.mit.edu        # the machine Srinivas uses
```

- Ask Dr. Chen to add your account to both machines.
- If `ssh` hangs from off campus, connect to the MIT VPN first, or ask how the group connects.
- Ask where shared datasets and scratch space live, so you don't fill your home directory.

## 1. Be a good neighbor on a shared GPU server

Other people's jobs run on these GPUs. Check before you start, and pick a GPU nobody is using.

```bash
nvidia-smi                       # every GPU: memory used, utilization, and the processes on it
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv    # GPU processes
ps -o user,pid,etime,args -p <pid>                                       # whose process is it?
nvtop                            # live view, if installed (q to quit)
htop                             # CPU and RAM (q to quit)
who                              # who else is logged in
```

Then:

- **Use one free GPU only:** `export CUDA_VISIBLE_DEVICES=2` (the index of a GPU with ~0% utilization and
  little memory used). Your program then sees only that GPU, as "device 0". The harness maps
  utilization sampling to the right physical GPU automatically.
- **Run inside `tmux`** so a dropped connection doesn't kill your job: `tmux new -s jasper`, run things,
  detach with `Ctrl-b d`, reattach later with `tmux attach -t jasper`.
- **Profile only when the GPU is idle.** `ncu` (section 9) serializes kernels and slows the whole GPU,
  so check `nvidia-smi` first, or ask in Slack.
- **Never kill someone else's process.** If every GPU is busy, ask.

## 2. Set up an environment

```bash
conda create -n jasper python=3.11 -y
conda activate jasper
pip install torch                     # pick the CUDA build that matches `nvcc --version` at pytorch.org if needed
pip install apache-tvm-ffi numpy matplotlib nvidia-ml-py pyyaml gdown
conda install -c conda-forge cmake -y # only if the system cmake is old
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU's name. On some servers you first need `module load cuda`.

## 3. Build Jasper

```bash
git clone https://github.com/zikunw/jasper.git
cd jasper
cmake -B build -DJASPER_BUILD_FFI=ON -DJASPER_BUILD_CMD=ON
cmake --build build -j
cmake --install build
pip install -e python/
python -c "import jasper; print('jasper ok')"
cd ..
```

If CMake cannot find **Thrust**, point it at the copy inside the CUDA toolkit:

```bash
find "$CUDA_HOME" -name thrust-config.cmake       # e.g. .../lib64/cmake/thrust/thrust-config.cmake
cmake -B build -DJASPER_BUILD_FFI=ON -DJASPER_BUILD_CMD=ON -DThrust_DIR=<that folder>
```

Written against Jasper commit `a0173f4` (Oct 20, 2026). The repo changes often; if an API call fails,
compare against `python/jasper/__init__.py`.

## 4. Check the harness (no GPU needed)

```bash
bash tests/test_harness.sh
```

Runs every built-in pattern and four scaled-down CANDOR runbooks against an exact CPU mock of Jasper.
Every recall must be 1.0 and live counts must match; anything else means the bookkeeping is broken.

## 5. CANDOR datasets and runbooks

### Get them

Put this folder, CANDOR-Bench and the data side by side:

```bash
git clone https://github.com/CGCL-codes/CANDOR-Bench.git      # for its runbooks/ folder
mkdir -p data/candor && cd data/candor
gdown --folder https://drive.google.com/drive/folders/1PngXRH9jnN86T8RNiU-QyGqOillfQE_p -O sift        # ~500 MB
gdown --folder https://drive.google.com/drive/folders/1ZkWOrja-0A6C9yh3ysFoCP6w5u7oWjQx -O openimages  # ~2 GB
gdown --folder https://drive.google.com/drive/folders/1m06VVmXmklHr7QZzdz6w8EtYmuRGIl9s -O glove       # ~500 MB
gdown --folder https://drive.google.com/drive/folders/1TnLNJNVqyFrEzKGfQVdvUC8Al-tmjVg0 -O msong       # ~1.6 GB
gdown --folder https://drive.google.com/drive/folders/1Hp6SI8YOFPdWbmC1a4_-1dZWxZH3CHMS -O coco        # ~300 MB
gdown --folder https://drive.google.com/drive/folders/1gNK1n-do-7d5N-Z1tuAoXe5Xq3I8fZIH -O sun         # ~160 MB
cd ../..
```

(Links from CANDOR's `datasets/download_utils.py`. Google Drive sometimes rate-limits; retry later or
download in the browser.) Every CANDOR file is `[n][dim]` + float32, which Jasper reads directly, so
always use `--dtype f32`. No conversion needed.

| Folder | Base file | Query file | Size | Used for |
| --- | --- | --- | --- | --- |
| `sift` | `data_1000000_128` | `queries_10000_128` | 1M x 128 | batch size, event rate, deletion runbooks |
| `openimages` | `base.1M.fbin` | `query.public.10K.fbin` | 1M x 512 | concept drift (data is stored in time order) |
| `glove` | `base.fbin` | `query.fbin` | 1.19M x 100 | general experiment |
| `msong` | `base.fbin` | `query.fbin` | 992K x 420 | general experiment |
| `coco` | `base.fbin` | `query.fbin` | 100K x 768 | multi-modal |
| `sun` | `base.fbin` | `query.fbin` | 79K x 512 | general |

### First run (a few minutes)

The first 20 batches of CANDOR's SIFT batch-size runbook:

```bash
python3 stream_bench.py \
  --base data/candor/sift/data_1000000_128 --queries data/candor/sift/queries_10000_128 \
  --runbook CANDOR-Bench/runbooks/batch_sizes/batch_10000.yaml --runbook-dataset sift \
  --max-batches 20 --out results/first_run.csv
```

Expect recall@10 above 0.9 at beam width 64; much lower means a problem.

### Full set

Edit the paths at the top of `run_candor.sh` if needed, then:

```bash
bash run_candor.sh                 # everything (hours)
bash run_candor.sh batch           # only experiments with "batch" in the name
python3 plot_runs.py "results/candor/sift_batch_[0-9]*.csv" --name batch_sizes
python3 plot_runs.py "results/candor/sift_*delete*.csv"     --name deletes
```

| Experiment (`run_candor.sh`) | CANDOR runbook | Question for Jasper |
| --- | --- | --- |
| `sift_batch_{500,2500,10000,50000}` | `batch_sizes/` | How do insert speed and GPU use change with batch size? |
| `sift_batch_delete_{0.1,0.5}` | `deletion_patterns/` | Each batch deletes its own last 10% / 50% |
| `sift_bulk_delete_{0.1,0.5}` | `bulk_deletion/` | Insert everything, then delete the oldest 10% / 50% at once |
| `openimages_concept_drift` | `concept_drift/` | Does recall hold as the data drifts? |
| `glove_general`, `msong_general` | `general_experiment/` | Anisotropic and high-dimensional data |

### How a runbook maps to Jasper

| Runbook operation | What the harness does |
| --- | --- |
| `initial` (rows start..end) | `Graph.build` on those rows |
| `batch_insert` | `append` each batch of `batchSize` rows, in file order |
| `batch_insert_delete` | after each batch, delete the last `deletion_percentage` of that batch (as CANDOR does) |
| `delete` (rows start..end) | one bulk `mark_deleted`, recorded as its own `bulk_delete` row |
| `search` | evaluate recall and QPS on fresh ground truth |
| `startHPC`, `endHPC`, `waitPending` | ignored |
| `enableScenario`, `stress_test`, `auto_batch_delete` | refused (noise/drop injection and stress tests aren't reproduced) |

Differences from CANDOR itself:
- **No arrival clock or dropping queue.** Batches are applied as fast as Jasper can go. Instead, each
  batch records `event_rate` and `keeps_up` (whether Jasper's insert speed beats the runbook's rate).
- **Deletes need `consolidate`** to repair the graph. `--consolidate-every` sets how often (default in
  `run_candor.sh`: every 10 batches, and after a bulk delete).
- **Evaluation runs about 10 times per stream** (`--eval-every` to change), not every 1% like CANDOR.
- One GPU, vectors stored as float16.

## 6. Built-in patterns and batch-size sweep

Also works on any dataset, without a runbook:

```bash
bash run_sweep.sh                         # edit dataset paths at the top first
python3 plot_sweep.py results/sift1m_*.csv
```

| `--pattern` | What happens each batch |
| --- | --- |
| `insert_only` | Insert only; the index grows |
| `sliding_window` | Insert, then delete the oldest vectors so the live count stays at `--window` |
| `random_delete` | Insert, then delete `--delete-ratio` x batch size random live vectors |
| `batch_tail_delete` | Insert, then delete the last `--delete-ratio` of that batch (CANDOR-style) |

`--order shuffle` inserts in random order. To use TEXMEX `.fvecs` files (e.g. GIST), convert first:
`python3 convert_vecs.py gist_base.fvecs data/gist_base.bin`.

## 7. Output columns

| Column | Meaning |
| --- | --- |
| `batch`, `step` | Batch number (0 = build) and what happened: `build`, `batch_insert`, `bulk_delete`, ... |
| `batch_size`, `append_s`, `append_vps` | Batch size, time to insert it, vectors per second |
| `build_s`, `build_vps` | Initial build time and speed |
| `event_rate`, `keeps_up` | Runbook arrival rate, and whether `append_vps` beats it |
| `n_deleted`, `delete_s`, `consolidate_s` | Deletes this step, time to mark them, consolidate time |
| `n_live` | Live vectors counted by the harness (trust this one) |
| `jasper_n_live`, `jasper_n_tombstoned` | Jasper's own counts (see section 10) |
| `<phase>_gpu_util_mean/max` | % of time the GPU was running a kernel during that phase (NVML) |
| `<phase>_util_samples` | Utilization samples that phase got; 0 = too short to measure |
| `gpu_mem_used_gb` | GPU memory in use (whole GPU) after the step |
| `eval` | Why an evaluation ran here (`batch 40`, `runbook search`, ...) |
| `recall@10_bw<W>`, `qps_bw<W>` | Recall and queries/second at beam width W, on fresh ground truth |

## 8. What to look at first

1. **Insert speed vs batch size** (`plot_runs.py` on the batch-size runs): does throughput collapse at
   small batches?
2. **GPU utilization during insert:** low % at small batches means the GPU waits on per-batch overhead.
   Confirm with Nsight Systems (section 9) before claiming it.
3. **`keeps_up`:** at what batch sizes could Jasper sustain CANDOR's 10K vectors/s?
4. **Recall over the stream:** stable, or does it drift down (concept drift, deletions)?
5. **GPU memory under deletes:** it should keep growing even when `n_live` doesn't (section 10).

## 9. Profiling the GPU in more depth

NVML utilization only says whether *a* kernel was running, not how hard the GPU worked, and it misses
short phases. Use NVIDIA's tools for the "is the GPU underused?" question:

```bash
# Live view in a second terminal: SM % and memory % each second, for your GPU only
nvidia-smi dmon -s u -i $CUDA_VISIBLE_DEVICES

# Timeline of every kernel and copy: shows idle gaps between kernels
nsys profile -o bs500 --trace=cuda python3 stream_bench.py ... --max-batches 20
nsys stats bs500.nsys-rep            # text summary; or open the file in the Nsight Systems GUI

# Per-kernel detail: achieved SM throughput, memory bandwidth (% of peak), occupancy.
# Very slow and affects the whole GPU: only when nobody else is using it.
ncu --set full --launch-count 20 -o append_kernels python3 stream_bench.py ... --max-batches 2
```

If `ncu` reports `ERR_NVGPUCTRPERM`, it needs permission to read GPU counters; ask the admins.

## 10. Things to know about Jasper (from reading its code)

- **Ids:** the initial build gives ids 0..n-1; `append` hands out new, increasing ids; `search` returns
  them. The harness tracks the dataset row <-> id mapping.
- **Consolidate does not free memory.** It repairs edges and clears tombstones, but the slots stay
  allocated; `compact()` is compiled out by default (`-DJASPER_ENABLE_COMPACT=ON` to enable). So after a
  consolidate `jasper_n_live` still counts removed vectors (use `n_live`), and GPU memory keeps growing
  under deletes: the memory-bottleneck concern from the meeting.
- **Consolidate takes an exclusive lock:** no search runs while it does.
- **Settings:** max degree R in {8, 16, 32, 64}; distance `l2` or `ip`; vectors stored as float16.
- **`append` discards RaBitQ codes,** so the harness uses plain `search`.
- **Reference numbers** from Jasper's CI (their GPU and settings): bigann10M build 454K vec/s, append
  252K vec/s, consolidate 3.1 s, recall 0.957.

## 11. Limitations of this harness

- Ground truth is recomputed by brute force at each evaluation; on big runs lower `--n-queries` or
  raise `--eval-every`.
- Host-to-GPU copies of each batch are excluded from `append_s`.
- Updates and searches run one after another, never concurrently.
- CANDOR's noise and drop scenarios, stress tests and concurrent mode are not reproduced.
