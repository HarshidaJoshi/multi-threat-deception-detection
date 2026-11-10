# GPU run guide (Kaggle) — the experiments a journal reviewer will ask for

This runs, in one command: all four image models on the **full 100,000-image training set**, **3 repeated
seeds** (mean ± std), the ablations, cross-domain tests (CIFAKE), significance tests, and **DistilBERT / BERT**
baselines for fake news and recruitment scams (3 seeds). It ends with `aggregated_tables.md`, ready to paste
into the paper.

> The scripts were checked on the CPU laptop only (data loading, metrics, aggregation, a tiny image run).
> The PyTorch part has **not** run anywhere yet, so always do the **smoke run (step 5)** first.

## 1. What to upload
| What | How | Size |
|---|---|---|
| `gpu_package.zip` (code + ISOT, WELFake, LIAR, EMSCAD) | Kaggle → *Datasets → New dataset → upload* the zip (Kaggle unzips it) | ~0.4 GB |
| Face images | Easiest: in the notebook press *Add data* and search **"140k Real and Fake Faces"** (public, ~4 GB, no upload). Check that it contains `real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}`. If not, upload your own `deepfakeimage dataset.zip`. | 4 GB |
| CIFAKE (second generator) | *Add data* → search **"CIFAKE: Real and AI-Generated Synthetic Images"** (public, ~110 MB) | 0.1 GB |

## 2. Create the notebook
*Code → New notebook*. In the right panel: **Accelerator = GPU T4 ×2 (or P100)**, **Internet = On** (needed to
download ImageNet and Hugging Face weights). *Add data* → the three datasets above.

## 3. Cell 1 — set up (adjust the two `/kaggle/input/...` folder names to what you see in the left panel)
```python
!mkdir -p /kaggle/working/project && cp -r /kaggle/input/<your-package-dataset>/* /kaggle/working/project/
!pip install -q scikit-learn scipy matplotlib
import os; print(os.listdir('/kaggle/working/project'))   # should show scripts, utils, datasets, fake-and-real-news-dataset
```

## 4. Find the exact data paths
```python
!ls /kaggle/input
!ls /kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake     # must list train valid test
```

## 5. Smoke run (≈ 10–15 min) — do this first
```python
!cd /kaggle/working/project && python scripts/gpu_run_all.py --smoke \
  --image-data /kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake \
  --cifake-test /kaggle/input/cifake-real-and-ai-generated-synthetic-images/test \
  --root /kaggle/working/project --out /kaggle/working/out_smoke --mixed-precision
```
It must finish with “Done” and a `aggregated_tables.md`. Fix any error here (send it to me) before the long run.

## 6. Full run (estimate: 6–10 GPU-hours; split it, see below)
```python
!cd /kaggle/working/project && python scripts/gpu_run_all.py \
  --image-data /kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake \
  --cifake-test /kaggle/input/cifake-real-and-ai-generated-synthetic-images/test \
  --root /kaggle/working/project --out /kaggle/working/out --mixed-precision
```
* Use **Save Version → Save & Run All (Commit)** so it runs in the background even if you close the browser.
* A session is limited to ~12 h and the weekly GPU quota is ~30 h. If it stops, just run the same command again:
  finished seeds are skipped.
* To split: first `--seeds 1`, later `--seeds 2 3` (same `--out`); then `--skip-image --skip-text` is not needed —
  `gpu_aggregate.py --out ...` rebuilds the tables.
* Faster alternatives: `--image-models resnet50 efficientnetb0` (the two that matter most), or `--text-models distilbert-base-uncased` only.

## 7. Download the results
Everything is in `/kaggle/working/out` (open the *Output* tab, download all). Send me the folder, or at least:
`aggregated_tables.md`, `aggregated.json`, and each `image_seed*/results.json` + `robustness.json`.
I will then rewrite the paper's methods, results and discussion with the real numbers.

## Colab instead of Kaggle
Same commands; mount Google Drive, copy the package and the face dataset there, set *Runtime → T4 GPU*.
Colab sessions are shorter (~4–12 h), so run seeds one at a time.

## What the numbers will let the paper claim
* Full-data **fine-tuning** (not only a frozen backbone) for ResNet50/EfficientNetB0/MobileNetV2, mean ± std over 3 seeds.
* Cross-generator behaviour (faces → CIFAKE) and McNemar significance tests between models.
* Whether transformer text models close the cross-dataset gap that TF-IDF models showed (they may not — report either way).
