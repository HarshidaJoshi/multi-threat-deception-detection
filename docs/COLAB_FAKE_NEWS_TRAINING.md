# Train BERT + RoBERTa on Google Colab (GPU)

Colab starts in `/content` with **no project files**. You must upload the code + dataset first, then run from the **project root** (folder that contains `scripts/`).

## Option A — Upload a ZIP (easiest)

### 1. On your PC, create a zip with this layout (scripts alone is **not** enough):

```
/content/                          ← project root after unzip
  scripts/train_fake_news_transformers.py
  scripts/train_text_benchmark.py
  utils/text_ml.py
  utils/__init__.py
  datasets/text/WELFake_Dataset.csv
```

**Easiest:** zip the whole folder `deepfake_detection_project` and extract so `scripts/` and `utils/` are **siblings** under one folder.

**If you only uploaded `scripts.zip`:** you will get `FileNotFoundError` on WELFake — add the CSV and `utils/` as below.

### 2. In Colab: **Runtime → Change runtime type → T4 GPU**

If you see `nvidia-smi: command not found` and `cuda: False`:

1. **Runtime → Change runtime type**
2. **Hardware accelerator:** choose **T4 GPU** (not CPU)
3. Click **Save**
4. **Runtime → Restart session** (important)
5. Run cell 1 again — should show `cuda: True`

Free Colab sometimes shows **GPU not available** (quota). Wait and retry later, or use CPU fallback in the notebook (small `--max-train`, slow).

### 3. Paste these cells in order

**Cell 1 — GPU check**

```python
!nvidia-smi
import torch
print("cuda:", torch.cuda.is_available())
```

**Cell 2 — Upload zip**

```python
from google.colab import files
uploaded = files.upload()  # pick your zip file
```

**Cell 3 — Unzip and go to project root**

```python
import zipfile, os, glob
zips = [f for f in uploaded if f.endswith(".zip")]
assert zips, "Upload a .zip of the project"
zipfile.ZipFile(zips[0]).extractall("/content")
# Find folder that contains scripts/train_fake_news_transformers.py
matches = glob.glob("/content/**/train_fake_news_transformers.py", recursive=True)
assert matches, "Zip must include scripts/train_fake_news_transformers.py"
PROJECT = matches[0].split("/scripts/")[0]
os.chdir(PROJECT)
print("Project root:", PROJECT)
!ls scripts/train_fake_news_transformers.py
!ls datasets/text/WELFake_Dataset.csv
```

**Cell 4 — Install deps (light; no Django/TensorFlow needed)**

```python
!pip install -q torch transformers scikit-learn numpy accelerate
```

**Cell 4b — (Only if CSV/utils missing) upload WELFake + utils**

```python
from google.colab import files
import os, shutil
os.makedirs("/content/datasets/text", exist_ok=True)
print("1) Select WELFake_Dataset.csv from your PC (datasets/text/)")
up = files.upload()
for name in up:
    if name.lower().endswith(".csv"):
        shutil.move(name, "/content/datasets/text/WELFake_Dataset.csv")
print("2) Select a zip of your utils/ folder, then:")
up2 = files.upload()
for name in up2:
    if name.endswith(".zip"):
        !unzip -o {name} -d /content
!test -f /content/utils/text_ml.py && echo "utils OK"
!test -f /content/datasets/text/WELFake_Dataset.csv && echo "WELFake OK"
```

**Cell 5 — Train (full WELFake on GPU, ~30–90 min)**

```python
import os
PROJECT = "/content"  # must contain scripts/, utils/, datasets/text/
os.chdir(PROJECT)
!python scripts/train_fake_news_transformers.py --architecture both --epochs 3 --batch-size 16
```

If the CSV is elsewhere:

```python
!python scripts/train_fake_news_transformers.py --architecture both --epochs 3 \
  --welfake-csv /content/WELFake_Dataset.csv
```

**Cell 6 — Download trained models back to your PC**

```python
from google.colab import files
import shutil
shutil.make_archive("/content/fake_news_models", "zip", f"{PROJECT}/trained_models",
                    base_dir=".")
files.download("/content/fake_news_models.zip")
```

### 4. On your PC

Unzip into `F:\deepfake_detection_project\trained_models\` (merge folders):

- `fake_news_bert_welfake/` (must include `model.safetensors`)
- `fake_news_roberta_welfake/`
- `*.meta.json`

Restart Django and select BERT or RoBERTa on the Fake News page.

---

## Option B — Google Drive

1. Copy the project folder to Drive.
2. Mount drive in Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
PROJECT = "/content/drive/MyDrive/deepfake_detection_project"  # adjust path
%cd $PROJECT
!python scripts/train_fake_news_transformers.py --architecture both --epochs 3
```

---

## Common error

| Error | Fix |
|-------|-----|
| `can't open file .../scripts/train_fake_news_transformers.py` | You are not in the project root. Run `%cd` to the folder that **contains** `scripts/`. |
| `WELFake_Dataset.csv` missing | Add `datasets/text/WELFake_Dataset.csv` to the zip. |
| `cuda: False` | Enable **GPU** in Runtime settings. |

---

## Open ready-made notebook

1. In Colab: **File → Upload notebook**
2. Choose: `F:\deepfake_detection_project\notebooks\colab_train_fake_news.ipynb`
3. **Runtime → T4 GPU**, then **Run all**

All code is also in `docs/COLAB_COPY_PASTE.py` (commented blocks per cell).
