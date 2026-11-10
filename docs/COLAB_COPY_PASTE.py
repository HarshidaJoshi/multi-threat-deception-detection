# =============================================================================
# XAI-Verify — Google Colab: train BERT + RoBERTa (copy each block into a cell)
# Runtime → Change runtime type → T4 GPU
# =============================================================================

# --- CELL 1: GPU ---
# !nvidia-smi
# import torch
# print("cuda:", torch.cuda.is_available())
# assert torch.cuda.is_available(), "Enable T4 GPU in Runtime settings"

# --- CELL 2: Upload full project zip ---
# from google.colab import files
# import glob, os, zipfile
# print("Upload deepfake_detection_project.zip")
# uploaded = files.upload()
# zips = [f for f in uploaded if f.lower().endswith(".zip")]
# assert zips, "Upload a zip"
# for z in zips:
#     zipfile.ZipFile(z).extractall("/content")
# matches = glob.glob("/content/**/train_fake_news_transformers.py", recursive=True)
# assert matches, "Need scripts/train_fake_news_transformers.py in zip"
# PROJECT = matches[0].split("/scripts/")[0]
# os.chdir(PROJECT)
# print("Project root:", PROJECT)

# --- CELL 3 (optional): Upload CSV + utils if not in zip ---
# from google.colab import files
# import os, shutil, zipfile
# PROJECT = globals().get("PROJECT", "/content")
# os.makedirs(f"{PROJECT}/datasets/text", exist_ok=True)
# print("Upload WELFake_Dataset.csv")
# for name in files.upload().keys():
#     if name.lower().endswith(".csv"):
#         shutil.move(name, f"{PROJECT}/datasets/text/WELFake_Dataset.csv")
# print("Upload utils.zip")
# for name in files.upload().keys():
#     if name.lower().endswith(".zip"):
#         zipfile.ZipFile(name).extractall(PROJECT)

# --- CELL 4: Preflight ---
# import os
# from pathlib import Path
# PROJECT = Path(os.getcwd())
# for rel in [
#     "scripts/train_fake_news_transformers.py",
#     "utils/text_ml.py",
#     "datasets/text/WELFake_Dataset.csv",
# ]:
#     p = PROJECT / rel
#     print("OK" if p.is_file() else "MISSING", rel)

# --- CELL 5: Install ---
# !pip install -q torch transformers scikit-learn numpy accelerate

# --- CELL 6: Train ---
# !python scripts/train_fake_news_transformers.py --architecture both --epochs 3 --batch-size 16

# --- CELL 7: Download ---
# from google.colab import files
# import shutil
# shutil.make_archive("/content/fake_news_models", "zip", "trained_models")
# files.download("/content/fake_news_models.zip")
