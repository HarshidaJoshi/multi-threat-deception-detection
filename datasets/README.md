# Deepfake training dataset

**Folders are empty and ready for your images.**

## Add your data

```
datasets/real/     ← authentic photos (camera, stock, screenshots of real photos)
datasets/fake/     ← AI-generated images (ChatGPT, Midjourney, Leonardo, etc.)
```

Supported: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`  
Target: **50+ per folder** (200+ recommended for dissertation)

## Train after copying images

```powershell
python manage.py train_deepfake --dataset datasets
```

Full guide: [docs/DATASET_GUIDE.md](../docs/DATASET_GUIDE.md)

---

<details>
<summary>Demo data (optional — not recommended for real testing)</summary>

```powershell
python manage.py seed_demo_dataset --dataset datasets
python manage.py train_deepfake --dataset datasets
```
</details>
