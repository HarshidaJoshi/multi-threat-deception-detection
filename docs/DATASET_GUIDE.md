# Dataset guide for dissertation (real vs AI / deepfake)

Use this plan to fix false positives (real screenshot → fake) and false negatives (AI portrait → real).

---

## Golden rule

| Content | Label | Folder |
|---------|--------|--------|
| Real camera / stock photo | **real** | `datasets/real/` |
| Screenshot of a **real** photo | **real** | `datasets/real/` |
| ChatGPT / Midjourney / Leonardo / DALL·E image | **fake** | `datasets/fake/` |
| Screenshot of an **AI** image | **fake** | `datasets/fake/` |

**Screenshot does not change the label** — only the *content* matters.

---

## Folder layout (use flat folders)

The trainer reads **only** these two folders (images can be directly inside or one level down):

```
datasets/
├── real/                          ← ALL authentic images (training)
│   ├── camera_person_001.jpg
│   ├── camera_landscape_002.jpg
│   ├── screenshot_jump_001.png      ← your real jumping-man screenshot
│   ├── stock_portrait_003.jpg
│   └── ...
│
├── fake/                          ← ALL AI / synthetic images (training)
│   ├── ai_saree_portrait_001.png  ← your AI saree image
│   ├── chatgpt_face_002.png
│   ├── midjourney_scene_003.jpg
│   └── ...
│
└── holdout/                       ← OPTIONAL: never train on these (manual UI testing)
    ├── real/
    │   └── screenshot_jump_test.png
    └── fake/
        └── ai_saree_test.png
```

**Important:** Put files **inside** `real/` and `fake/`. Use **filenames** to remember type (`screenshot_`, `camera_`, `ai_`, `chatgpt_`).  
Do **not** use `datasets/holdout/` for training — only for tests you run in the web UI after training.

---

## How many images?

| Stage | Per class (real + fake) | Purpose |
|-------|-------------------------|---------|
| Minimum | 50 each | Basic training |
| Recommended | **200+ each** | Dissertation-quality results |
| Holdout test | 20–30 each | Images **never** copied into `real/` or `fake/` |

---

## What to collect for `datasets/real/`

Include **variety** — especially screenshots:

1. **Camera photos** — phone JPG with EXIF  
2. **Screenshots of real photos** — like your jumping-man image (PNG, no EXIF)  
3. **Stock / Google download** — real JPG/PNG (Dreamstime-style portraits)  
4. **Social media crops** — compressed, lower resolution  
5. **Different scenes** — outdoor, indoor, faces, full body, objects  

This teaches the model: *screenshot of real content is still real.*

---

## What to collect for `datasets/fake/`

1. **AI portraits** — Leonardo, Midjourney, DALL·E (like your saree image)  
2. **ChatGPT images** — faces, landscapes, products  
3. **Mixed styles** — cartoon, photorealistic, illustrations  
4. **Screenshots of AI images** — same as real class, but content is AI  

This teaches the model: *modern AI faces are fake.*

---

## Your three test images — where they belong

| Image | True label | Training folder | Notes |
|-------|------------|-----------------|-------|
| Jumping man (screenshot) | **real** | `datasets/real/screenshot_jump_xxx.png` | Fixes false positive |
| Dreamstime girl (if real stock) | **real** | `datasets/real/stock_xxx.jpg` | |
| Saree woman (AI) | **fake** | `datasets/fake/ai_saree_xxx.png` | Fixes false negative |

Keep **copies** of these in `datasets/holdout/` if you want to test after training without them being in the training set.

---

## Step-by-step workflow

### 1. Remove demo synthetic data

Delete or move old demo files from `datasets/real/` and `datasets/fake/`:

```
demo_real_*.jpg
demo_fake_*.jpg
```

### 2. Add your real dataset

Copy 200+ images into `datasets/real/` using clear names:

```
camera_001.jpg
screenshot_002.png
stock_003.jpg
```

### 3. Add your fake dataset

Copy 200+ AI images into `datasets/fake/`:

```
ai_portrait_001.png
chatgpt_002.webp
```

### 4. Train

```powershell
python manage.py train_deepfake --dataset datasets
```

### 5. Test (do not add holdout images to training folders)

1. Start server: `.\start_xai_verify.bat`  
2. Open `/deepfake/`  
3. Upload images from `datasets/holdout/` only  
4. Check: screenshot real → **Real**, AI saree → **Fake**

---

## Filename convention (recommended)

```
real/camera_<subject>_<001>.jpg
real/screenshot_<subject>_<001>.png
real/stock_<subject>_<001>.jpg

fake/ai_portrait_<001>.png
fake/chatgpt_<subject>_<001>.png
fake/midjourney_<001>.jpg
```

---

## Dissertation table (example)

You can report dataset composition like this:

| Class | Source types | Count |
|-------|----------------|-------|
| Real | Camera, screenshots, stock | 200 |
| Fake | ChatGPT, Midjourney, Leonardo AI | 200 |
| Holdout test | Not used in training | 30 |

---

## Common mistakes to avoid

| Mistake | Result |
|---------|--------|
| Training only on `seed_demo_dataset` | 100% val accuracy, fails on real images |
| No screenshots in `real/` | Real screenshots still called fake |
| No AI portraits in `fake/` | AI images still called real |
| Testing on images used in training | Accuracy looks better than it is |
| Mixing holdout into `real/`/`fake/` | Overfitting to test images |

---

## Quick checklist

- [ ] Removed `demo_real_*` and `demo_fake_*`  
- [ ] 200+ images in `datasets/real/` (include screenshots)  
- [ ] 200+ images in `datasets/fake/` (include AI portraits)  
- [ ] Holdout set saved separately  
- [ ] Retrained with `train_deepfake`  
- [ ] Tested jumping screenshot → Real  
- [ ] Tested AI saree → Fake  
