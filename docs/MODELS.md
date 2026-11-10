# Model details (for understanding)

Plain-language reference for your dissertation — no extra commands required.

---

## DeepFake image detection

### EfficientNetB0 (default in dropdown)

- **What it is:** A convolutional neural network (CNN) pre-trained on ImageNet, then fine-tuned on your `datasets/real/` and `datasets/fake/` folders.
- **Input:** One image (resized to 224×224 pixels).
- **How it decides:** The network outputs a number between 0 and 1 = probability the image is fake/AI. Above 0.5 → classified as fake/deepfake.
- **What the UI shows:**
  - **Prediction** — Real, Deepfake, or AI Generated
  - **Confidence** — how sure the model is (for this one image)
  - **Grad-CAM heatmap** — red areas the model focused on
  - **Training validation accuracy** — how well the model did on held-out images *during training* (not the same as confidence on one upload)

### Forensics (second option in dropdown)

- **What it is:** Rule-based checks — no deep learning.
- **Checks:** Camera EXIF data, image sharpness, texture noise, file type.
- **Use:** Baseline comparison; weak on modern AI faces.

---

## Misinformation — fake news

### BERT (default)

- **What it is:** BERT transformer (`elozano/bert-base-cased-fake-news`) from HuggingFace, fine-tuned for fake vs real news.
- **Input:** Text (news post, headline, paragraph).
- **Output:** Fake News or Real News + confidence %.
- **XAI:** LIME (important words), SHAP-style feature bars.

### RoBERTa

- Same task as BERT but RoBERTa architecture (`hamzab/roberta-fake-news-classification`) — for **comparing models** in your thesis.

### Forensics

- Keyword scoring (scam, urgent, fake, etc.) — baseline only.

---

## Recruitment scam

- Rule-based under misinformation (job text, company name, website domain).
- Not a separate deep learning model.

---

## Why 100% accuracy appeared

That number is **training validation accuracy** on your **demo/synthetic** dataset (60+60 simple images). It does **not** mean the model is 100% correct on real Google photos. For real-world images you need a real dataset and separate testing.

---

## Metrics for your dissertation (conceptual)

You can report these in your thesis without any special project command:

| Metric | Simple meaning |
|--------|----------------|
| **Accuracy** | Correct predictions ÷ total test samples |
| **Precision** | Of all called “fake”, how many were truly fake |
| **Recall** | Of all truly fake, how many were detected |
| **F1 score** | Balance between precision and recall |
| **Specificity** | Of all truly real, how many were called real |

**Confusion matrix:**

```
                 Predicted Real    Predicted Fake
Actually Real         TN                FP
Actually Fake         FN                TP
```

- **False positive (FP):** Real photo called fake (e.g. your jumping-man image).
- **False negative (FN):** AI image called real (e.g. your saree AI portrait).

**How to get numbers for thesis:** Label a test set manually (e.g. 30 real + 30 fake images), run each through the web UI with each model, record predictions in Excel, then calculate accuracy/F1 using standard formulas or spreadsheet functions.

---

## Your three test images (summary)

| Image | Likely truth | Model tended to say | Why |
|-------|--------------|---------------------|-----|
| Girl, Dreamstime | Real stock photo | Real | Looks like professional photo; model may be correct |
| Man jumping | Real photo | Fake | Trained on wrong data; PNG, no EXIF |
| Saree woman | AI generated | Real | Model never saw similar AI portraits in training |

**Fix for better results:** Retrain on real photos + real AI images (200+ each), then test on new images never used in training.
