# Springer Template Format Guide (ICACS / SICACS 2026)

Your folder `Word+Latex-Template` is the **Springer LNCS / LNNS proceedings** package (`llncs` + Word macro `splnproc1703`).

---

## What format to use

| Option | File in your template folder | When to use |
|--------|------------------------------|-------------|
| **LaTeX (recommended here)** | `llncs.cls` + `samplepaper.tex` | Cleanest Springer formatting; paper already prepared |
| **Word** | `splnproc1703.docm` (Windows) or `splnproc1703_mac.docm` | If you prefer MS Word |
| Instructions | `SPLNPROC Word 2010-2016 Technical Instructions.pdf`, `Quick Start.pdf`, `llncsdoc.pdf` | Read once before editing |

---

## A) LaTeX — ready file (already created)

**Folder:** `f:\deepfake_detection_project\docs\sicacs2026_llncs\`

| File | Role |
|------|------|
| `paper.tex` | Full survey in LNCS style |
| `llncs.cls` | Springer class |
| `splncs04.bst` | BibTeX style (optional) |
| `fig1.png` | Taxonomy figure |

### How to compile
1. Install MiKTeX or TeX Live (Windows).
2. Open `paper.tex` in TeXstudio / Overleaf.
3. Replace `First Author`, affiliation, email, ORCID.
4. Compile: **pdfLaTeX** (run twice).
5. Upload the resulting **PDF** to CMT.

### Overleaf tip
Upload the whole `sicacs2026_llncs` folder as a project; set `paper.tex` as main file.

---

## B) Word — how to fill `splnproc1703.docm`

1. Open  
   `f:\MTECH\Literature Review\Word+Latex-Template\Word+Latex-Template\splnproc1703.docm`  
   (enable macros if Word asks).
2. Keep Springer styles only (`Title`, `Author`, `Abstract`, `Heading 1`, `Heading 2`, `Keywords`, etc.).  
   Do **not** use Normal + manual bold for headings.
3. Paste content from `docs\SICACS2026_CASE_A_FINAL.md` section by section.
4. Insert taxonomy image:  
   `docs\figures\sicacs2026_taxonomy_figure.png`  
   Caption **below** figure:  
   `Fig. 1. Taxonomy of multi-threat digital deception detection…`
5. Table captions go **above** tables (Springer rule).
6. Abstract ≈ **150–250 words** (already OK).
7. Keywords: use `\and`-style separators in Word as shown in template sample (`keyword · keyword`).
8. Remove: headers/footers you added, hyperlinks, Markdown `#` symbols.
9. Save as `.docx` / export **PDF** for CMT.

### Word heading map (from Springer sample)

| Level | Example | Style |
|-------|---------|--------|
| Title | Paper title | 14 pt bold, centered |
| 1st | `1 Introduction` | 12 pt bold |
| 2nd | `2.1 Scope…` | 10 pt bold |
| 3rd | run-in bold | unnumbered |
| 4th | run-in italic | lowest |

### References in Word (LNCS style example)
```
1. Mirsky, Y., Lee, W.: The creation and detection of deepfakes: a survey.
   ACM Computing Surveys 54(1), 1–41 (2021). https://doi.org/10.1145/3425780
```

---

## C) Conference rules reminder

- Minimum about **10 pages** in the Springer template.
- No custom headers/footers; avoid embedded web links in manuscript.
- Submit **PDF** only via CMT: https://cmt3.research.microsoft.com/SICACS2026/Submission/Index
- Proceedings series: Springer **Lecture Notes in Networks and Systems (LNNS)** (same author template family as LNCS).

---

## Quick choice

- Know LaTeX / Overleaf → use **`docs/sicacs2026_llncs/paper.tex`**
- Only Word → open **`splnproc1703.docm`** and paste from Case A using this guide
