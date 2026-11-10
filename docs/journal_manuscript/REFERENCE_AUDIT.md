# Reference audit

The review manuscript now uses `references.bib`. Every BibTeX entry has a corresponding in-text
citation, and every citation key resolves.

## Sources added to strengthen the central claim

- Yan et al. (2023), **DeepfakeBench**, official NeurIPS proceedings.
- Nadimpalli and Rattani (2022), cross-dataset deepfake generalisation, official CVF proceedings.
- Calvo and Laureano (2026), fake-news structural homogeneity and cross-dataset evaluation,
  official ACL Anthology entry, DOI `10.63317/4ka6fv4hbw9z`.
- Rössler et al. (2019), FaceForensics++, official ICCV proceedings.
- Wang et al. (2020), generator-specific synthetic-image cues, official CVPR proceedings.

These references support the paper’s actual novelty (evaluation under environment shift) rather
than suggesting novelty in ResNet, TF-IDF, or Grad-CAM themselves.

## Entries removed from the old manual bibliography

The previous bibliography contained several broad recent papers that were not necessary for a
specific claim. They were removed to keep the review focused and to avoid citation padding. In
particular, the entry titled only “DeepFake Detection Using CNN Models with Explainable AI
Techniques: IEEE Access (2025)” was misidentified: “ACCESS” is the acronym of the 2025 IEEE
conference where the paper appeared, not the journal *IEEE Access*. The paper is real, but the old
record omitted its authors and pages and the revised manuscript does not rely on it. It has
therefore not been carried into `references.bib`.

Removal does **not** imply that each omitted paper is invalid; it means the submitted manuscript
does not rely on it. No citation was retained solely to increase the reference count.

## Author check before submission

Although metadata was checked against publisher or official proceedings pages where possible,
authors should run a final DOI lookup through Crossref or their reference manager after importing
`references.bib`, especially for capitalization and final issue/article-number updates in 2025–2026
publications.
