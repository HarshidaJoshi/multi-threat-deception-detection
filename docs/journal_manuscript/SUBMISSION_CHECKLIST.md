# Cybernetics and Systems submission checklist

Official instructions: https://www.tandfonline.com/action/authorSubmission?show=instructions&journalCode=ucbs20

## Prepared files

- `manuscript.tex` — anonymous review manuscript
- `references.bib` — cited and internally consistent BibTeX database
- `title_page.tex` — identifying author information (separate upload)
- `cover_letter.md` — journal-fit cover-letter draft
- `figs/system_architecture.png` — 300 dpi manuscript copy
- `figs/gradcam_examples.png` — 300 dpi manuscript copy
- `figs/system_architecture.tiff` — 300 dpi preferred-format production copy
- `figs/gradcam_examples.tiff` — 300 dpi preferred-format production copy

## Author information still required

- [ ] Institutional email for each author (the journal explicitly requires institutional addresses)
- [ ] Institutional profile URL for each author
- [ ] ORCID for each author, or “not available”
- [ ] Corresponding author selection
- [ ] Biography for each author (maximum 200 words suggested)
- [ ] Final CRediT roles for each author
- [ ] Confirm exact author order; the journal does not permit post-submission author-list changes

## Statements to confirm

- [ ] Replace the bracketed funding statement in `manuscript.tex`
- [ ] Replace the bracketed competing-interests statement
- [ ] Insert a public repository URL and DOI/persistent identifier (or anonymous reviewer link)
- [ ] Review the generative-AI disclosure for accuracy
- [ ] Confirm the cover letter’s originality and simultaneous-submission statement
- [ ] Add identifying acknowledgments only after anonymous review, if applicable

## Technical checks

- [x] Unstructured abstract: 100 words
- [x] Keywords: five
- [x] Anonymous manuscript and separate title page
- [x] BibTeX supplied for LaTeX submission
- [x] Every bibliography entry cited; no unresolved citation keys
- [x] Figures are 300 dpi and have image descriptions
- [x] Tables are editable LaTeX
- [x] LaTeX environments and braces pass static validation
- [ ] Compile on Overleaf or a machine with TeX Live/MiKTeX (no LaTeX compiler is installed locally)
- [ ] Inspect the compiled PDF for ≤20 pages, overfull tables, and font embedding
- [ ] Upload editable source (`.tex`, `.bib`, figures) with the review PDF

## Build on Overleaf / TeX Live

The main file is `manuscript.tex`; use BibTeX and compile in this order:

```text
pdflatex manuscript
bibtex manuscript
pdflatex manuscript
pdflatex manuscript
```

For the separate identifying page:

```text
pdflatex title_page
```

The journal accepts format-free initial submissions but requests an editable version at revision.
It has no submission, publication, or page charges for the subscription route; open access is
optional and may require an APC.
