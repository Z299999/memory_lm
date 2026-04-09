# Add New Paper Prompt

Use this prompt when you have added a new paper PDF (and optional supplementary PDF) into `literature/pdf_papers/` and want to register it in the repository.

## Prompt

You are helping me register a new paper in this repository.

Inputs (fill in placeholders):
- New paper main PDF filename (already in `literature/pdf_papers/`): `{MAIN_PDF_FILENAME}`
- Optional supplementary PDF filename: `{SUPP_PDF_FILENAME_OR_EMPTY}`
- New paper ID (unique): `{BID}` (e.g., `b00002`)
- Short name (for folder): `{SHORTNAME}` (e.g., `nonlinear-feedback-asymmetric-polarization`)
- Metadata:
  - Title: `{TITLE}`
  - Authors: `{AUTHORS}`
  - Year: `{YEAR}`
  - BibTeX-style key: `{BIBKEY}` (e.g., `leonard2021nonlinear`)

Task:
1. Create the per-paper folder:

```bash
mkdir -p literature/pdf_papers/{BID}_{SHORTNAME}
```

2. Move the PDFs into that folder (include the supplementary file only if it exists):

```bash
mv literature/pdf_papers/{MAIN_PDF_FILENAME} literature/pdf_papers/{BID}_{SHORTNAME}/
mv literature/pdf_papers/{SUPP_PDF_FILENAME_OR_EMPTY} literature/pdf_papers/{BID}_{SHORTNAME}/  # optional
```

3. Append a new JSON line to `literature/bibliography.jsonl` with this structure:

```json
{"id":"{BID}","bibkey":"{BIBKEY}","type":"paper","title":"{TITLE}","authors":"{AUTHORS}","year":{YEAR},"path":"literature/pdf_papers/{BID}_{SHORTNAME}"}
```

4. Confirm:
   - The folder `literature/pdf_papers/{BID}_{SHORTNAME}` exists and contains the PDFs.
   - The new line is present in `literature/bibliography.jsonl` and is valid JSON (one line per entry).

Quality requirements:
- Do not use spaces in filenames or folder names.
- Keep `{BID}` unique and monotonically increasing when possible.
- Ensure the JSON line has double quotes around strings and no trailing commas.

