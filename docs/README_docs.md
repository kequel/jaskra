# Project Documentation

This directory holds all non-code documentation for the **Glaucoma Detection** project:
meeting minutes, Scrum/agile course deliverables, and technical reports.

## Directory structure
```
docs/
├── meetings/                  - Minutes of weekly team meetings (MoM)
│   ├── 2025/
│   ├── 2026/
│   └── meeting_template.tex   - LaTeX template for meeting minutes
├── scrum/                     - Agile/Scrum course deliverables (see scrum/README_scrum_docs.md)
├── technical/                 - Technical reports, grouped by area
│   ├── ai/
│   ├── app/
│   ├── camera/
│   └── technical_template.tex - LaTeX template for technical reports
└── README_docs.md
```

## LaTeX templates
Use the shared templates to keep documents consistent:

- **`meetings/meeting_template.tex`** — meeting minutes. Filename convention: `YYYY-MM-DD_meeting.pdf`.
- **`technical/technical_template.tex`** — technical reports & documentation. Filename convention: `short_description.pdf`.

## Linking issues from documents
When referencing a GitHub issue inside a LaTeX document:
```latex
\href{https://github.com/kequel/jaskra/issues/X}{Issue title \#X}
```

## Language
Meeting minutes and Scrum deliverables are written in **Polish** (course requirement).
Technical reports must be in English.
