# Project Documentation Management 
This directory contains all technical reports, meeting minutes, and design documentation for the **Glaucoma Detection** project.

##  Directory Structure

* `meetings/` – Meetins description from weekly team meetings (MoM).
* `reports/` – Progress reports for supervisor.
* `technical/` – Technical documentation divided by project sector:
    * `camera/` – Optical calculations, hardware assembly, and materials.
    * `mobile/` – App architecture, FastAPI integration, and UI/UX flows.
    * `ai/` – Model training, dataset descriptions, and evaluation metrics.
* `other/` – Any other reports, mostly for studies purposes.

---

## LaTeX Templates

To maintain a professional and consistent look across all documents, use the provided `.tex` templates:

1.  **`meetings_template.tex`**
    * **Use for:** Internal team meetings.
    * **Filename convention:** `YYYY-MM-DD_meeting.pdf`

2.  **`raports_and_technical_template.tex`**
    * **Use for:** Progress reports and technical documentation in `technical/` subdirectories.
    * **Filename convention (reports):** `YYYY-MM-DD_progress.pdf`
    * **Filename convention (technical):** `short_description.pdf`

---

## Other Instructions
When documenting TODOs, use the following syntax to link directly to the repository:
```latex
\href{[https://github.com/kequel/jaskra/issues/X](https://github.com/kequel/jaskra/issues/X)}{Issue title \#X}