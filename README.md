# Glaucoma AI-Powered Fundus Image Detection 
## Project Overview
Glaucoma is one of the leading causes of irreversible vision loss worldwide. Early detection is critical, yet manual analysis of fundus photographs is time-consuming and requires specialized ophthalmological expertise.

This project automates that process using an AI pipeline (YOLO + UNet++) integrated into an iOS mobile application, accompanied by a custom smartphone fundus camera attachment. The project is supervised by prof. dr hab. inż. Andrzej Czyżewski at Gdańsk University of Technology.

## Team & Responsibilities

| Member | Role |
|--------|------|
| Martyna Borkowska | Project Leader, Backend Developer |
| Karolina Glaza | Scrum Master, Repository Admin, Frontend Developer |
| Agnieszka Pawłowska | Fundus Camera Specialist, Yolo Developer|
| Amila Amarasekara | Yolo Lead Developer |
| Mateusz Dobry | Unet++ Lead Developer |
| Antoni Naczke | Fundus Camera Specialist, Unet++ Developer|


> **Release 1.0** — initial version with the legacy backend (`/analyze-glaucoma` JSON + streaming endpoint) and the first iOS UI.

## Repository Structure
```
/
├── ai/
│   ├── pipeline.py - Inference pipeline (YOLO + UNet++ + CDR calculation)
│   ├── yolo/       - ROI detection (training notebooks & scripts)
│   ├── unet/       - Segmentation model (training)
│   └── data/       - Sample fundus images & masks
├── backend/ - FastAPI backend, hosted on Microsoft Azure
├── mobile/  - iOS app (Swift Playgrounds)
├── tests/   - Pytest test suite (work in progress)
└── docs/    - Project documentation
```

> **Model weights** (`*.pt` / `*.pth`) are **not committed to the repository** — they are
> published as assets of the GitHub Release tagged `models-v1` and downloaded automatically
> by CI. This keeps the repo lightweight and avoids Git LFS storage limits.

## Technical Stack
### AI Pipeline
- **YOLO** - detects the optic disc ROI in the fundus image
- **UNet++** - segments optic disc and cup
- **CDR (Cup-to-Disc Ratio)** - computed from segmentation masks
- **PyTorch** - training and inference
- **Jupyter Notebook** - data exploration and prototyping
 
### Backend
- **FastAPI** - REST API exposing two endpoints:
  - `POST /analyze-glaucoma` - standard JSON response
  - `POST /analyze-glaucoma-stream` - NDJSON stream with real-time progress steps
- **Microsoft Azure** - cloud hosting


Accepts a fundus image (JPEG), returns: `has_glaucoma`, `confidence`, `cup_to_disc_ratio`, `image_base64` (annotated image)
 
### Mobile
- **Swift Playgrounds** - iOS application (iPhone & iPad)


Sends fundus photo to backend, displays diagnosis with CDR scale and confidence score
 
### Hardware
- Smartphone fundus camera attachment (in development)

## Tests

Automated tests live in `tests/` and use **pytest**. The suite is **work in progress** — backend and UNet++ coverage exist today; pipeline, YOLO and mobile tests are still being added.

```
tests/
├── conftest.py   - shared fixtures (images, masks, mock pipeline outputs)
├── base_test.py  - BaseTest with shared assertion helpers
├── backend/      - API, integration, streaming & validation tests
└── ai_unet/      - UNet++ training tests
```
Planned: `ai_pipeline/`, `ai_yolo/`, `mobile/`.

Conventions (full rules in [`tests/README_test_writting_rules.md`](tests/README_test_writting_rules.md)):
- Test classes inherit from `BaseTest`; all fixtures live in `conftest.py`.
- Every test class and method carries a mandatory docstring "test plan" (`TEST NAME` / `COMPONENT` / numbered steps).
- Naming: files `test_<module>.py`, classes `Test<Feature>`, methods `test_<what>_<expected_result>`.

Run locally:
```bash
pip install -r backend/requirements.txt pytest pytest-cov httpx Pillow

# unit tests
pytest tests/backend/test_api.py -v

# integration tests (start the API first)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
pytest tests/backend/test_integration.py -v
```

## Branch Strategy
 
- `main` is the stable branch - changes land only via Pull Requests
- Each GitHub Issue has exactly one corresponding branch, branched from `main`
- Branch naming: `{issue-number}-short-description`
- Every PR to `main` requires **2 approvals** from team members other than the author and status checks to pass

### CI/CD (GitHub Actions)

 CI checks run on every PR to `main`. The backend integration test downloads the model weights from the `models-v1` GitHub Release before starting the API. Merge is blocked if backend syntax/imports are invalid, the `/analyze-glaucoma` endpoint returns an incorrect response, any Swift file fails to compile, or `pipeline.py` has syntax errors. Style issues and TODOs are reported as warnings only.
 
### Labeling
PRs are automatically labeled: `ai`, `backend`, `mobile`, `hardware`, `docs`, `ci`, `dependencies`