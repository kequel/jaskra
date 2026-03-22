# Glaucoma AI-Powered Fundus Image Detection 
## Project Overview
Glaucoma is one of the leading causes of irreversible vision loss worldwide. Early detection is critical, yet manual analysis of fundus photographs is time-consuming and requires specialized ophthalmological expertise.

This project automates that process using an AI pipeline (YOLO + UNet++) integrated into an iOS mobile application, accompanied by a custom smartphone fundus camera attachment. The project is supervised by prof. dr hab. inż. Andrzej Czyżewski at Gdańsk University of Technology.

## Team & Responsibilities

| Member | Role |
|--------|------|
| Martyna Borkowska | Project Leader, Backend Developer |
| Karolina Glaza | Repository Admin, Frontend Developer |
| Agnieszka Pawłowska | Fundus Camera Specialist, Yolo Developer|
| Amila Amarasekara | Yolo Lead Developer |
| Mateusz Dobry | Unet++ Lead Developer |
| Antoni Naczke | Fundus Camera Specialist, Unet++ Developer|


## Repository Structure
```
/
├── ai/
│   ├── pipeline/ -  Pipeline (YOLO + UNet++ + CDR calculation)
│   ├── yolo/ - ROI detection training
│   └── unet/ - Segmentation model
├── backend/ - FastAPI backend, hosted on Microsoft Azure
├── mobile/ - iOS app (Swift Playgrounds)
├── hardware/ - Fundus camera attachment
└── docs/ - Project documentation
```

## Technical Stack
### AI Pipeline
- **YOLO** - detects the optic disc ROI in the fundus image
- **UNet++** - segments optic disc and cup
- **CDR (Cup-to-Disc Ratio)** - computed from segmentation masks
- **PyTorch** - training and inference
- **Jupyter Notebook** - data exploration and prototyping
 
### Backend
- **FastAPI** - REST API exposing `/analyze-glaucoma` endpoint
- **Microsoft Azure** - cloud hosting


Accepts a fundus image (JPEG), returns: `has_glaucoma`, `confidence`, `cup_to_disc_ratio`, `image_base64` (annotated image)
 
### Mobile
- **Swift Playgrounds** - iOS application (iPhone & iPad)


Sends fundus photo to backend, displays diagnosis with CDR scale and confidence score
 
### Hardware
- Smartphone fundus camera attachment (in development)

## Branch Strategy
 
- `main` is the stable branch - changes land only via Pull Requests
- Each GitHub Issue has exactly one corresponding branch, branched from `main`
- Branch naming: `{issue-number}-short-description`
- Every PR to `main` requires **2 approvals** from team members other than the author and status checks to pass

### CI/CD (GitHub Actions)

 CI checks run on every PR to `main`. Merge is blocked if backend syntax/imports are invalid, the `/analyze-glaucoma` endpoint returns an incorrect response, any Swift file fails to compile, or `pipeline.py` has syntax errors. Style issues and TODOs are reported as warnings only.
 
### Labeling
PRs are automatically labeled: `ai`, `backend`, `mobile`, `hardware`, `docs`, `ci`, `dependencies`