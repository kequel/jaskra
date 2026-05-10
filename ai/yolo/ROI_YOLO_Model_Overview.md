# ROI YOLO Model Folder Overview

This folder contains the main files used to train, evaluate, and run a YOLO-based ROI detector for fundus images.

## Folder contents

### `labeling_script_yolo/`
This folder contains the automatic labeling script for YOLO format annotations.

- `main.py` is used to generate YOLO labels automatically for image pairs where both of the following are available:
  - the original full fundus image,
  - the corresponding cropped image.
- The script finds the best matching ROI in the full image and saves the result as a YOLO bounding box annotation.

### `weights/`
This folder contains the trained model weights.

- `last.pt` — weights from the **last training epoch**.
- `best.pt` — the **best-performing weights** saved during training based on validation performance.

### `evaluation_files/`
This folder contains the model evaluation outputs, including plots and example prediction batches.

It includes files such as:
- confusion matrices,
- precision-recall and confidence curves,
- training/validation loss plots,
- example training and validation prediction images.

### Model evaluation summary
The evaluation results show that the model performed **very well** on the ROI detection task.

Main metrics:
- **mAP@50:** 98.7%
- **Precision:** 98.9%
- **Recall:** 99.0%
- **F1:** 98.9%

This means the model is very good at finding the ROI in fundus images, with very few false positives and false negatives.

### `Google_Colab_Training_Notebook.ipynb`
This notebook is a Google Colab training notebook prepared by **Roboflow**.

It can be used to train a **YOLOv11** model on a custom dataset in Colab.

### `Predict_ROI.ipynb`
This notebook contains a prediction script used to detect the ROI and crop fundus images to the detected region only.

It is intended for inference: given an input fundus image, the script runs the trained model and saves the cropped ROI image.

## Dataset and training setup

The model was trained on the **G1020** dataset with additional augmentation applied.

As a result, the final training dataset contained **2469 images**:

- **train:** 2175
- **val:** 194
- **test:** 100

### Preprocessing
- Resize to **512 × 512**

### Augmentation
- **Horizontal flip**
- **Rotation:** between **-15° and +15°**
- **Saturation:** between **-25% and +25%**
- **Blur:** up to **2.5 px**
