from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import os
import sys
import tempfile
import numpy as np
import cv2


app = FastAPI()

@app.post("/analyze-glaucoma")
async def analyze_glaucoma(file: UploadFile = File(...)):
    # 1. Read the file sent by the iOS app
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # =========================================================
    # // AI model invocation starts here
    # =========================================================
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    ai_dir = os.path.join(root_dir, 'ai')
    
    if ai_dir not in sys.path:
        sys.path.append(ai_dir)
        
    from pipeline.pipeline import GlaucomaPipeline

    # Global initialization
    global glaucoma_pipeline
    if 'glaucoma_pipeline' not in globals():
        yolo_path = os.path.join(ai_dir, 'pipeline', 'models', 'best.pt')
        unet_path = os.path.join(ai_dir, 'pipeline', 'models', 'unetpp_best.pth')
        print("[*] Inicjalizacja modeli AI...")
        glaucoma_pipeline = GlaucomaPipeline(yolo_path=yolo_path, unet_path=unet_path, device='cpu')

    # Temporary file for pipeline
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        image.save(tmp_path)

    try:
        result = glaucoma_pipeline.run(tmp_path)
        
        if result is not None:
            _, _, _, cdr_val, _, _ = result
            cup_to_disc_ratio = round(float(cdr_val), 2)
            is_glaucoma = bool(cup_to_disc_ratio > 0.65)
            confidence = 0.95 
        else:
            is_glaucoma = False
            cup_to_disc_ratio = 0.0
            confidence = 0.0
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

  

    
    open_cv_image = np.array(image) 

    #Applying masks to image
    if result is not None:
        full_img, crops, masks, cdr_val, _, _ = result
        
        if len(crops) > 0:
            x1, y1, x2, y2 = crops[0]
            disc_mask = masks[0][0] 
            cup_mask = masks[0][1]  

            roi = open_cv_image[y1:y2, x1:x2]
            
            roi_h, roi_w = roi.shape[:2]
            disc_resized = cv2.resize(disc_mask, (roi_w, roi_h))
            cup_resized = cv2.resize(cup_mask, (roi_w, roi_h))

            roi[disc_resized > 0.5] = roi[disc_resized > 0.5] * 0.5 + np.array([0, 255, 0]) * 0.5
            roi[cup_resized > 0.5] = roi[cup_resized > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5

            open_cv_image[y1:y2, x1:x2] = roi

    image = Image.fromarray(open_cv_image)

    # =========================================================
    # // AI model invocation ends here
    # =========================================================
    
    # =========================================================
    # // AI model invocation ends here
    # =========================================================
    
    # 2. Encode the processed image to Base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # 3. Return the response to Swift in JSON format
    return JSONResponse(content={
        "has_glaucoma": is_glaucoma,
        "confidence": confidence,
        "cup_to_disc_ratio": cup_to_disc_ratio,
        "image_base64": img_base64
    })