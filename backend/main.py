from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
import io
import base64
import random

app = FastAPI()

@app.post("/analyze-glaucoma")
async def analyze_glaucoma(file: UploadFile = File(...)):
    # 1. Read the file sent by the iOS app
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # =========================================================
    # // AI model invocation here (glaucoma detection and masks)
    # =========================================================
    
    # Simulate model results:
    is_glaucoma = random.choice([True, False])
    confidence = round(random.uniform(0.85, 0.99), 2)
    cup_to_disc_ratio = round(random.uniform(0.3, 0.8), 2) 
    
    # Simulate applying a mask to the image
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.ellipse(
        [(width * 0.3, height * 0.3), (width * 0.7, height * 0.7)], 
        outline="red" if is_glaucoma else "green", 
        width=8
    )
    
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