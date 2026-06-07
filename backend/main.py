from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext
from datetime import datetime, timedelta
from PIL import Image
from threading import Lock
from dotenv import load_dotenv
import io
import base64
import os
import sys
import tempfile
import numpy as np
import cv2
import json
import asyncio
import jwt
import shutil
import pika
# =========================================================
# 1. DATABASE CONFIGURATION (SQLite)
# =========================================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./glaucoma_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class History(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    is_glaucoma = Column(Boolean)
    cup_to_disc_ratio = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================================================
# 2. IDENTITY & JWT BEARER TOKENS
# =========================================================
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("CRITICAL ERROR: SECRET_KEY environment variable is missing! Application stopped for security reasons.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except jwt.PyJWTError:
        return None
    user = db.query(User).filter(User.username == username).first()
    return user

app = FastAPI()
model_lock = Lock()  # Safeguard for lazy loading AI models

# =========================================================
# 3. IDENTITY ENDPOINTS
# =========================================================
@app.post("/register")
def register_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    hashed_password = get_password_hash(form_data.password)
    new_user = User(username=form_data.username, hashed_password=hashed_password)
    try:
        db.add(new_user)
        db.commit()
        return {"message": "Account created successfully!"}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/history")
def get_user_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="You must be logged in to view history")
    records = db.query(History).filter(History.user_id == current_user.id).all()
    return {"history": [{"id": r.id, "is_glaucoma": r.is_glaucoma, "cdr": r.cup_to_disc_ratio, "date": r.created_at} for r in records]}


# =========================================================
# 4. STREAMING ENDPOINT (NDJSON + DB)
# =========================================================
@app.post("/analyze-glaucoma-stream")
async def analyze_glaucoma_stream(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    async def event_generator():
        tmp_path = None
        try:
            yield json.dumps({"status": "progress", "step": 1, "message": "Image received..."}) + "\n"
            await asyncio.sleep(0.1)

            # Stream directly to temp file (fixes memory issue)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name

            image = Image.open(tmp_path).convert('RGB')

            current_dir = os.path.dirname(os.path.abspath(__file__))
            ai_dir = os.path.join(os.path.dirname(current_dir), 'ai')
            if ai_dir not in sys.path:
                sys.path.append(ai_dir)

            global glaucoma_pipeline
            if 'glaucoma_pipeline' not in globals():
                yield json.dumps({"status": "progress", "step": 2, "message": "Loading AI models..."}) + "\n"
                await asyncio.sleep(0.1)

                # Lock to prevent concurrent initialization (fixes race condition)
                with model_lock:
                    if 'glaucoma_pipeline' not in globals():
                        from pipeline import GlaucomaPipeline
                        glaucoma_pipeline = GlaucomaPipeline(
                            yolo_path=os.path.join(ai_dir, 'yolo', 'yolo-roi-v1.pt'),
                            unet_path=os.path.join(ai_dir, 'unet', 'unetpp-seg-v1.pth'),
                            device='cpu'
                        )

            yield json.dumps({"status": "progress", "step": 3, "message": "Running AI inference..."}) + "\n"
            await asyncio.sleep(0.1)
            result = await run_in_threadpool(glaucoma_pipeline.run, tmp_path)

            yield json.dumps({"status": "progress", "step": 4, "message": "Processing image masks..."}) + "\n"
            await asyncio.sleep(0.1)

            is_glaucoma, cup_to_disc_ratio, confidence = False, 0.0, 0.0
            open_cv_image = np.array(image)

            if result is not None:
                full_img, crops, masks, cdr_val, _, _ = result
                cup_to_disc_ratio = round(float(cdr_val), 2)
                is_glaucoma, confidence = bool(cup_to_disc_ratio > 0.65), 0.95

                if len(crops) > 0 and len(masks) > 0 and len(masks[0]) >= 2:
                    x1, y1, x2, y2 = crops[0]
                    roi = open_cv_image[y1:y2, x1:x2]
                    roi_h, roi_w = roi.shape[:2]
                    roi[cv2.resize(masks[0][0], (roi_w, roi_h)) > 0.5] = roi[cv2.resize(masks[0][0], (roi_w, roi_h)) > 0.5] * 0.5 + np.array([0, 255, 0]) * 0.5
                    roi[cv2.resize(masks[0][1], (roi_w, roi_h)) > 0.5] = roi[cv2.resize(masks[0][1], (roi_w, roi_h)) > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5
                    open_cv_image[y1:y2, x1:x2] = roi

            buffered = io.BytesIO()
            Image.fromarray(open_cv_image).save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            if current_user:
                db.add(History(user_id=current_user.id, is_glaucoma=is_glaucoma, cup_to_disc_ratio=cup_to_disc_ratio))
                db.commit()

            yield json.dumps({
                "status": "success", "step": 5, "message": "Analysis completed!",
                "data": {"has_glaucoma": is_glaucoma, "confidence": confidence, "cup_to_disc_ratio": cup_to_disc_ratio, "image_base64": img_base64, "saved_to_db": bool(current_user)}
            }) + "\n"

        except Exception as e:
            # Hide raw exceptions from client
            yield json.dumps({"status": "error", "message": "An internal server error occurred."}) + "\n"
        finally:
            # Always clean up temp file (fixes memory/disk leak)
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

# =========================================================
# 5. DISTRIBUTED SYSTEM DEMO (LAVINMQ / RABBITMQ)
# =========================================================

@app.post("/demo-distributed")
async def demo_distributed_system(file: UploadFile = File(...)):
    amqp_url = os.getenv("AMQP_URL")

    if not amqp_url:
        return {
            "status": "error",
            "message": "Missing AMQP_URL environment variable on Azure!"
        }

    try:
        contents = await file.read()
        image_b64 = base64.b64encode(contents).decode('utf-8')

        params = pika.URLParameters(amqp_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()

        channel.queue_declare(queue='glaucoma_queue')

        channel.basic_publish(
            exchange='',
            routing_key='glaucoma_queue',
            body=image_b64
        )
        connection.close()

        return {
            "status": "success",
            "message": "Image successfully sent to the queue! Waiting for the worker."
        }
    except Exception:
        return {"status": "error", "message": "Queue connection failed due to an internal server error."}
