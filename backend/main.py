from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from typing import Optional
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
import uuid

# =========================================================
# 1. DATABASE CONFIGURATION (SQLite)
# =========================================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./glaucoma_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, nullable=True)
    avatar_kind = Column(String)
    avatar_color = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class History(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    is_glaucoma = Column(Boolean)
    cup_to_disc_ratio = Column(Float)
    confidence = Column(Float, nullable=True)
    image_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================================================
# 2. IDENTITY, JWT & SCHEMAS
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
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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

# Pydantic Schemas for Patients
class PatientCreate(BaseModel):
    first_name: str
    last_name: str
    email: Optional[str] = None
    avatar_kind: str
    avatar_color: str

class PatientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    avatar_kind: Optional[str] = None
    avatar_color: Optional[str] = None

app = FastAPI()
model_lock = Lock() 

os.makedirs("static/uploads", exist_ok=True)

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


# =========================================================
# 4. PATIENTS CRUD ENDPOINTS
# =========================================================
@app.post("/patients")
def create_patient(patient: PatientCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    new_patient = Patient(
        owner_id=current_user.id,
        first_name=patient.first_name,
        last_name=patient.last_name,
        email=patient.email,
        avatar_kind=patient.avatar_kind,
        avatar_color=patient.avatar_color
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return new_patient

@app.get("/patients")
def list_patients(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return db.query(Patient).filter(Patient.owner_id == current_user.id).all()

@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.owner_id == current_user.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@app.put("/patients/{patient_id}")
def update_patient(patient_id: int, updates: PatientUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.owner_id == current_user.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    update_data = updates.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(patient, key, value)
        
    db.commit()
    db.refresh(patient)
    return patient

@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.owner_id == current_user.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    db.query(History).filter(History.patient_id == patient_id).delete()
    db.delete(patient)
    db.commit()
    return {"message": "Patient and related history deleted successfully"}


# =========================================================
# 5. HISTORY & STATIC ENDPOINTS
# =========================================================
@app.get("/history")
def get_user_history(patient_id: Optional[int] = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="You must be logged in to view history")
    
    query = db.query(History).filter(History.user_id == current_user.id)
    if patient_id:
        query = query.filter(History.patient_id == patient_id)
        
    records = query.all()
    return {"history": [
        {
            "id": r.id, 
            "patient_id": r.patient_id,
            "is_glaucoma": r.is_glaucoma, 
            "cdr": r.cup_to_disc_ratio, 
            "confidence": r.confidence,
            "image_url": r.image_url,
            "date": r.created_at
        } for r in records
    ]}

@app.get("/static/uploads/{filename}")
def get_uploaded_image(filename: str, current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    file_path = os.path.join("static", "uploads", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(file_path)

# =========================================================
# 6. STREAMING ENDPOINT (NDJSON + DB + PATIENT + IMAGE SAVE)
# =========================================================
@app.post("/analyze-glaucoma-stream")
async def analyze_glaucoma_stream(
    file: UploadFile = File(...), 
    patient_id: Optional[int] = Form(None), 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    domain = os.getenv("WEBSITE_HOSTNAME", "127.0.0.1:8000")
    protocol = "https" if "azure" in domain.lower() else "http"
    base_url = f"{protocol}://{domain}"

    async def event_generator():
        tmp_path = None
        try:
            yield json.dumps({"status": "progress", "step": 1, "message": "Image received..."}) + "\n"
            await asyncio.sleep(0.1)

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

                with model_lock:
                    if 'glaucoma_pipeline' not in globals():
                        from pipeline import GlaucomaPipeline
                        glaucoma_pipeline = GlaucomaPipeline(
                            yolo_path=os.path.join(ai_dir, 'yolo', 'yolo-roi-v1.pt'),
                            unet_path=os.path.join(ai_dir, 'unet', 'unetpp_smdg_v3.pth'),
                            device='cpu'
                        )

            yield json.dumps({"status": "progress", "step": 3, "message": "Running AI inference..."}) + "\n"
            await asyncio.sleep(0.1)
            result = await run_in_threadpool(glaucoma_pipeline.run, tmp_path)

            yield json.dumps({"status": "progress", "step": 4, "message": "Processing image masks..."}) + "\n"
            await asyncio.sleep(0.1)

            is_glaucoma, cup_to_disc_ratio, confidence = False, 0.0, 0.0
            open_cv_image = np.array(image)
            image_url = None

            if result is not None:
                full_img, crops, masks, cdr_val, _, _ = result
                cup_to_disc_ratio = round(float(cdr_val), 2)
                is_glaucoma, confidence = bool(cup_to_disc_ratio > 0.65), 0.95

                if len(crops) > 0 and len(masks) > 0 and len(masks[0]) >= 2:
                    x1, y1, x2, y2 = crops[0]
                    roi = open_cv_image[y1:y2, x1:x2]
                    roi_h, roi_w = roi.shape[:2]
                    
                    mask_disc = cv2.resize(masks[0][0], (roi_w, roi_h)) > 0.5
                    mask_cup = cv2.resize(masks[0][1], (roi_w, roi_h)) > 0.5
                    
                    roi[mask_disc] = roi[mask_disc] * 0.5 + np.array([0, 255, 0]) * 0.5
                    roi[mask_cup] = roi[mask_cup] * 0.5 + np.array([255, 0, 0]) * 0.5
                    open_cv_image[y1:y2, x1:x2] = roi

            buffered = io.BytesIO()
            final_img = Image.fromarray(open_cv_image)
            await run_in_threadpool(final_img.save, buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            if current_user:
                if patient_id:
                    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.owner_id == current_user.id).first()
                    if not patient:
                        yield json.dumps({"status": "error", "message": "Unauthorized or missing patient."}) + "\n"
                        return

                saved_filename = f"{uuid.uuid4()}.jpg"
                save_path = os.path.join("static", "uploads", saved_filename)
                await run_in_threadpool(final_img.save, save_path, format="JPEG")
                image_url = f"{base_url}/static/uploads/{saved_filename}"

                db.add(History(
                    user_id=current_user.id, 
                    patient_id=patient_id, 
                    is_glaucoma=is_glaucoma, 
                    cup_to_disc_ratio=cup_to_disc_ratio,
                    confidence=confidence,
                    image_url=image_url
                ))
                db.commit()

            yield json.dumps({
                "status": "success", "step": 5, "message": "Analysis completed!",
                "data": {
                    "has_glaucoma": is_glaucoma, 
                    "confidence": confidence, 
                    "cup_to_disc_ratio": cup_to_disc_ratio, 
                    "image_base64": img_base64, 
                    "image_url": image_url,
                    "saved_to_db": bool(current_user)
                }
            }) + "\n"

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(error_trace)
            yield json.dumps({"status": "error", "message": "An internal server error occurred."}) + "\n"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                await run_in_threadpool(os.remove, tmp_path)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

# =========================================================
# 7. DISTRIBUTED SYSTEM DEMO (LAVINMQ / RABBITMQ)
# =========================================================
@app.post("/demo-distributed")
async def demo_distributed_system(file: UploadFile = File(...)):
    amqp_url = os.getenv("AMQP_URL")

    if not amqp_url:
        return {"status": "error", "message": "Missing AMQP_URL environment variable on Azure!"}

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