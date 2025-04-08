from fastapi import FastAPI, UploadFile, File, Form
import shutil
from predict_model import predict
import os
from train_model import train

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API de classification image+texte active"}

@app.post("/predict")
async def predict_product(description: str = Form(...), file: UploadFile = File(...)):
    file_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = predict(description, file_path)
    os.remove(file_path)
    return {"predicted_prdtypecode": result}

@app.post("/training")
def train_model():
    train()
    return {"message": "Modèle réentraîné avec succès"}