from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
from .ml_service import ml_service

app = FastAPI(
    title="TomatoGuard API",
    description="Tomato Disease Detection System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "TomatoGuard API",
        "version": "1.0.0",
        "endpoints": [
            "/predict/leaf - POST: Predict leaf disease",
            "/predict/fruit - POST: Predict fruit disease",
            "/predict/auto - POST: Auto-detect and predict",
            "/health - GET: Check API health"
        ]
    }

@app.get("/health")
async def health_check():
    models_loaded = {
        "leaf_model": ml_service.leaf_model is not None,
        "fruit_model": ml_service.fruit_model is not None
    }
    return {
        "status": "healthy" if all(models_loaded.values()) else "partial",
        "models_loaded": models_loaded
    }

@app.post("/predict/leaf")
async def predict_leaf_disease(file: UploadFile = File(...)):
    """Predict tomato leaf disease from uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Predict
        result = ml_service.predict_leaf_disease(image_bytes)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/fruit")
async def predict_fruit_disease(file: UploadFile = File(...)):
    """Predict tomato fruit disease from uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Predict
        result = ml_service.predict_fruit_disease(image_bytes)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/auto")
async def predict_auto_detect(file: UploadFile = File(...)):
    """Auto-detect leaf or fruit and predict disease"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Predict with auto-detection
        result = ml_service.predict_with_auto_detect(image_bytes)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict diseases for multiple images"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "error": "File must be an image"
            })
            continue
        
        try:
            image_bytes = await file.read()
            result = ml_service.predict_with_auto_detect(image_bytes)
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "total_images": len(files),
        "predictions": results
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)