#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction Service API for Trading Bot

This module provides a REST API for making predictions with trained models.
"""

import os
import json
import pickle
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Create FastAPI app
app = FastAPI(
    title="Trading Bot Prediction API",
    description="API for making predictions with trained trading models",
    version="1.0.0"
)

# Set default model directory
DEFAULT_MODEL_DIR = os.environ.get('MODEL_DIR', './output/models/latest')

# Global variables
loaded_models = {}
model_metadata = None

# Model and data schemas
class PredictionFeatures(BaseModel):
    """Input features for prediction."""
    features: Dict[str, float] = Field(..., description="Feature values as key-value pairs")
    regime: Optional[str] = Field(None, description="Market regime (if known)")
    symbol: Optional[str] = Field(None, description="Trading symbol")
    timestamp: Optional[str] = Field(None, description="Timestamp for this data point")

class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: Any = Field(..., description="Model prediction (class or value)")
    probability: Optional[float] = Field(None, description="Probability or confidence (if available)")
    regime: Optional[str] = Field(None, description="Market regime used for prediction")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Model version or run ID")
    timestamp: str = Field(..., description="Timestamp of the prediction")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Feature importance or explanation")

class ModelInfo(BaseModel):
    """Model information."""
    name: str
    version: str
    type: str
    metrics: Dict[str, Any]
    timestamp: str

def load_models(model_dir: str = DEFAULT_MODEL_DIR) -> Dict[str, Any]:
    """
    Load all models from the specified directory.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Dictionary of loaded models
    """
    global loaded_models, model_metadata
    
    # Reset loaded models
    loaded_models = {}
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
    else:
        model_metadata = {
            'run_id': 'unknown',
            'timestamp': datetime.now().isoformat(),
            'models': []
        }
    
    # Load each model from the directory
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl'):
            model_name = filename.replace('.pkl', '')
            model_path = os.path.join(model_dir, filename)
            
            try:
                with open(model_path, 'rb') as f:
                    model_package = pickle.load(f)
                    
                # Extract model and relevant information
                if isinstance(model_package, dict) and 'model' in model_package:
                    loaded_models[model_name] = model_package
                    print(f"Loaded model {model_name} from {model_path}")
                else:
                    # Direct model object
                    loaded_models[model_name] = {'model': model_package}
                    print(f"Loaded model {model_name} from {model_path}")
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
    
    return loaded_models

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load models when the API starts."""
    try:
        load_models()
        print(f"Loaded {len(loaded_models)} models successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Trading Bot Prediction API",
        "version": "1.0.0",
        "models_loaded": len(loaded_models),
        "model_version": model_metadata.get('run_id', 'unknown') if model_metadata else 'unknown',
        "status": "ready" if loaded_models else "no_models_loaded"
    }

@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get information about available models."""
    if not loaded_models:
        raise HTTPException(status_code=404, detail="No models loaded")
    
    models_info = []
    for name, model_data in loaded_models.items():
        metrics = {}
        if 'performance_metrics' in model_data:
            metrics = model_data['performance_metrics']
        elif model_metadata and 'metrics' in model_metadata:
            metrics = model_metadata.get('metrics', {})
            
        model_info = ModelInfo(
            name=name,
            version=model_metadata.get('run_id', 'unknown') if model_metadata else 'unknown',
            type=name.split('_')[0] if '_' in name else 'unknown',
            metrics=metrics,
            timestamp=model_metadata.get('timestamp', datetime.now().isoformat()) if model_metadata else datetime.now().isoformat()
        )
        models_info.append(model_info)
    
    return models_info

@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    if not loaded_models or model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model_data = loaded_models[model_name]
    metrics = {}
    if 'performance_metrics' in model_data:
        metrics = model_data['performance_metrics']
    elif model_metadata and 'metrics' in model_metadata:
        metrics = model_metadata.get('metrics', {})
        
    model_info = ModelInfo(
        name=model_name,
        version=model_metadata.get('run_id', 'unknown') if model_metadata else 'unknown',
        type=model_name.split('_')[0] if '_' in model_name else 'unknown',
        metrics=metrics,
        timestamp=model_metadata.get('timestamp', datetime.now().isoformat()) if model_metadata else datetime.now().isoformat()
    )
    
    return model_info

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PredictionFeatures, model_name: str = None):
    """
    Make a prediction using the specified or most appropriate model.
    
    Args:
        features: Input features for prediction
        model_name: Optional name of the model to use
        
    Returns:
        Prediction response with model outputs
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    # Auto-select model if not specified
    if not model_name:
        # Try to select based on regime
        if features.regime and f"regime_{features.regime}" in loaded_models:
            model_name = f"regime_{features.regime}"
        elif 'meta_model' in loaded_models:
            model_name = 'meta_model'
        elif 'primary' in loaded_models:
            model_name = 'primary'
        else:
            # Just use first available model
            model_name = list(loaded_models.keys())[0]
    
    # Check if model exists
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Get model package
    model_package = loaded_models[model_name]
    model = model_package.get('model')
    
    # Convert features to DataFrame for prediction
    feature_dict = features.features
    feature_df = pd.DataFrame([feature_dict])
    
    # Make prediction
    try:
        prediction = model.predict(feature_df)[0]
        
        # Try to get probability if applicable
        probability = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(feature_df)[0]
                probability = float(proba.max())
            except:
                pass
        
        # Try to get feature importance/explanation
        explanation = None
        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            importances = model.feature_importances_
            explanation = {
                'feature_importance': dict(zip(feature_dict.keys(), importances))
            }
        
        # Create response
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            regime=features.regime,
            model_name=model_name,
            model_version=model_metadata.get('run_id', 'unknown') if model_metadata else 'unknown',
            timestamp=datetime.now().isoformat(),
            explanation=explanation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/ensemble", response_model=PredictionResponse)
async def predict_ensemble(features: PredictionFeatures):
    """
    Make a prediction using the ensemble model, which combines predictions from
    multiple regime-specific models.
    
    Args:
        features: Input features for prediction
        
    Returns:
        Prediction response with ensemble model outputs
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    # Check if we have meta model
    if 'meta_model' not in loaded_models:
        raise HTTPException(status_code=404, detail="Ensemble meta-model not found")
    
    # Get regime models
    regime_models = {name: model for name, model in loaded_models.items() 
                    if name.startswith('regime_')}
    
    if not regime_models:
        raise HTTPException(status_code=404, detail="No regime models found for ensemble")
    
    # Convert features to DataFrame
    feature_dict = features.features
    feature_df = pd.DataFrame([feature_dict])
    
    # Make predictions with each regime model
    ensemble_features = {}
    for name, model_package in regime_models.items():
        model = model_package.get('model')
        try:
            # Get prediction
            pred = model.predict(feature_df)[0]
            ensemble_features[f"{name}_pred"] = pred
            
            # Try to get probability if applicable
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_df)[0]
                for i, p in enumerate(proba):
                    ensemble_features[f"{name}_prob_{i}"] = p
        except:
            pass
    
    # Create ensemble feature DataFrame
    ensemble_df = pd.DataFrame([ensemble_features])
    
    # If we have regime column, add it
    if features.regime:
        ensemble_df['regime'] = features.regime
    
    # Make prediction with meta-model
    meta_model = loaded_models['meta_model'].get('model')
    try:
        prediction = meta_model.predict(ensemble_df)[0]
        
        # Try to get probability if applicable
        probability = None
        if hasattr(meta_model, 'predict_proba'):
            try:
                proba = meta_model.predict_proba(ensemble_df)[0]
                probability = float(proba.max())
            except:
                pass
        
        # Create response
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            regime=features.regime,
            model_name='ensemble',
            model_version=model_metadata.get('run_id', 'unknown') if model_metadata else 'unknown',
            timestamp=datetime.now().isoformat(),
            explanation={'ensemble_features': ensemble_features}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble prediction error: {str(e)}")

@app.post("/reload")
async def reload_models(model_dir: str = DEFAULT_MODEL_DIR):
    """
    Reload models from the specified directory.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Status message
    """
    try:
        models = load_models(model_dir)
        return {
            "status": "success",
            "message": f"Reloaded {len(models)} models from {model_dir}",
            "models": list(models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload models: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 