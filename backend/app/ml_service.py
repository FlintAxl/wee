import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import json
import os
from typing import Tuple, List, Dict, Any
from .recommendations import get_recommendations

class TomatoGuardMLService:
    """ML service for tomato disease detection"""
    
    def __init__(self):
        self.leaf_model = None
        self.fruit_model = None
        self.leaf_classes = None
        self.fruit_classes = None
        
        # Define expected classes for your dataset
        self.expected_leaf_classes = [
            'bacterial_spot_speck',
            'early_blight',
            'healthy',
            'late_blight',
            'septoria_leaf_spot'
        ]
        
        self.expected_fruit_classes = [
            'anthracnose',
            'blossom_end_rot',
            'buckeye_rot',
            'gray_mold',
            'healthy'
        ]
        
        self.load_models()
    
    def load_models(self):
        """Load trained models and class mappings"""
        try:
            # Load leaf model
            leaf_model_path = os.path.join(os.path.dirname(__file__), 
                                          '..', '..', 'ml', 'models', 'leaf_model.h5')
            if os.path.exists(leaf_model_path):
                self.leaf_model = keras.models.load_model(leaf_model_path)
                print("✅ Leaf model loaded successfully")
            else:
                print("⚠️ Leaf model not found at:", leaf_model_path)
            
            # Load fruit model
            fruit_model_path = os.path.join(os.path.dirname(__file__), 
                                           '..', '..', 'ml', 'models', 'fruit_model.h5')
            if os.path.exists(fruit_model_path):
                self.fruit_model = keras.models.load_model(fruit_model_path)
                print("✅ Fruit model loaded successfully")
            else:
                print("⚠️ Fruit model not found at:", fruit_model_path)
            
            # Load or create class indices
            leaf_indices_path = os.path.join(os.path.dirname(__file__), 
                                           '..', '..', 'ml', 'models', 'leaf_class_indices.json')
            if os.path.exists(leaf_indices_path):
                with open(leaf_indices_path, 'r') as f:
                    leaf_indices = json.load(f)
                    self.leaf_classes = {v: k for k, v in leaf_indices.items()}
                    print(f"✅ Leaf classes loaded: {list(self.leaf_classes.values())}")
            else:
                # Create default indices if file doesn't exist
                print("⚠️ Leaf class indices not found, using defaults")
                self.leaf_classes = {i: cls for i, cls in enumerate(self.expected_leaf_classes)}
            
            fruit_indices_path = os.path.join(os.path.dirname(__file__), 
                                            '..', '..', 'ml', 'models', 'fruit_class_indices.json')
            if os.path.exists(fruit_indices_path):
                with open(fruit_indices_path, 'r') as f:
                    fruit_indices = json.load(f)
                    self.fruit_classes = {v: k for k, v in fruit_indices.items()}
                    print(f"✅ Fruit classes loaded: {list(self.fruit_classes.values())}")
            else:
                # Create default indices if file doesn't exist
                print("⚠️ Fruit class indices not found, using defaults")
                self.fruit_classes = {i: cls for i, cls in enumerate(self.expected_fruit_classes)}
                    
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for EfficientNetB0"""
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize to 224x224
        image = image.resize((224, 224))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict_leaf_disease(self, image_bytes: bytes) -> Dict[str, Any]:
        """Predict leaf disease from image"""
        if self.leaf_model is None or self.leaf_classes is None:
            return {"error": "Leaf model not loaded"}
        
        try:
            # Preprocess image
            image_array = self.preprocess_image(image_bytes)
            
            # Make prediction
            predictions = self.leaf_model.predict(image_array, verbose=0)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get disease name
            disease_name = self.leaf_classes.get(predicted_class_idx, "unknown")
            
            # Get all probabilities
            all_probs = {}
            for idx, prob in enumerate(predictions[0]):
                class_name = self.leaf_classes.get(idx, f"class_{idx}")
                all_probs[class_name] = float(prob)
            
            # Get recommendations
            recommendations = get_recommendations(disease_name)
            
            return {
                "type": "leaf",
                "disease": disease_name,
                "confidence": confidence,
                "display_name": recommendations["name"],
                "severity": recommendations["severity"],
                "recommendations": recommendations["recommendations"],
                "prevention": recommendations["prevention"],
                "organic_control": recommendations["organic_control"],
                "all_probabilities": all_probs
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_fruit_disease(self, image_bytes: bytes) -> Dict[str, Any]:
        """Predict fruit disease from image"""
        if self.fruit_model is None or self.fruit_classes is None:
            return {"error": "Fruit model not loaded"}
        
        try:
            # Preprocess image
            image_array = self.preprocess_image(image_bytes)
            
            # Make prediction
            predictions = self.fruit_model.predict(image_array, verbose=0)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get disease name
            disease_name = self.fruit_classes.get(predicted_class_idx, "unknown")
            
            # Get all probabilities
            all_probs = {}
            for idx, prob in enumerate(predictions[0]):
                class_name = self.fruit_classes.get(idx, f"class_{idx}")
                all_probs[class_name] = float(prob)
            
            # Get recommendations
            recommendations = get_recommendations(disease_name)
            
            return {
                "type": "fruit",
                "disease": disease_name,
                "confidence": confidence,
                "display_name": recommendations["name"],
                "severity": recommendations["severity"],
                "recommendations": recommendations["recommendations"],
                "prevention": recommendations["prevention"],
                "organic_control": recommendations["organic_control"],
                "all_probabilities": all_probs
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_with_auto_detect(self, image_bytes: bytes) -> Dict[str, Any]:
        """Automatically detect if image is leaf or fruit and predict"""
        # For now, run both models and return higher confidence result
        leaf_result = self.predict_leaf_disease(image_bytes)
        fruit_result = self.predict_fruit_disease(image_bytes)
        
        if "error" in leaf_result and "error" in fruit_result:
            return {"error": "Both models failed"}
        elif "error" in leaf_result:
            return fruit_result
        elif "error" in fruit_result:
            return leaf_result
        else:
            # Return result with higher confidence
            if leaf_result["confidence"] > fruit_result["confidence"]:
                return leaf_result
            else:
                return fruit_result

# Global instance
ml_service = TomatoGuardMLService()