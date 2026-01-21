import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd

def evaluate_leaf_model(model_path='../models/leaf_model.h5', data_dir='../data/processed_split'):
    """Evaluate leaf disease classification model - UPDATED"""
    
    # Expected classes for your dataset
    expected_classes = [
        'bacterial_spot_speck',
        'early_blight',
        'healthy',
        'late_blight',
        'septoria_leaf_spot'
    ]
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print(f"   Expected path: {os.path.abspath(model_path)}")
        return None
    
    model = keras.models.load_model(model_path)
    
    
    # Load class indices
    with open('../models/leaf_class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Test data generator
    test_dir = os.path.join(data_dir, 'test', 'leaf')
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # Important for evaluation
    )
    
    # Get predictions
    print("Generating predictions...")
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_classes, average='weighted'
    )
    
    # Classification report
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(
        y_true, y_pred_classes, 
        target_names=target_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Print results
    print("\n" + "="*50)
    print("LEAF MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Test Samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(pd.DataFrame(report).transpose())
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Leaf Diseases - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('../models/leaf_confusion_matrix.png', dpi=300)
    plt.show()
    
    # Per-class metrics
    print("\n" + "="*50)
    print("PER-CLASS METRICS")
    print("="*50)
    
    per_class_metrics = []
    for i, class_name in enumerate(target_names):
        class_idx = class_indices[class_name]
        class_mask = y_true == class_idx
        
        if np.sum(class_mask) > 0:
            class_accuracy = accuracy_score(
                y_true[class_mask], 
                y_pred_classes[class_mask]
            )
            
            class_precision = report[class_name]['precision']
            class_recall = report[class_name]['recall']
            class_f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            
            per_class_metrics.append({
                'Class': class_name,
                'Accuracy': class_accuracy,
                'Precision': class_precision,
                'Recall': class_recall,
                'F1-Score': class_f1,
                'Support': support
            })
    
    per_class_df = pd.DataFrame(per_class_metrics)
    print(per_class_df.to_string(index=False))
    
    # Save detailed results
    results = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'test_samples': int(len(y_true))
        },
        'per_class': report,
        'confusion_matrix': cm.tolist()
    }
    
    with open('../models/leaf_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: ../models/leaf_evaluation_results.json")
    print(f"Confusion matrix saved to: ../models/leaf_confusion_matrix.png")
    
    return results

if __name__ == "__main__":
    evaluate_leaf_model()