import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json
from datetime import datetime
from sklearn.utils import class_weight

# ============ ADD THIS FUNCTION ============
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.float16):
        return float(obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return obj
# ============ END OF NEW FUNCTION ============

def create_leaf_model(num_classes=5):  # Changed from 4 to 5 (4 diseases + healthy)
    """Create EfficientNetB0 model for leaf diseases"""
    # Load pre-trained EfficientNetB0
    base_model = applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Create new model on top
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_leaf_model(data_dir='../data/processed_split', model_save_path='../models/leaf_model.h5'):
    """Train leaf disease classification model with class weights"""
    
    # Paths
    train_dir = os.path.join(data_dir, 'train', 'leaf')
    val_dir = os.path.join(data_dir, 'val', 'leaf')
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Data generators
    batch_size = 32
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Save class indices
    class_indices = train_generator.class_indices
    with open('../models/leaf_class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    
    print("="*60)
    print("LEAF MODEL TRAINING")
    print("="*60)
    print("Class indices:", class_indices)
    print("Training samples:", train_generator.samples)
    print("Validation samples:", val_generator.samples)
    
    # Calculate class weights for imbalance
    print("\nCalculating class weights...")
    class_weights_dict = {}
    
    # Get counts for each class
    counts = []
    for class_name in sorted(class_indices.keys()):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(images)
            counts.append(count)
            print(f"  {class_name}: {count} samples")
    
    if len(counts) > 0:
        # Calculate weights (inverse frequency)
        max_count = max(counts)
        for idx, (class_name, class_idx) in enumerate(class_indices.items()):
            weight = max_count / counts[idx] if counts[idx] > 0 else 1.0
            # Cap weights to avoid extreme values
            weight = min(weight, 10.0)  # Maximum 10x weight
            class_weights_dict[class_idx] = weight
            print(f"  Weight for {class_name}: {weight:.2f}")
    
    print("\n" + "-"*60)
    
    # Create model
    model, base_model = create_leaf_model(num_classes=len(class_indices))
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_save_path.replace('.h5', '_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.CSVLogger(
            filename='../models/leaf_training_log.csv',
            separator=',',
            append=False
        )
    ]
    
    # Initial training with frozen base model
    print("Phase 1: Training top layers...")
    history1 = model.fit(
        train_generator,
        epochs=25,
        validation_data=val_generator,
        callbacks=callbacks_list,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Fine-tuning: Unfreeze some layers
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nPhase 2: Fine-tuning...")
    history2 = model.fit(
        train_generator,
        epochs=15,
        validation_data=val_generator,
        callbacks=callbacks_list,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Save final model
    model.save(model_save_path)
    print(f"\n✅ Model saved to {model_save_path}")
    
    # ============ UPDATED CODE HERE ============
    # Save training history WITH SERIALIZATION FIX
    history = {
        'phase1': convert_to_serializable(history1.history),
        'phase2': convert_to_serializable(history2.history),
        'timestamp': datetime.now().isoformat(),
        'class_indices': class_indices,
        'class_weights': convert_to_serializable(class_weights_dict)
    }
    
    with open('../models/leaf_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    # ============ END OF UPDATED CODE ============
    
    print("✅ Training history saved")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL VALIDATION METRICS")
    print("="*60)
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    return model

if __name__ == "__main__":
    train_leaf_model()