"""
QUICK AUGMENTATION FOR MINORITY CLASSES - 10 minutes
"""
import tensorflow as tf
import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path

def augment_minority_classes():
    """Augment buckeye_rot and bacterial_spot_speck"""
    
    print("="*70)
    print("🚀 AUGMENTING MINORITY CLASSES")
    print("="*70)
    
    # TARGET: Get to 500 images each
    TARGETS = {
        'fruit/buckeye_rot': 500,      # Currently: ~74 total
        'leaf/bacterial_spot_speck': 500  # Currently: ~168 total
    }
    
    for class_path, target in TARGETS.items():
        print(f"\n📈 Augmenting {class_path} to {target} images...")
        
        # Find all existing images
        source_dir = Path(f'../data/processed/{class_path}')
        if not source_dir.exists():
            print(f"  ⚠️  Not found: {source_dir}")
            continue
        
        # Load images
        images = []
        for f in os.listdir(source_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(source_dir / f)
                img = img.resize((224, 224))
                images.append(np.array(img))
        
        current = len(images)
        needed = target - current
        
        if needed <= 0:
            print(f"  ✅ Already has {current} images")
            continue
        
        print(f"  Current: {current}, Need: {needed} more")
        
        # Create augmented versions directory
        aug_dir = Path(f'../data/processed_augmented/{class_path}')
        aug_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy originals first
        for i, img_array in enumerate(images[:50]):  # Use first 50 only
            img = Image.fromarray(img_array)
            img.save(aug_dir / f'original_{i:03d}.jpg')
        
        # Create data generator for augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )
        
        # Convert to numpy array
        images_np = np.array(images[:50])  # Use 50 originals
        
        # Generate augmented images
        batch_size = 10
        total_generated = 0
        
        # Fit the generator
        datagen.fit(images_np)
        
        # Generate in batches
        for batch in datagen.flow(
            images_np,
            batch_size=batch_size,
            save_to_dir=aug_dir,
            save_prefix='aug_',
            save_format='jpg'
        ):
            total_generated += batch_size
            if total_generated >= needed:
                break
        
        print(f"  ✅ Generated {total_generated} augmented images")
        print(f"  📁 Saved to: {aug_dir}")
    
    print("\n" + "="*70)
    print("🎉 AUGMENTATION COMPLETE!")
    print("="*70)
    
    # Update the splits
    print("\n📋 NEXT: Run update_splits.py to add augmented images to dataset")

if __name__ == "__main__":
    augment_minority_classes()