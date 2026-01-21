import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

def split_dataset(input_dir, output_dir, test_size=0.2, val_size=0.15, random_state=42):
    """Split dataset into train/val/test with stratification - FIXED VERSION"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Process each category (leaf and fruit)
    for category_dir in input_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name  # 'leaf' or 'fruit'
            print(f"\nProcessing category: {category_name}")
            
            # Create category directories in output
            for split in splits:
                (output_path / split / category_name).mkdir(parents=True, exist_ok=True)
            
            # Process each disease within this category
            for disease_dir in category_dir.iterdir():
                if disease_dir.is_dir():
                    disease_name = disease_dir.name
                    
                    # Get all images for this disease
                    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
                    images = []
                    for ext in image_extensions:
                        images.extend(disease_dir.glob(ext))
                    
                    if len(images) == 0:
                        print(f"  Warning: {disease_name} has no images in {disease_dir}")
                        continue
                    
                    print(f"  {disease_name}: {len(images)} images found")
                    
                    # Create labels array for stratification
                    labels = [disease_name] * len(images)
                    
                    # First split: test set
                    train_val, test, train_val_labels, test_labels = train_test_split(
                        images, 
                        labels,
                        test_size=test_size, 
                        random_state=random_state,
                        stratify=labels
                    )
                    
                    # Second split: train and validation
                    train, val, train_labels, val_labels = train_test_split(
                        train_val,
                        train_val_labels,
                        test_size=val_size/(1-test_size),  # Adjust for already removed test
                        random_state=random_state,
                        stratify=train_val_labels
                    )
                    
                    # Copy images to respective directories
                    for img_list, split_name in zip([train, val, test], ['train', 'val', 'test']):
                        # Create disease directory within category
                        split_disease_dir = output_path / split_name / category_name / disease_name
                        split_disease_dir.mkdir(parents=True, exist_ok=True)
                        
                        for img_path in img_list:
                            shutil.copy2(img_path, split_disease_dir / img_path.name)
                    
                    print(f"    Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test - FIXED VERSION')
    parser.add_argument('--input', required=True, help='Input directory with processed images')
    parser.add_argument('--output', required=True, help='Output directory for split dataset')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation set proportion')
    
    args = parser.parse_args()
    
    print("="*50)
    print("SPLITTING DATASET - FIXED VERSION")
    print("="*50)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Test size: {args.test_size}")
    print(f"Validation size: {args.val_size}")
    print("="*50)
    
    split_dataset(args.input, args.output, args.test_size, args.val_size)
    print("\n" + "="*50)
    print("Dataset splitting completed successfully!")
    print("="*50)
    
    # Show summary
    output_path = Path(args.output)
    print("\nFINAL DIRECTORY STRUCTURE:")
    for split in ['train', 'val', 'test']:
        split_path = output_path / split
        if split_path.exists():
            print(f"\n{split.upper()}:")
            for category in ['leaf', 'fruit']:
                category_path = split_path / category
                if category_path.exists():
                    print(f"  {category}:")
                    for disease_dir in category_path.iterdir():
                        if disease_dir.is_dir():
                            image_count = len(list(disease_dir.glob('*.jpg'))) + \
                                         len(list(disease_dir.glob('*.jpeg'))) + \
                                         len(list(disease_dir.glob('*.png')))
                            print(f"    {disease_dir.name}: {image_count} images")

if __name__ == "__main__":
    main()