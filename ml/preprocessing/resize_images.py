import os
from PIL import Image
import argparse
from pathlib import Path

def resize_images(input_dir, output_dir, target_size=(224, 224)):
    """Resize all images to target size for EfficientNetB0"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    for disease_dir in input_path.iterdir():
        if disease_dir.is_dir():
            output_disease_dir = output_path / disease_dir.relative_to(input_path)
            output_disease_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in disease_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    try:
                        # Open and resize image
                        img = Image.open(img_file)
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save resized image
                        output_file = output_disease_dir / img_file.name
                        img.save(output_file)
                        print(f"Resized: {img_file} -> {output_file}")
                        
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Resize images for EfficientNetB0')
    parser.add_argument('--input', required=True, help='Input directory with raw images')
    parser.add_argument('--output', required=True, help='Output directory for resized images')
    
    args = parser.parse_args()
    
    print("Starting image resizing...")
    resize_images(args.input, args.output)
    print("Image resizing completed!")

if __name__ == "__main__":
    main()