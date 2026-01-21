"""
Analyze dataset distribution and imbalances
"""
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def analyze_dataset(data_dir='../data/processed_split'):
    """Analyze dataset distribution across splits"""
    base_path = Path(data_dir)
    
    results = {}
    
    for split in ['train', 'val', 'test']:
        split_path = base_path / split
        results[split] = {}
        
        for category in ['leaf', 'fruit']:
            category_path = split_path / category
            results[split][category] = {}
            
            total_images = 0
            for disease_dir in category_path.iterdir():
                if disease_dir.is_dir():
                    images = list(disease_dir.glob('*.jpg')) + \
                             list(disease_dir.glob('*.jpeg')) + \
                             list(disease_dir.glob('*.png'))
                    
                    count = len(images)
                    results[split][category][disease_dir.name] = count
                    total_images += count
            
            results[split][category]['_total'] = total_images
    
    # Print analysis
    print("="*80)
    print("DATASET ANALYSIS REPORT")
    print("="*80)
    
    for category in ['leaf', 'fruit']:
        print(f"\n{category.upper()} CATEGORY:")
        print("-"*40)
        
        all_classes = set()
        for split in ['train', 'val', 'test']:
            all_classes.update(results[split][category].keys())
        
        all_classes = sorted([c for c in all_classes if not c.startswith('_')])
        
        # Print header
        print(f"{'Class':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8} {'%':>6}")
        print("-"*80)
        
        for cls in all_classes:
            train = results['train'][category].get(cls, 0)
            val = results['val'][category].get(cls, 0)
            test = results['test'][category].get(cls, 0)
            total = train + val + test
            
            # Calculate percentage of total
            cat_total = results['train'][category]['_total'] + \
                       results['val'][category]['_total'] + \
                       results['test'][category]['_total']
            percentage = (total / cat_total * 100) if cat_total > 0 else 0
            
            print(f"{cls:<25} {train:>8} {val:>8} {test:>8} {total:>8} {percentage:>6.1f}%")
        
        # Print totals
        train_total = results['train'][category]['_total']
        val_total = results['val'][category]['_total']
        test_total = results['test'][category]['_total']
        grand_total = train_total + val_total + test_total
        
        print("-"*80)
        print(f"{'TOTAL':<25} {train_total:>8} {val_total:>8} {test_total:>8} {grand_total:>8}")
    
    # Calculate imbalance ratios
    print("\n" + "="*80)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*80)
    
    for category in ['leaf', 'fruit']:
        print(f"\n{category.upper()} IMBALANCE:")
        
        # Get training set counts for imbalance calculation
        train_counts = {}
        for cls, count in results['train'][category].items():
            if not cls.startswith('_'):
                train_counts[cls] = count
        
        if train_counts:
            max_count = max(train_counts.values())
            min_count = min(train_counts.values())
            
            print(f"  Maximum class: {max(train_counts.items(), key=lambda x: x[1])}")
            print(f"  Minimum class: {min(train_counts.items(), key=lambda x: x[1])}")
            print(f"  Imbalance ratio: {max_count/min_count:.1f}:1")
            
            # Suggested class weights
            print("  Suggested class weights:")
            for cls, count in sorted(train_counts.items()):
                weight = max_count / count if count > 0 else 1.0
                weight = min(weight, 10.0)  # Cap at 10x
                print(f"    {cls:<20}: {weight:.2f}")
    
    # Save results to JSON
    with open('../models/dataset_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Analysis saved to: ../models/dataset_analysis.json")
    
    return results

def plot_distribution(data_dir='../data/processed_split'):
    """Create visualization of dataset distribution"""
    base_path = Path(data_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Dataset Distribution Analysis', fontsize=16, fontweight='bold')
    
    for idx, split in enumerate(['train', 'val', 'test']):
        for jdx, category in enumerate(['leaf', 'fruit']):
            split_path = base_path / split / category
            
            counts = {}
            for disease_dir in split_path.iterdir():
                if disease_dir.is_dir():
                    images = list(disease_dir.glob('*.jpg')) + \
                             list(disease_dir.glob('*.jpeg')) + \
                             list(disease_dir.glob('*.png'))
                    counts[disease_dir.name] = len(images)
            
            if counts:
                ax = axes[jdx, idx]
                classes = list(counts.keys())
                values = list(counts.values())
                
                bars = ax.bar(classes, values)
                ax.set_title(f'{category.title()} - {split.title()}', fontweight='bold')
                ax.set_ylabel('Number of Images')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../models/dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Distribution plot saved to: ../models/dataset_distribution.png")

if __name__ == "__main__":
    # Create analysis directory
    analysis_dir = Path('../models')
    analysis_dir.mkdir(exist_ok=True)
    
    print("Analyzing dataset distribution...")
    results = analyze_dataset()
    plot_distribution()