import os
import cv2
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction import extract_crack_features
from dataset import get_all_pairs

def run_eda(output_dir='logs/eda'):
    os.makedirs(output_dir, exist_ok=True)
    all_pairs = get_all_pairs()
    print(f'Running EDA across {len(all_pairs)} dataset pairs...')

    records = []
    for idx, (img_path, mask_path) in enumerate(all_pairs):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        feats = extract_crack_features(mask)

        records.append({
            'filename': os.path.basename(img_path),
            'has_crack': feats['has_crack'],
            'crack_pixel_count': feats['crack_pixel_count'],
            'affected_area_pct': feats['affected_area_pct'],
            'length_px': feats['length_px'],
            'max_width_px': feats['max_width_px'],
            'mean_width_px': feats['mean_width_px'],
            'component_count': feats['component_count'],
            'branch_points': feats['branch_points']
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(all_pairs):
            print(f'Processed [{idx + 1}/{len(all_pairs)}] images')

    df = pd.DataFrame(records)
    dataset_csv = os.path.join(output_dir, 'dataset_features.csv')
    df.to_csv(dataset_csv, index=False)
    print(f'Extracted features saved to {dataset_csv}')

    # Summary Statistics
    num_cols = ['affected_area_pct', 'length_px', 'max_width_px', 'mean_width_px', 'component_count', 'branch_points']
    stats_df = df[num_cols].describe(percentiles=[0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95]).T
    stats_path = os.path.join(output_dir, 'feature_stats.csv')
    stats_df.to_csv(stats_path)
    print(f'Feature summary statistics saved to {stats_path}')

    # Plot Styling
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    # Individual distributions
    for col in ['length_px', 'max_width_px', 'affected_area_pct', 'branch_points']:
        plt.figure(figsize=(7, 4.5))
        sns.histplot(df[col], kde=True, color='#1f77b4', bins=30)
        p33 = df[col].quantile(0.33)
        p66 = df[col].quantile(0.66)
        plt.axvline(p33, color='goldenrod', linestyle='--', label=f'33rd percentile ({p33:.1f})')
        plt.axvline(p66, color='crimson', linestyle='--', label=f'66th percentile ({p66:.1f})')
        plt.title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        plt.xlabel(col, fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'), dpi=150)
        plt.close()

    # Correlation Matrix
    plt.figure(figsize=(8, 6.5))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', cbar=True, square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Crack Morphological Features', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    print('Distribution plots and correlation heatmap generated.')

    return df, stats_df

if __name__ == '__main__':
    run_eda()
