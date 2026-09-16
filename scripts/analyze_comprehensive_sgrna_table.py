#!/usr/bin/env python
"""
Analyze the comprehensive sgRNA table and generate summary statistics.

This script analyzes:
1. sgRNA sequence features vs activity
2. Mismatch position effects on activity
3. Distance to PAM vs activity
4. Model embedding correlations with experimental activity
5. Dataset-specific statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def load_data(input_path: Path) -> pd.DataFrame:
    """Load comprehensive sgRNA table."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} guides")
    return df


def basic_statistics(df: pd.DataFrame):
    """Print basic statistics."""
    print("\n" + "="*70)
    print("BASIC STATISTICS")
    print("="*70)
    
    print(f"\nTotal guides: {len(df)}")
    print(f"\nBy dataset:")
    print(df['dataset'].value_counts())
    
    print(f"\nData completeness:")
    key_cols = [
        'sgrna_sequence', 'target_genomic_sequence', 'pam_sequence',
        'n_mismatches', 'embedding_norm', 'virtual_scale',
        'mean_relative_gamma', 'relative_activity_mean'
    ]
    for col in key_cols:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            pct = 100 * n_valid / len(df)
            print(f"  {col}: {n_valid} ({pct:.1f}%)")


def sequence_analysis(df: pd.DataFrame):
    """Analyze sequence features."""
    print("\n" + "="*70)
    print("SEQUENCE ANALYSIS")
    print("="*70)
    
    # Filter guides with sequences
    df_seq = df[df['sgrna_sequence'].notna()].copy()
    print(f"\nAnalyzing {len(df_seq)} guides with sequences")
    
    if 'sgrna_gc_content' in df_seq.columns:
        print(f"\nGC content statistics:")
        print(f"  Mean: {df_seq['sgrna_gc_content'].mean():.3f}")
        print(f"  Median: {df_seq['sgrna_gc_content'].median():.3f}")
        print(f"  Range: {df_seq['sgrna_gc_content'].min():.3f} - {df_seq['sgrna_gc_content'].max():.3f}")
    
    if 'pam_sequence' in df_seq.columns:
        pam_counts = df_seq['pam_sequence'].value_counts()
        print(f"\nPAM sequence distribution (top 10):")
        for pam, count in pam_counts.head(10).items():
            print(f"  {pam}: {count}")


def mismatch_analysis(df: pd.DataFrame):
    """Analyze mismatch patterns."""
    print("\n" + "="*70)
    print("MISMATCH ANALYSIS")
    print("="*70)
    
    df_mm = df[df['n_mismatches'].notna()].copy()
    print(f"\nAnalyzing {len(df_mm)} guides with mismatch data")
    
    if len(df_mm) == 0:
        print("No mismatch data available")
        return
    
    print(f"\nMismatch distribution:")
    mm_counts = df_mm['n_mismatches'].value_counts().sort_index()
    for n_mm, count in mm_counts.items():
        pct = 100 * count / len(df_mm)
        print(f"  {int(n_mm)} mismatches: {count} guides ({pct:.1f}%)")
    
    # Distance to PAM analysis
    df_dist = df_mm[df_mm['min_distance_to_pam'].notna()].copy()
    if len(df_dist) > 0:
        print(f"\nDistance to PAM (for guides with mismatches):")
        print(f"  Mean: {df_dist['min_distance_to_pam'].mean():.2f}")
        print(f"  Median: {df_dist['min_distance_to_pam'].median():.1f}")
        print(f"  Std: {df_dist['min_distance_to_pam'].std():.2f}")
        print(f"  Range: {df_dist['min_distance_to_pam'].min():.0f} - {df_dist['min_distance_to_pam'].max():.0f}")
        
        # Binned distribution
        print(f"\n  Distance distribution:")
        bins = [0, 5, 10, 15, 20]
        labels = ['0-4 (near PAM)', '5-9', '10-14', '15-19 (far from PAM)']
        df_dist['distance_bin'] = pd.cut(df_dist['min_distance_to_pam'], bins=bins, labels=labels, include_lowest=True)
        for label in labels:
            count = (df_dist['distance_bin'] == label).sum()
            pct = 100 * count / len(df_dist)
            print(f"    {label}: {count} ({pct:.1f}%)")


def activity_analysis(df: pd.DataFrame):
    """Analyze activity scores."""
    print("\n" + "="*70)
    print("ACTIVITY ANALYSIS")
    print("="*70)
    
    # GSE132080 activity
    df_gse132080 = df[df['mean_relative_gamma'].notna()].copy()
    if len(df_gse132080) > 0:
        print(f"\nGSE132080 mean_relative_gamma ({len(df_gse132080)} guides):")
        print(f"  Mean: {df_gse132080['mean_relative_gamma'].mean():.3f}")
        print(f"  Median: {df_gse132080['mean_relative_gamma'].median():.3f}")
        print(f"  Range: {df_gse132080['mean_relative_gamma'].min():.3f} - {df_gse132080['mean_relative_gamma'].max():.3f}")
    
    # GSE283614 activity
    df_gse283614 = df[df['relative_activity_mean'].notna()].copy()
    if len(df_gse283614) > 0:
        print(f"\nGSE283614 relative_activity_mean ({len(df_gse283614)} guides):")
        print(f"  Mean: {df_gse283614['relative_activity_mean'].mean():.3f}")
        print(f"  Median: {df_gse283614['relative_activity_mean'].median():.3f}")
        print(f"  Range: {df_gse283614['relative_activity_mean'].min():.3f} - {df_gse283614['relative_activity_mean'].max():.3f}")


def embedding_correlation_analysis(df: pd.DataFrame):
    """Analyze correlation between embeddings and activity."""
    print("\n" + "="*70)
    print("EMBEDDING vs ACTIVITY CORRELATION")
    print("="*70)
    
    # GSE132080: embedding_norm vs mean_relative_gamma
    df_corr = df[(df['embedding_norm'].notna()) & (df['mean_relative_gamma'].notna())].copy()
    if len(df_corr) > 10:
        r, p = pearsonr(df_corr['embedding_norm'], df_corr['mean_relative_gamma'])
        rho, p_spearman = spearmanr(df_corr['embedding_norm'], df_corr['mean_relative_gamma'])
        
        print(f"\nGSE132080: embedding_norm vs mean_relative_gamma ({len(df_corr)} guides)")
        print(f"  Pearson r = {r:.4f}, p = {p:.4e}")
        print(f"  Spearman rho = {rho:.4f}, p = {p_spearman:.4e}")
    
    # Virtual scale vs activity
    df_corr2 = df[(df['virtual_scale'].notna()) & (df['mean_relative_gamma'].notna())].copy()
    if len(df_corr2) > 10:
        r, p = pearsonr(df_corr2['virtual_scale'], df_corr2['mean_relative_gamma'])
        rho, p_spearman = spearmanr(df_corr2['virtual_scale'], df_corr2['mean_relative_gamma'])
        
        print(f"\nGSE132080: virtual_scale vs mean_relative_gamma ({len(df_corr2)} guides)")
        print(f"  Pearson r = {r:.4f}, p = {p:.4e}")
        print(f"  Spearman rho = {rho:.4f}, p = {p_spearman:.4e}")
    
    # GSE283614
    df_corr3 = df[(df['embedding_norm'].notna()) & (df['relative_activity_mean'].notna())].copy()
    if len(df_corr3) > 10:
        r, p = pearsonr(df_corr3['embedding_norm'], df_corr3['relative_activity_mean'])
        rho, p_spearman = spearmanr(df_corr3['embedding_norm'], df_corr3['relative_activity_mean'])
        
        print(f"\nGSE283614: embedding_norm vs relative_activity_mean ({len(df_corr3)} guides)")
        print(f"  Pearson r = {r:.4f}, p = {p:.4e}")
        print(f"  Spearman rho = {rho:.4f}, p = {p_spearman:.4e}")


def mismatch_activity_correlation(df: pd.DataFrame):
    """Analyze correlation between mismatch position/distance and activity."""
    print("\n" + "="*70)
    print("MISMATCH POSITION vs ACTIVITY")
    print("="*70)
    
    # Distance to PAM vs activity (GSE132080)
    df_corr = df[(df['min_distance_to_pam'].notna()) & (df['mean_relative_gamma'].notna())].copy()
    if len(df_corr) > 10:
        r, p = pearsonr(df_corr['min_distance_to_pam'], df_corr['mean_relative_gamma'])
        rho, p_spearman = spearmanr(df_corr['min_distance_to_pam'], df_corr['mean_relative_gamma'])
        
        print(f"\nGSE132080: min_distance_to_pam vs mean_relative_gamma ({len(df_corr)} guides)")
        print(f"  Pearson r = {r:.4f}, p = {p:.4e}")
        print(f"  Spearman rho = {rho:.4f}, p = {p_spearman:.4e}")
        print(f"  Interpretation: Negative correlation means mismatches closer to PAM reduce activity")


def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """Generate a text summary report."""
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    report = []
    report.append("="*70)
    report.append("sgRNA Comprehensive Analysis Summary")
    report.append("="*70)
    report.append("")
    
    # Dataset overview
    report.append("1. DATASET OVERVIEW")
    report.append("-" * 70)
    report.append(f"Total guides analyzed: {len(df)}")
    report.append("")
    for dataset in df['dataset'].unique():
        if pd.notna(dataset):
            count = (df['dataset'] == dataset).sum()
            report.append(f"  {dataset}: {count} guides")
    report.append("")
    
    # Sequence information
    report.append("2. SEQUENCE INFORMATION")
    report.append("-" * 70)
    n_seq = df['sgrna_sequence'].notna().sum()
    n_target = df['target_genomic_sequence'].notna().sum()
    n_pam = df['pam_sequence'].notna().sum()
    report.append(f"  sgRNA sequences: {n_seq}")
    report.append(f"  Target genomic sequences: {n_target}")
    report.append(f"  PAM sequences: {n_pam}")
    report.append("")
    
    # Mismatch statistics
    report.append("3. MISMATCH ANALYSIS")
    report.append("-" * 70)
    df_mm = df[df['n_mismatches'].notna()]
    if len(df_mm) > 0:
        report.append(f"  Guides with mismatch data: {len(df_mm)}")
        mm_dist = df_mm['n_mismatches'].value_counts().sort_index()
        for n_mm, count in mm_dist.items():
            report.append(f"    {int(n_mm)} mismatches: {count} guides")
        
        if df_mm['min_distance_to_pam'].notna().any():
            mean_dist = df_mm['min_distance_to_pam'].mean()
            report.append(f"  Mean distance to PAM: {mean_dist:.2f}")
    report.append("")
    
    # Model embeddings
    report.append("4. MODEL EMBEDDINGS")
    report.append("-" * 70)
    n_emb = df['embedding_norm'].notna().sum()
    n_scale = df['virtual_scale'].notna().sum()
    report.append(f"  Guides with embedding_norm: {n_emb}")
    report.append(f"  Guides with virtual_scale: {n_scale}")
    
    if n_emb > 0:
        mean_emb = df['embedding_norm'].mean()
        report.append(f"  Mean embedding_norm: {mean_emb:.4f}")
    report.append("")
    
    # Activity scores
    report.append("5. ACTIVITY SCORES")
    report.append("-" * 70)
    n_gamma = df['mean_relative_gamma'].notna().sum()
    n_rel = df['relative_activity_mean'].notna().sum()
    report.append(f"  GSE132080 (mean_relative_gamma): {n_gamma} guides")
    report.append(f"  GSE283614 (relative_activity_mean): {n_rel} guides")
    report.append("")
    
    # Key correlations
    report.append("6. KEY CORRELATIONS")
    report.append("-" * 70)
    
    # Embedding vs activity
    df_corr = df[(df['embedding_norm'].notna()) & (df['mean_relative_gamma'].notna())]
    if len(df_corr) > 10:
        r, _ = pearsonr(df_corr['embedding_norm'], df_corr['mean_relative_gamma'])
        rho, _ = spearmanr(df_corr['embedding_norm'], df_corr['mean_relative_gamma'])
        report.append(f"  Embedding_norm vs activity (GSE132080):")
        report.append(f"    Pearson r = {r:.4f}, Spearman rho = {rho:.4f}")
    
    # Distance to PAM vs activity
    df_corr2 = df[(df['min_distance_to_pam'].notna()) & (df['mean_relative_gamma'].notna())]
    if len(df_corr2) > 10:
        r, _ = pearsonr(df_corr2['min_distance_to_pam'], df_corr2['mean_relative_gamma'])
        rho, _ = spearmanr(df_corr2['min_distance_to_pam'], df_corr2['mean_relative_gamma'])
        report.append(f"  Distance_to_PAM vs activity (GSE132080):")
        report.append(f"    Pearson r = {r:.4f}, Spearman rho = {rho:.4f}")
    
    report.append("")
    report.append("="*70)
    
    # Write report
    report_text = "\n".join(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding='utf-8')
    
    print(f"\nSummary report saved to: {output_path}")
    print("\nReport preview:")
    print(report_text)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze comprehensive sgRNA table')
    parser.add_argument('--input', type=Path,
                       default=Path('results/comprehensive_sgrna_analysis_table.csv'),
                       help='Input comprehensive table')
    parser.add_argument('--output', type=Path,
                       default=Path('results/comprehensive_sgrna_summary_report.txt'),
                       help='Output summary report')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Load data
    df = load_data(args.input)
    
    # Run analyses
    basic_statistics(df)
    sequence_analysis(df)
    mismatch_analysis(df)
    activity_analysis(df)
    embedding_correlation_analysis(df)
    mismatch_activity_correlation(df)
    
    # Generate summary report
    generate_summary_report(df, args.output)


if __name__ == '__main__':
    main()
