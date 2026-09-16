#!/usr/bin/env python
"""
Integrate sgRNA-level results from multiple sources into a comprehensive analysis table.

This script combines:
- sgRNA sequences and embeddings from model exports
- Guide assignments and target genes from experimental data
- Activity scores from screening data
- Expression data from h5ad files
- Prepares framework for genomic sequence extraction and mismatch analysis
"""

import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import argparse
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_gse132080_embeddings(embedding_file: Path) -> pd.DataFrame:
    """Load GSE132080 sgRNA embeddings and virtual scales."""
    logger.info(f"Loading GSE132080 embeddings from {embedding_file}")
    df = pd.read_csv(embedding_file)
    
    # Rename columns for consistency
    df = df.rename(columns={
        'label': 'guide_label',
        'target': 'target_gene',
        'sequence': 'sgrna_sequence',
        's': 'virtual_scale',
        'embedding_l2_norm': 'embedding_norm'
    })
    
    # Add dataset identifier
    df['dataset'] = 'GSE132080'
    
    logger.info(f"Loaded {len(df)} guides from GSE132080")
    return df


def load_gse283614_guides(guide_file: Path, activity_file: Path) -> pd.DataFrame:
    """Load GSE283614 guide assignments and activity scores."""
    logger.info(f"Loading GSE283614 guide assignments from {guide_file}")
    guides_df = pd.read_csv(guide_file)
    
    logger.info(f"Loading GSE283614 activity scores from {activity_file}")
    activity_df = pd.read_csv(activity_file, sep='\t')
    
    # Get unique guides (one row per guide, not per cell)
    unique_guides = guides_df[['guide_identity', 'target_gene', 'guide_type', 'guide_family']].drop_duplicates()
    
    # Merge with activity data
    merged = unique_guides.merge(
        activity_df[['guide_identity', 'sequence', 'starting_position_hg38', 
                     'screen_strength_mean', 'relative_activity_mean',
                     'screen_strength_ESC', 'relative_activity_ESC',
                     'screen_strength_DE', 'relative_activity_DE']],
        on='guide_identity',
        how='left'
    )
    
    # Rename for consistency
    merged = merged.rename(columns={
        'sequence': 'sgrna_sequence',
        'starting_position_hg38': 'genomic_position_hg38'
    })
    
    # Add dataset identifier
    merged['dataset'] = 'GSE283614'
    
    logger.info(f"Loaded {len(merged)} unique guides from GSE283614")
    return merged


def extract_guide_expression(adata_path: Path, guide_col: str = 'perturbation_label') -> pd.DataFrame:
    """Extract per-guide mean expression from h5ad file."""
    logger.info(f"Loading expression data from {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    # Get guide labels
    if guide_col not in adata.obs.columns:
        logger.warning(f"Column {guide_col} not found in obs. Available: {adata.obs.columns.tolist()}")
        return pd.DataFrame()
    
    # Calculate mean expression per guide
    guide_labels = adata.obs[guide_col]
    
    # Convert to dense if sparse
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # Group by guide and calculate mean
    unique_guides = guide_labels.unique()
    mean_expressions = []
    
    for guide in unique_guides:
        mask = guide_labels == guide
        mean_expr = X[mask].mean(axis=0)
        mean_expressions.append({
            'guide_identity': guide,
            'n_cells': mask.sum(),
            'mean_expression': mean_expr,
            'total_counts_mean': X[mask].sum(axis=1).mean()
        })
    
    expr_df = pd.DataFrame(mean_expressions)
    logger.info(f"Extracted expression for {len(expr_df)} guides")
    
    return expr_df


def parse_genomic_position(pos_str: str) -> Dict[str, any]:
    """Parse genomic position string to extract chromosome, position, strand."""
    if pd.isna(pos_str):
        return {'chromosome': None, 'position': None, 'strand': None}
    
    # Format: chr12:7787667 or chr12:7798890/7905847
    parts = pos_str.split(':')
    if len(parts) != 2:
        return {'chromosome': None, 'position': None, 'strand': None}
    
    chrom = parts[0]
    # Handle multiple positions (take first)
    pos = parts[1].split('/')[0]
    
    try:
        pos = int(pos)
    except:
        pos = None
    
    return {'chromosome': chrom, 'position': pos, 'strand': None}


def add_genomic_info(df: pd.DataFrame) -> pd.DataFrame:
    """Parse genomic position and prepare for sequence extraction."""
    if 'genomic_position_hg38' not in df.columns:
        df['chromosome'] = None
        df['position'] = None
        df['strand'] = None
        return df
    
    genomic_info = df['genomic_position_hg38'].apply(parse_genomic_position)
    df['chromosome'] = genomic_info.apply(lambda x: x['chromosome'])
    df['position'] = genomic_info.apply(lambda x: x['position'])
    df['strand'] = genomic_info.apply(lambda x: x['strand'])
    
    return df


def calculate_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic sequence features from sgRNA sequences."""
    if 'sgrna_sequence' not in df.columns:
        return df
    
    def calc_gc_content(seq):
        if pd.isna(seq) or len(seq) == 0:
            return None
        gc = sum(1 for base in seq.upper() if base in 'GC')
        return gc / len(seq)
    
    df['sgrna_length'] = df['sgrna_sequence'].apply(lambda x: len(x) if pd.notna(x) else None)
    df['sgrna_gc_content'] = df['sgrna_sequence'].apply(calc_gc_content)
    
    return df


def prepare_mismatch_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add placeholder columns for mismatch analysis (to be filled later)."""
    df['target_genomic_sequence'] = None  # To be filled with sequence extraction
    df['target_sequence_with_pam'] = None  # sgRNA target + PAM
    df['n_mismatches'] = None  # Number of mismatches
    df['mismatch_positions'] = None  # Comma-separated positions
    df['mismatch_distances_to_pam'] = None  # Comma-separated distances
    df['min_distance_to_pam'] = None  # Minimum distance
    df['pam_sequence'] = None  # PAM sequence (NGG for SpCas9)
    
    return df


def integrate_all_data(gse132080_embedding_file: Optional[Path],
                      gse283614_guide_file: Optional[Path],
                      gse283614_activity_file: Optional[Path],
                      gse283614_adata_file: Optional[Path],
                      output_file: Path) -> pd.DataFrame:
    """Main integration function."""
    dfs = []
    
    # Load GSE132080 data if provided
    if gse132080_embedding_file and gse132080_embedding_file.exists():
        df_132080 = load_gse132080_embeddings(gse132080_embedding_file)
        df_132080 = calculate_sequence_features(df_132080)
        dfs.append(df_132080)
    
    # Load GSE283614 data if provided
    if gse283614_guide_file and gse283614_guide_file.exists():
        if not gse283614_activity_file or not gse283614_activity_file.exists():
            logger.warning("GSE283614 activity file not found")
            return pd.DataFrame()
        
        df_283614 = load_gse283614_guides(gse283614_guide_file, gse283614_activity_file)
        df_283614 = add_genomic_info(df_283614)
        df_283614 = calculate_sequence_features(df_283614)
        
        # Add expression data if available
        if gse283614_adata_file and gse283614_adata_file.exists():
            expr_df = extract_guide_expression(gse283614_adata_file)
            if not expr_df.empty:
                df_283614 = df_283614.merge(expr_df, on='guide_identity', how='left')
        
        dfs.append(df_283614)
    
    if not dfs:
        logger.error("No data loaded")
        return pd.DataFrame()
    
    # Combine all datasets
    # Note: GSE132080 and GSE283614 have different column sets
    # We'll keep them separate but with consistent naming
    all_data = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    
    # Add mismatch analysis columns
    all_data = prepare_mismatch_analysis_columns(all_data)
    
    # Reorder columns for better readability
    priority_cols = [
        'dataset', 'guide_identity', 'guide_label', 'target_gene', 'guide_type',
        'sgrna_sequence', 'sgrna_length', 'sgrna_gc_content',
        'chromosome', 'position', 'genomic_position_hg38',
        'virtual_scale', 'embedding_norm',
        'screen_strength_mean', 'relative_activity_mean',
        'n_cells', 'total_counts_mean'
    ]
    
    # Get remaining columns
    remaining_cols = [col for col in all_data.columns if col not in priority_cols]
    
    # Reorder (only include columns that exist)
    existing_priority = [col for col in priority_cols if col in all_data.columns]
    final_cols = existing_priority + remaining_cols
    all_data = all_data[final_cols]
    
    # Save
    logger.info(f"Saving integrated data to {output_file}")
    all_data.to_csv(output_file, index=False)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Integration Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total guides: {len(all_data)}")
    logger.info(f"Datasets: {all_data['dataset'].value_counts().to_dict()}")
    logger.info(f"Guides with sequences: {all_data['sgrna_sequence'].notna().sum()}")
    logger.info(f"Guides with genomic positions: {all_data['position'].notna().sum()}")
    logger.info(f"Guides with embeddings: {all_data['embedding_norm'].notna().sum()}")
    logger.info(f"Guides with activity scores: {all_data['relative_activity_mean'].notna().sum()}")
    logger.info(f"\nOutput saved to: {output_file}")
    logger.info(f"{'='*60}\n")
    
    return all_data


def main():
    parser = argparse.ArgumentParser(description='Integrate sgRNA-level results')
    parser.add_argument('--gse132080-embedding', type=Path,
                       default=Path('runs/gse132080_sgrna_200ep/guide_scale_export/guide_virtual_scales.csv'),
                       help='GSE132080 embedding file')
    parser.add_argument('--gse283614-guides', type=Path,
                       default=Path('reproducibility/prepared/gse283614/gse283614_de_guide_assignments.csv'),
                       help='GSE283614 guide assignments')
    parser.add_argument('--gse283614-activity', type=Path,
                       default=Path('reproducibility/prepared/gse283614/gse283614_table_s1_activity.tsv'),
                       help='GSE283614 activity scores')
    parser.add_argument('--gse283614-adata', type=Path,
                       default=Path('reproducibility/prepared/gse283614/gse283614_de_assigned.h5ad'),
                       help='GSE283614 h5ad file')
    parser.add_argument('--output', type=Path,
                       default=Path('results/integrated_sgrna_table.csv'),
                       help='Output file')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Run integration
    integrate_all_data(
        gse132080_embedding_file=args.gse132080_embedding,
        gse283614_guide_file=args.gse283614_guides,
        gse283614_activity_file=args.gse283614_activity,
        gse283614_adata_file=args.gse283614_adata,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
