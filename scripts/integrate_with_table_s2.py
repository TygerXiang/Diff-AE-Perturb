#!/usr/bin/env python
"""
Integrate sgRNA data with genomic sequences from GSE132080 Table S2 (complete library).

Table S2 contains ALL guides including perfect matches (_00).
Table S8 only contains guides with mismatches.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_table_s2(table_s2_path: Path) -> pd.DataFrame:
    """Load GSE132080 Table S2 with all guide sequences."""
    print(f"Loading Table S2 from {table_s2_path}...")
    df = pd.read_csv(table_s2_path, sep='\t')
    
    # First column is the guide label
    df['guide_label'] = df.iloc[:, 0]
    
    print(f"Loaded {len(df)} guides from Table S2")
    print(f"Columns: {df.columns.tolist()[:5]}...")
    
    return df


def normalize_guide_label(label: str) -> str:
    """
    Normalize guide label to match Table S2 format.
    
    guide_virtual_scales has: ALDOA_ALDOA_+_30077139.23-P1P2_00
    Table S2 has: ALDOA_+_30077139.23-P1P2_00
    
    Remove the duplicate gene name.
    """
    if pd.isna(label) or label in ['control', '*']:
        return label
    
    parts = label.split('_')
    if len(parts) >= 2 and parts[0] == parts[1]:
        # Remove the first occurrence of duplicate gene name
        return '_'.join(parts[1:])
    return label


def calculate_mismatch_info_from_s2(row) -> dict:
    """
    Calculate mismatch information from Table S2 data.
    
    Table S2 has:
    - sequence: the sgRNA sequence (20bp)
    - mismatch 1 position: negative number (-1 = closest to PAM)
    - mismatch 2 position: for double mismatches
    """
    result = {
        'target_genomic_sequence': None,
        'n_mismatches': 0,
        'mismatch_positions': '',
        'mismatch_distances_to_pam': '',
        'min_distance_to_pam': None,
        's2_mismatch1_position': None,
        's2_mismatch2_position': None
    }
    
    # Get sequence
    if pd.notna(row.get('sequence')):
        result['target_genomic_sequence'] = row['sequence']
    
    # Get mismatch positions
    mm1 = row.get('mismatch 1 position')
    mm2 = row.get('mismatch 2 position')
    
    mismatch_positions = []
    distances_to_pam = []
    
    if pd.notna(mm1):
        try:
            mm1_int = int(float(mm1))
            result['s2_mismatch1_position'] = mm1_int
            if mm1_int < 0:
                # Convert: -1 → distance 0, -2 → distance 1, ..., -20 → distance 19
                distance = abs(mm1_int) - 1
                position = 19 - distance  # 0-based position from 5' end
                mismatch_positions.append(position)
                distances_to_pam.append(distance)
        except:
            pass
    
    if pd.notna(mm2):
        try:
            mm2_int = int(float(mm2))
            result['s2_mismatch2_position'] = mm2_int
            if mm2_int < 0:
                distance = abs(mm2_int) - 1
                position = 19 - distance
                mismatch_positions.append(position)
                distances_to_pam.append(distance)
        except:
            pass
    
    # Fill in results
    if mismatch_positions:
        result['n_mismatches'] = len(mismatch_positions)
        result['mismatch_positions'] = ','.join(map(str, mismatch_positions))
        result['mismatch_distances_to_pam'] = ','.join(map(str, distances_to_pam))
        result['min_distance_to_pam'] = min(distances_to_pam)
    
    return result


def integrate_genome_sequences(integrated_table_path: Path,
                               table_s2_path: Path,
                               output_path: Path):
    """Main integration function."""
    
    # Load integrated table
    print(f"\nLoading integrated sgRNA table from {integrated_table_path}...")
    df_integrated = pd.read_csv(integrated_table_path)
    print(f"Loaded {len(df_integrated)} guides")
    
    # Load Table S2
    df_s2 = load_table_s2(table_s2_path)
    
    # Normalize guide labels for matching
    print("\nNormalizing guide labels for matching...")
    df_integrated['guide_label_normalized'] = df_integrated['guide_label'].apply(normalize_guide_label)
    
    print("Sample normalized labels:")
    sample_labels = df_integrated[df_integrated['guide_label'].notna()].head(5)
    for idx, row in sample_labels.iterrows():
        print(f"  Original: {row['guide_label']}")
        print(f"  Normalized: {row['guide_label_normalized']}")
    
    # Merge on normalized guide_label
    print("\nMerging with Table S2...")
    df_merged = df_integrated.merge(
        df_s2[['guide_label', 'perfect match sgRNA', 'gene', 'sequence',
               'mismatch 1 position', 'new pairing mismatch 1',
               'mismatch 2 position', 'new pairing mismatch 2',
               'off-target specificity score (CFD)']],
        left_on='guide_label_normalized',
        right_on='guide_label',
        how='left',
        suffixes=('', '_s2')
    )
    
    print(f"Merged: {len(df_merged)} total guides")
    print(f"GSE132080 guides with S2 data: {df_merged['sequence'].notna().sum()}")
    
    # Process each guide
    print("\nCalculating mismatches and distances to PAM...")
    results = []
    
    for idx, row in df_merged.iterrows():
        result = {
            'target_genomic_sequence': None,
            'target_sequence_with_pam': None,
            'pam_sequence': None,
            'fully_matched_sequence': None,
            'n_mismatches': None,
            'mismatch_positions': None,
            'mismatch_distances_to_pam': None,
            'min_distance_to_pam': None,
            's2_mismatch1_position': None,
            's2_mismatch2_position': None,
            'cfd_specificity_score': None
        }
        
        # For GSE132080 guides with Table S2 data
        if pd.notna(row.get('sequence')):
            # Get the sgRNA sequence (20bp)
            sgrna_seq = row['sequence']
            result['target_genomic_sequence'] = sgrna_seq
            
            # For perfect match guides, extract PAM from genome if available
            # For now, we'll note that PAM extraction requires additional genomic data
            
            # Get fully matched sequence
            result['fully_matched_sequence'] = row.get('perfect match sgRNA')
            
            # Calculate mismatch info
            mm_info = calculate_mismatch_info_from_s2(row)
            result.update(mm_info)
            
            # CFD specificity score
            if pd.notna(row.get('off-target specificity score (CFD)')):
                result['cfd_specificity_score'] = row['off-target specificity score (CFD)']
        
        # For GSE283614 guides, keep existing data
        elif pd.notna(row.get('genomic_position_hg38')):
            pass
        
        results.append(result)
    
    # Update dataframe with new columns
    for key in results[0].keys():
        if key not in df_merged.columns:
            df_merged[key] = [r[key] for r in results]
        else:
            # Fill NaN values
            new_values = [r[key] for r in results]
            mask = df_merged[key].isna()
            df_merged.loc[mask, key] = pd.Series([r[key] for i, r in enumerate(results) if mask.iloc[i]])
    
    # Reorder columns for better readability
    priority_cols = [
        'dataset', 'guide_identity', 'guide_label', 'target_gene', 'guide_type',
        'sgrna_sequence', 'sgrna_length', 'sgrna_gc_content',
        'fully_matched_sequence', 'target_genomic_sequence', 
        'target_sequence_with_pam', 'pam_sequence',
        'n_mismatches', 'mismatch_positions', 'mismatch_distances_to_pam',
        'min_distance_to_pam', 's2_mismatch1_position', 's2_mismatch2_position',
        'cfd_specificity_score',
        'chromosome', 'position', 'genomic_position_hg38',
        'virtual_scale', 'embedding_norm',
        'screen_strength_mean', 'relative_activity_mean'
    ]
    
    existing_priority = [col for col in priority_cols if col in df_merged.columns]
    remaining_cols = [col for col in df_merged.columns if col not in priority_cols]
    df_merged = df_merged[existing_priority + remaining_cols]
    
    # Save
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "="*70)
    print("Final Integration Summary")
    print("="*70)
    print(f"Total guides: {len(df_merged)}")
    print(f"\nBy dataset:")
    print(df_merged['dataset'].value_counts())
    print(f"\nSequence information:")
    print(f"  Guides with sgRNA sequences: {df_merged['sgrna_sequence'].notna().sum()}")
    print(f"  Guides with target sequences: {df_merged['target_genomic_sequence'].notna().sum()}")
    print(f"  Guides with PAM sequences: {df_merged['pam_sequence'].notna().sum()}")
    print(f"\nMismatch analysis:")
    print(f"  Guides with mismatch data: {df_merged['n_mismatches'].notna().sum()}")
    
    if df_merged['n_mismatches'].notna().any():
        print(f"\n  Mismatch distribution:")
        mismatch_counts = df_merged['n_mismatches'].value_counts().sort_index()
        for mm, count in mismatch_counts.items():
            print(f"    {int(mm)} mismatches: {count} guides")
        
        if df_merged['min_distance_to_pam'].notna().any():
            print(f"\n  Distance to PAM (for guides with mismatches):")
            print(f"    Mean: {df_merged['min_distance_to_pam'].mean():.2f}")
            print(f"    Median: {df_merged['min_distance_to_pam'].median():.1f}")
            print(f"    Range: {df_merged['min_distance_to_pam'].min():.0f} - {df_merged['min_distance_to_pam'].max():.0f}")
    
    print(f"\nCFD specificity scores:")
    print(f"  Guides with CFD scores: {df_merged['cfd_specificity_score'].notna().sum()}")
    
    print(f"\nModel embeddings:")
    print(f"  Guides with embedding_norm: {df_merged['embedding_norm'].notna().sum()}")
    print(f"  Guides with virtual_scale: {df_merged['virtual_scale'].notna().sum()}")
    
    print(f"\nOutput saved to: {output_path}")
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Integrate sgRNA data with complete sequences from Table S2'
    )
    parser.add_argument('--input', type=Path,
                       default=Path('results/integrated_sgrna_table.csv'),
                       help='Input integrated sgRNA table')
    parser.add_argument('--table-s2', type=Path,
                       default=Path(__file__).resolve().parents[1] / 'datasets' / 'GSE132080' / '41587_2019_387_MOESM3_ESM' / 'Table_S2_large-scale_library_sequences_revised.txt',
                       help='Path to Table S2')
    parser.add_argument('--output', type=Path,
                       default=Path('results/comprehensive_sgrna_analysis_table.csv'),
                       help='Output comprehensive table')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    if not args.table_s2.exists():
        print(f"Error: Table S2 not found: {args.table_s2}")
        return
    
    # Run integration
    integrate_genome_sequences(args.input, args.table_s2, args.output)


if __name__ == '__main__':
    main()
