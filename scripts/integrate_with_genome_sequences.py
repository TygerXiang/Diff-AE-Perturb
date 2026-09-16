#!/usr/bin/env python
"""
Integrate sgRNA data with genomic sequences from GSE132080 Table S8.

This script:
1. Loads the integrated sgRNA table
2. Loads Table S8 with genome input sequences and mismatch positions
3. Calculates mismatch positions and distances to PAM
4. Generates the final comprehensive sgRNA analysis table
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_table_s8(table_s8_path: Path) -> pd.DataFrame:
    """Load GSE132080 Table S8 with genome sequences."""
    print(f"Loading Table S8 from {table_s8_path}...")
    df = pd.read_csv(table_s8_path, sep='\t')
    
    # Parse the guide label from the first column
    df['guide_label'] = df.iloc[:, 0]  # First column is the guide label
    
    print(f"Loaded {len(df)} guides from Table S8")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def parse_mismatch_position(pos_str) -> Optional[int]:
    """Parse mismatch position from Table S8 format (e.g., '-17.0')."""
    if pd.isna(pos_str):
        return None
    try:
        # Negative values indicate position from PAM (3' end)
        # -1 is closest to PAM, -20 is farthest
        return int(float(pos_str))
    except:
        return None


def calculate_mismatch_distances_to_pam(mismatch_positions: List[int], 
                                       sgrna_length: int = 20) -> List[int]:
    """
    Calculate distance from mismatch positions to PAM.
    
    Table S8 uses negative positions where:
    - Position -1 is the base closest to PAM (position 19 in 0-indexed)
    - Position -20 is the base farthest from PAM (position 0 in 0-indexed)
    
    We convert to distance from PAM:
    - Position -1 → distance 0 (closest to PAM)
    - Position -20 → distance 19 (farthest from PAM)
    """
    distances = []
    for pos in mismatch_positions:
        if pos < 0:
            # Convert negative position to distance from PAM
            # -1 → 0, -2 → 1, ..., -20 → 19
            distance = abs(pos) - 1
            distances.append(distance)
    return distances


def compare_sequences(sgrna_seq: str, target_seq: str) -> Tuple[int, List[int], List[int]]:
    """
    Compare sgRNA and target sequences to find mismatches.
    
    Args:
        sgrna_seq: sgRNA sequence (20bp)
        target_seq: Target genomic sequence (from Table S8 genome_input or sgRNA_input)
    
    Returns:
        Tuple of (n_mismatches, mismatch_positions_0based, distances_to_pam)
    """
    if not sgrna_seq or not target_seq:
        return 0, [], []
    
    sgrna_seq = sgrna_seq.upper()
    target_seq = target_seq.upper()
    
    # Find the sgRNA within the target sequence (target may include flanking + PAM)
    # Table S8 genome_input is typically 26bp: 3bp upstream + 20bp sgRNA + 3bp PAM
    # sgRNA_input is the actual variant sequence
    
    # Try to align: look for 20bp match region
    mismatch_positions = []
    
    # If sequences are same length, compare directly
    if len(sgrna_seq) == len(target_seq):
        for i, (s, t) in enumerate(zip(sgrna_seq, target_seq)):
            if s != t and s != 'N' and t != 'N':
                mismatch_positions.append(i)
    else:
        # If target is longer, extract the 20bp guide region
        # Typically: position 3:23 in a 26bp sequence
        if len(target_seq) >= 23:
            guide_region = target_seq[3:23]
            for i, (s, t) in enumerate(zip(sgrna_seq, guide_region)):
                if s != t and s != 'N' and t != 'N':
                    mismatch_positions.append(i)
        else:
            # Use minimum length
            min_len = min(len(sgrna_seq), len(target_seq))
            for i in range(min_len):
                if sgrna_seq[i] != target_seq[i]:
                    mismatch_positions.append(i)
    
    # Calculate distances to PAM (position 19 is closest, position 0 is farthest)
    distances_to_pam = [19 - pos for pos in mismatch_positions]
    
    return len(mismatch_positions), mismatch_positions, distances_to_pam


def normalize_guide_label(label: str) -> str:
    """
    Normalize guide label to match Table S8 format.
    
    guide_virtual_scales has: ALDOA_ALDOA_+_30077139.23-P1P2_00
    Table S8 has: ALDOA_+_30077139.23-P1P2_00
    
    Remove the duplicate gene name.
    """
    if pd.isna(label) or label in ['control', '*']:
        return label
    
    parts = label.split('_')
    if len(parts) >= 2 and parts[0] == parts[1]:
        # Remove the first occurrence of duplicate gene name
        return '_'.join(parts[1:])
    return label


def integrate_genome_sequences(integrated_table_path: Path,
                               table_s8_path: Path,
                               output_path: Path):
    """Main integration function."""
    
    # Load integrated table
    print(f"\nLoading integrated sgRNA table from {integrated_table_path}...")
    df_integrated = pd.read_csv(integrated_table_path)
    print(f"Loaded {len(df_integrated)} guides")
    
    # Load Table S8
    df_s8 = load_table_s8(table_s8_path)
    
    # Normalize guide labels for matching
    print("\nNormalizing guide labels for matching...")
    df_integrated['guide_label_normalized'] = df_integrated['guide_label'].apply(normalize_guide_label)
    
    print("Sample normalized labels:")
    sample_labels = df_integrated[df_integrated['guide_label'].notna()].head(3)
    for idx, row in sample_labels.iterrows():
        print(f"  Original: {row['guide_label']}")
        print(f"  Normalized: {row['guide_label_normalized']}")
    
    # Merge on normalized guide_label
    print("\nMerging with Table S8...")
    df_merged = df_integrated.merge(
        df_s8[['guide_label', 'perfect match sgRNA', 'sgRNA sequence', 
               'mismatch position', 'new pairing', 'genome input', 'sgRNA input',
               'mean relative gamma']],
        left_on='guide_label_normalized',
        right_on='guide_label',
        how='left',
        suffixes=('', '_s8')
    )
    
    print(f"Merged: {len(df_merged)} total guides")
    print(f"GSE132080 guides with S8 data: {df_merged['genome input'].notna().sum()}")
    
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
            'table_s8_mismatch_position': None,
            'mean_relative_gamma': None
        }
        
        # For GSE132080 guides with Table S8 data
        if pd.notna(row.get('genome input')):
            genome_input = row['genome input']
            sgrna_input = row['sgRNA input']
            
            # genome input is typically 26bp: 3bp + 20bp sgRNA + 3bp PAM
            result['target_sequence_with_pam'] = genome_input
            
            if len(genome_input) >= 23:
                # Extract 20bp target (positions 3:23)
                result['target_genomic_sequence'] = genome_input[3:23]
                # Extract PAM (last 3bp)
                result['pam_sequence'] = genome_input[-3:]
            
            # Get fully matched sequence
            result['fully_matched_sequence'] = row.get('perfect match sgRNA')
            
            # Get Table S8 mismatch position
            if pd.notna(row.get('mismatch position')):
                s8_pos = parse_mismatch_position(row['mismatch position'])
                result['table_s8_mismatch_position'] = s8_pos
                
                # Calculate distance to PAM from S8 position
                if s8_pos is not None and s8_pos < 0:
                    # S8 uses -1 for closest to PAM
                    distance = abs(s8_pos) - 1
                    result['mismatch_positions'] = str(19 - distance)  # Convert to 0-based position
                    result['mismatch_distances_to_pam'] = str(distance)
                    result['min_distance_to_pam'] = distance
                    result['n_mismatches'] = 1
            else:
                # Perfect match (no mismatch)
                result['n_mismatches'] = 0
                result['mismatch_positions'] = ''
                result['mismatch_distances_to_pam'] = ''
            
            # Activity score
            result['mean_relative_gamma'] = row.get('mean relative gamma')
        
        # For guides with sequences but no S8 data, compare directly
        elif pd.notna(row.get('sgrna_sequence')) and pd.notna(row.get('genomic_position_hg38')):
            # GSE283614 guides - keep existing data structure
            pass
        
        results.append(result)
    
    # Update dataframe with new columns
    for key in results[0].keys():
        # Only update if column doesn't exist or is empty
        if key not in df_merged.columns:
            df_merged[key] = [r[key] for r in results]
        else:
            # Fill NaN values
            new_values = [r[key] for r in results]
            df_merged[key] = df_merged[key].fillna(pd.Series(new_values, index=df_merged.index))
    
    # Reorder columns for better readability
    priority_cols = [
        'dataset', 'guide_identity', 'guide_label', 'target_gene', 'guide_type',
        'sgrna_sequence', 'sgrna_length', 'sgrna_gc_content',
        'fully_matched_sequence', 'target_genomic_sequence', 
        'target_sequence_with_pam', 'pam_sequence',
        'n_mismatches', 'mismatch_positions', 'mismatch_distances_to_pam',
        'min_distance_to_pam', 'table_s8_mismatch_position',
        'chromosome', 'position', 'genomic_position_hg38',
        'virtual_scale', 'embedding_norm',
        'screen_strength_mean', 'relative_activity_mean',
        'mean_relative_gamma'
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
    
    print(f"\nActivity scores:")
    print(f"  GSE132080 with mean_relative_gamma: {df_merged['mean_relative_gamma'].notna().sum()}")
    print(f"  GSE283614 with relative_activity_mean: {df_merged['relative_activity_mean'].notna().sum()}")
    
    print(f"\nModel embeddings:")
    print(f"  Guides with embedding_norm: {df_merged['embedding_norm'].notna().sum()}")
    print(f"  Guides with virtual_scale: {df_merged['virtual_scale'].notna().sum()}")
    
    print(f"\nOutput saved to: {output_path}")
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Integrate sgRNA data with genomic sequences from Table S8'
    )
    parser.add_argument('--input', type=Path,
                       default=Path('results/integrated_sgrna_table.csv'),
                       help='Input integrated sgRNA table')
    parser.add_argument('--table-s8', type=Path,
                       default=Path('../diff-project/datasets/GSE132080/41587_2019_387_MOESM3_ESM/Table_S8_machine_learning_input.txt'),
                       help='Path to Table S8')
    parser.add_argument('--output', type=Path,
                       default=Path('results/comprehensive_sgrna_analysis_table.csv'),
                       help='Output comprehensive table')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    if not args.table_s8.exists():
        print(f"Error: Table S8 not found: {args.table_s8}")
        return
    
    # Run integration
    integrate_genome_sequences(args.input, args.table_s8, args.output)


if __name__ == '__main__':
    main()
