#!/usr/bin/env python
"""
Quick integration of sgRNA-level results (without h5ad loading).
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("Loading GSE132080 embeddings...")
gse132080_file = Path('runs/gse132080_sgrna_200ep/guide_scale_export/guide_virtual_scales.csv')
df_132080 = pd.read_csv(gse132080_file)

# Rename columns
df_132080 = df_132080.rename(columns={
    'label': 'guide_label',
    'target': 'target_gene',
    'sequence': 'sgrna_sequence',
    's': 'virtual_scale',
    'embedding_l2_norm': 'embedding_norm'
})
df_132080['dataset'] = 'GSE132080'

print(f"Loaded {len(df_132080)} guides from GSE132080")

print("\nLoading GSE283614 guide assignments...")
gse283614_guides = Path('reproducibility/prepared/gse283614/gse283614_de_guide_assignments.csv')
guides_df = pd.read_csv(gse283614_guides)

print("\nLoading GSE283614 activity scores...")
gse283614_activity = Path('reproducibility/prepared/gse283614/gse283614_table_s1_activity.tsv')
activity_df = pd.read_csv(gse283614_activity, sep='\t')

# Get unique guides
unique_guides = guides_df[['guide_identity', 'target_gene', 'guide_type', 'guide_family']].drop_duplicates()

print(f"Found {len(unique_guides)} unique guides in GSE283614")

# Merge with activity
df_283614 = unique_guides.merge(
    activity_df[['guide_identity', 'sequence', 'starting_position_hg38', 
                 'screen_strength_mean', 'relative_activity_mean',
                 'screen_strength_ESC', 'relative_activity_ESC',
                 'screen_strength_DE', 'relative_activity_DE']],
    on='guide_identity',
    how='left'
)

# Rename
df_283614 = df_283614.rename(columns={
    'sequence': 'sgrna_sequence',
    'starting_position_hg38': 'genomic_position_hg38'
})
df_283614['dataset'] = 'GSE283614'

print(f"Merged {len(df_283614)} guides with activity data")

# Parse genomic positions
def parse_position(pos_str):
    if pd.isna(pos_str):
        return pd.Series({'chromosome': None, 'position': None})
    parts = pos_str.split(':')
    if len(parts) != 2:
        return pd.Series({'chromosome': None, 'position': None})
    chrom = parts[0]
    pos = parts[1].split('/')[0]
    try:
        pos = int(pos)
    except:
        pos = None
    return pd.Series({'chromosome': chrom, 'position': pos})

print("\nParsing genomic positions...")
genomic_info = df_283614['genomic_position_hg38'].apply(parse_position)
df_283614['chromosome'] = genomic_info['chromosome']
df_283614['position'] = genomic_info['position']

# Add sequence features
def calc_gc(seq):
    if pd.isna(seq) or len(seq) == 0:
        return None
    return sum(1 for b in seq.upper() if b in 'GC') / len(seq)

print("\nCalculating sequence features...")
for df in [df_132080, df_283614]:
    df['sgrna_length'] = df['sgrna_sequence'].apply(lambda x: len(x) if pd.notna(x) else None)
    df['sgrna_gc_content'] = df['sgrna_sequence'].apply(calc_gc)

# Add mismatch analysis placeholders
for df in [df_132080, df_283614]:
    df['target_genomic_sequence'] = None
    df['target_sequence_with_pam'] = None
    df['n_mismatches'] = None
    df['mismatch_positions'] = None
    df['mismatch_distances_to_pam'] = None
    df['min_distance_to_pam'] = None
    df['pam_sequence'] = None

# Combine
print("\nCombining datasets...")
all_data = pd.concat([df_132080, df_283614], axis=0, ignore_index=True, sort=False)

# Reorder columns
priority_cols = [
    'dataset', 'guide_identity', 'guide_label', 'target_gene', 'guide_type',
    'sgrna_sequence', 'sgrna_length', 'sgrna_gc_content',
    'chromosome', 'position', 'genomic_position_hg38',
    'virtual_scale', 'embedding_norm',
    'screen_strength_mean', 'relative_activity_mean',
    'screen_strength_ESC', 'relative_activity_ESC',
    'screen_strength_DE', 'relative_activity_DE',
    'target_genomic_sequence', 'target_sequence_with_pam',
    'n_mismatches', 'mismatch_positions', 'mismatch_distances_to_pam',
    'min_distance_to_pam', 'pam_sequence'
]

existing_cols = [col for col in priority_cols if col in all_data.columns]
remaining_cols = [col for col in all_data.columns if col not in priority_cols]
all_data = all_data[existing_cols + remaining_cols]

# Save
output_file = Path('results/integrated_sgrna_table.csv')
output_file.parent.mkdir(parents=True, exist_ok=True)

print(f"\nSaving to {output_file}...")
all_data.to_csv(output_file, index=False)

print("\n" + "="*60)
print("Integration Summary")
print("="*60)
print(f"Total guides: {len(all_data)}")
print(f"\nDatasets:")
print(all_data['dataset'].value_counts())
print(f"\nGuides with sequences: {all_data['sgrna_sequence'].notna().sum()}")
print(f"Guides with genomic positions: {all_data['position'].notna().sum()}")
print(f"Guides with embeddings: {all_data['embedding_norm'].notna().sum()}")
print(f"Guides with activity scores: {all_data['relative_activity_mean'].notna().sum()}")
print(f"\nColumns: {len(all_data.columns)}")
print(f"\nFirst few columns:")
print(all_data.columns.tolist()[:15])
print(f"\nOutput saved to: {output_file}")
print("="*60)
