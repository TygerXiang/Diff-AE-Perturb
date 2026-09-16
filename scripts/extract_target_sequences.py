#!/usr/bin/env python
"""
Extract target genomic sequences and calculate mismatch positions relative to PAM.

This script:
1. Reads the integrated sgRNA table
2. Extracts genomic sequences for guides with chromosome/position information
3. Calculates mismatches between sgRNA and target sequences
4. Computes distances from mismatch positions to PAM sequence
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

try:
    from pyfaidx import Fasta
    PYFAIDX_AVAILABLE = True
except ImportError:
    PYFAIDX_AVAILABLE = False
    Fasta = None  # Define as None for type hints
    warnings.warn("pyfaidx not available. Will use alternative method or mock data.")


def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base.upper(), 'N') for base in reversed(seq))


def extract_sequence_from_genome(chrom: str, position: int, strand: str, 
                                 length: int = 23, genome_fasta = None) -> Optional[str]:
    """
    Extract genomic sequence from reference genome.
    
    Args:
        chrom: Chromosome name (e.g., 'chr12')
        position: Genomic position (1-based)
        strand: '+' or '-'
        length: Length to extract (default 23 = 20bp sgRNA + 3bp PAM)
        genome_fasta: pyfaidx Fasta object
    
    Returns:
        Extracted sequence or None if extraction fails
    """
    if genome_fasta is None:
        return None
    
    try:
        # pyfaidx uses 0-based coordinates
        if strand == '+':
            seq = genome_fasta[chrom][position-1:position-1+length].seq
        else:
            # For minus strand, get sequence and reverse complement
            seq = genome_fasta[chrom][position-1:position-1+length].seq
            seq = reverse_complement(seq)
        
        return seq.upper()
    except Exception as e:
        warnings.warn(f"Failed to extract sequence for {chrom}:{position}: {e}")
        return None


def find_mismatches(sgrna_seq: str, target_seq: str) -> Tuple[int, List[int]]:
    """
    Find mismatch positions between sgRNA and target sequence.
    
    Args:
        sgrna_seq: sgRNA sequence (20bp)
        target_seq: Target genomic sequence (should be ≥20bp)
    
    Returns:
        Tuple of (number of mismatches, list of mismatch positions)
        Positions are 0-based from 5' end of sgRNA
    """
    if not sgrna_seq or not target_seq:
        return 0, []
    
    # Ensure sequences are same length (use min length)
    min_len = min(len(sgrna_seq), len(target_seq))
    sgrna_seq = sgrna_seq[:min_len].upper()
    target_seq = target_seq[:min_len].upper()
    
    mismatch_positions = []
    for i, (s, t) in enumerate(zip(sgrna_seq, target_seq)):
        if s != t and s != 'N' and t != 'N':
            mismatch_positions.append(i)
    
    return len(mismatch_positions), mismatch_positions


def calculate_distance_to_pam(mismatch_positions: List[int], sgrna_length: int = 20) -> List[int]:
    """
    Calculate distance from each mismatch position to PAM.
    
    For SpCas9, PAM is at 3' end (NGG), so distance is measured from 3' end.
    Position 0 (5' end) is farthest from PAM.
    Position 19 (3' end) is closest to PAM.
    
    Args:
        mismatch_positions: List of 0-based mismatch positions from 5' end
        sgrna_length: Length of sgRNA (default 20)
    
    Returns:
        List of distances to PAM (distance from 3' end)
    """
    # Distance to PAM = (sgrna_length - 1) - position
    # Position 19 (3' end) -> distance 0
    # Position 0 (5' end) -> distance 19
    return [sgrna_length - 1 - pos for pos in mismatch_positions]


def process_guide_sequences(df: pd.DataFrame, 
                           genome_fasta_path: Optional[Path] = None,
                           use_mock_sequences: bool = False) -> pd.DataFrame:
    """
    Process all guides: extract target sequences and calculate mismatches.
    
    Args:
        df: DataFrame with sgRNA data
        genome_fasta_path: Path to reference genome FASTA file
        use_mock_sequences: If True, generate mock target sequences for testing
    
    Returns:
        DataFrame with added sequence and mismatch information
    """
    df = df.copy()
    
    # Try to load genome if path provided
    genome_fasta = None
    if genome_fasta_path and PYFAIDX_AVAILABLE:
        try:
            genome_fasta = Fasta(str(genome_fasta_path))
            print(f"Loaded genome from {genome_fasta_path}")
        except Exception as e:
            print(f"Failed to load genome: {e}")
            genome_fasta = None
    
    # Process each guide
    results = []
    for idx, row in df.iterrows():
        result = {
            'target_genomic_sequence': None,
            'target_sequence_with_pam': None,
            'pam_sequence': None,
            'n_mismatches': None,
            'mismatch_positions': None,
            'mismatch_distances_to_pam': None,
            'min_distance_to_pam': None
        }
        
        # Check if we have necessary data
        if pd.notna(row.get('sgrna_sequence')) and pd.notna(row.get('chromosome')) and pd.notna(row.get('position')):
            sgrna_seq = row['sgrna_sequence']
            
            # Extract genomic sequence
            if genome_fasta:
                # Assume positive strand if not specified
                strand = row.get('strand', '+')
                target_with_pam = extract_sequence_from_genome(
                    row['chromosome'], 
                    row['position'], 
                    strand,
                    length=23,
                    genome_fasta=genome_fasta
                )
            elif use_mock_sequences:
                # For testing: create mock sequence with some mismatches
                target_with_pam = sgrna_seq + 'NGG'  # Perfect match + PAM
                # Add some random mismatches
                if np.random.random() > 0.5:
                    target_list = list(target_with_pam)
                    n_mutations = np.random.randint(0, 3)
                    for _ in range(n_mutations):
                        pos = np.random.randint(0, len(sgrna_seq))
                        bases = ['A', 'T', 'G', 'C']
                        bases.remove(target_list[pos])
                        target_list[pos] = np.random.choice(bases)
                    target_with_pam = ''.join(target_list)
            else:
                target_with_pam = None
            
            if target_with_pam:
                # Split into target (20bp) and PAM (3bp)
                target_seq = target_with_pam[:len(sgrna_seq)]
                pam_seq = target_with_pam[len(sgrna_seq):len(sgrna_seq)+3] if len(target_with_pam) >= len(sgrna_seq)+3 else None
                
                result['target_genomic_sequence'] = target_seq
                result['target_sequence_with_pam'] = target_with_pam
                result['pam_sequence'] = pam_seq
                
                # Calculate mismatches
                n_mm, mm_positions = find_mismatches(sgrna_seq, target_seq)
                result['n_mismatches'] = n_mm
                
                if mm_positions:
                    # Convert to comma-separated strings
                    result['mismatch_positions'] = ','.join(map(str, mm_positions))
                    
                    # Calculate distances to PAM
                    distances = calculate_distance_to_pam(mm_positions, len(sgrna_seq))
                    result['mismatch_distances_to_pam'] = ','.join(map(str, distances))
                    result['min_distance_to_pam'] = min(distances)
                else:
                    result['mismatch_positions'] = ''
                    result['mismatch_distances_to_pam'] = ''
                    result['min_distance_to_pam'] = None
        
        results.append(result)
    
    # Add results to dataframe
    for key in results[0].keys():
        df[key] = [r[key] for r in results]
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract target sequences and calculate mismatches')
    parser.add_argument('--input', type=Path,
                       default=Path('results/integrated_sgrna_table.csv'),
                       help='Input integrated sgRNA table')
    parser.add_argument('--genome', type=Path,
                       help='Path to reference genome FASTA (e.g., hg38.fa)')
    parser.add_argument('--output', type=Path,
                       default=Path('results/integrated_sgrna_table_with_sequences.csv'),
                       help='Output file with sequences and mismatches')
    parser.add_argument('--use-mock', action='store_true',
                       help='Generate mock sequences for testing (when genome not available)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} guides")
    
    # Check availability of genome
    if args.genome and not args.genome.exists():
        print(f"Warning: Genome file {args.genome} not found")
        args.genome = None
    
    if not args.genome and not args.use_mock:
        print("\nNo genome file provided and --use-mock not set.")
        print("Options:")
        print("1. Provide genome FASTA with --genome /path/to/hg38.fa")
        print("2. Use --use-mock to generate mock sequences for testing")
        print("\nWithout sequences, mismatch analysis cannot be performed.")
        print("Will save table with empty sequence columns.")
        use_mock = False
    else:
        use_mock = args.use_mock
    
    # Process sequences
    print("\nProcessing sequences...")
    df = process_guide_sequences(df, args.genome, use_mock_sequences=use_mock)
    
    # Save
    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("Sequence Extraction Summary")
    print("="*60)
    print(f"Total guides: {len(df)}")
    print(f"Guides with target sequences: {df['target_genomic_sequence'].notna().sum()}")
    print(f"Guides with PAM sequences: {df['pam_sequence'].notna().sum()}")
    print(f"Guides with mismatches calculated: {df['n_mismatches'].notna().sum()}")
    
    if df['n_mismatches'].notna().any():
        print(f"\nMismatch distribution:")
        print(df['n_mismatches'].value_counts().sort_index())
        
        if df['min_distance_to_pam'].notna().any():
            print(f"\nMin distance to PAM statistics:")
            print(f"  Mean: {df['min_distance_to_pam'].mean():.2f}")
            print(f"  Median: {df['min_distance_to_pam'].median():.2f}")
            print(f"  Range: {df['min_distance_to_pam'].min():.0f} - {df['min_distance_to_pam'].max():.0f}")
    
    print(f"\nOutput saved to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
