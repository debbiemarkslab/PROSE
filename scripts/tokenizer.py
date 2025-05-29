from poet.alphabets import Alphabet
import numpy as np
import json
import os
import gc
from collections import defaultdict
import pickle
import random
from tqdm import tqdm 


class DNATokenizer(Alphabet):
    def __init__(self, vocab_size=100):
        # Initialize with the basic DNA nucleotides A, T, C, G
        super().__init__(b"ATCG")
        
        # Store the maximum vocabulary size for BPE
        self.vocab_size = vocab_size
        
        # Dictionary to store the merged pairs
        self.merges = {}
        self.vocab = {b'A': 0, b'T': 1, b'C': 2, b'G': 3}
        
        # The inverse of merges for tokenization
        self.merge_ids = {}
        
        # To track if the tokenizer has been trained
        self.is_trained = False
    
    def train_on_dict(self, sequence_dict, verbose=False, batch_size=10000, sample_size=None):
        """
        Train the BPE tokenizer on a dictionary of DNA sequences
        
        Args:
            sequence_dict: Dictionary where keys are genes and values are lists of sequences
            verbose: Whether to print progress during training
            batch_size: Number of sequences to process at once
            sample_size: If provided, sample this many sequences from each gene (None for all)
        """
        if verbose:
            print(f"Training BPE tokenizer with target vocab size: {self.vocab_size}")
            print(f"Starting with base vocabulary: {list(self.vocab.keys())}")
            print(f"Processing {len(sequence_dict)} genes")
        
        # Initialize vocabulary with single characters
        vocab = self.vocab.copy()
        next_id = len(vocab)
        
        # Main BPE training loop
        iteration = 0
        while len(vocab) < self.vocab_size:
            iteration += 1
            
            # Count frequencies of adjacent pairs across all batches
            pair_counts = defaultdict(int)
            
            # Process each gene
            for gene_idx, (gene, sequences) in tqdm(enumerate(sequence_dict.items()), total=len(sequence_dict)):
                sequences = list(sequences)
                # Sample sequences if requested
                if sample_size is not None and len(sequences) > sample_size:
                    sequences_to_process = random.sample(sequences, sample_size)
                else:
                    sequences_to_process = sequences
                
                if verbose and gene_idx % 10 == 0:
                    print(f"Processing gene {gene_idx}/{len(sequence_dict)}: {gene} with {len(sequences_to_process)} sequences")
                
                # Process sequences in batches
                for batch_start in range(0, len(sequences_to_process), batch_size):
                    batch_end = min(batch_start + batch_size, len(sequences_to_process))
                    batch = sequences_to_process[batch_start:batch_end]
                    
                    # Process each sequence in the batch
                    for seq in batch:
                        if isinstance(seq, str):
                            seq = seq.encode('ascii')
                        
                        # Convert sequence to tokens
                        tokens = []
                        for c in seq:
                            tokens.append(bytes([c]))
                        
                        # Apply all current merges
                        for pair, replacement in self.merges.items():
                            i = 0
                            while i < len(tokens) - 1:
                                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                                    tokens[i] = replacement
                                    del tokens[i + 1]
                                else:
                                    i += 1
                        
                        # Count adjacent pairs in this processed sequence
                        for i in range(len(tokens) - 1):
                            pair = (tokens[i], tokens[i + 1])
                            pair_counts[pair] += 1
                
                # Explicitly free memory after processing each gene
                gc.collect()
            
            if not pair_counts:
                if verbose:
                    print("No more pairs to merge. Stopping early.")
                break
            
            # Find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]
            
            if verbose:
                print(f"Iteration {iteration}: Merging {best_pair[0].decode('ascii', errors='replace')},{best_pair[1].decode('ascii', errors='replace')} with frequency {best_count}")
                print(f"Current vocab size: {len(vocab)}")
            
            # Create a new token by merging the pair
            new_token = best_pair[0] + best_pair[1]
            
            # Add to vocabulary and merges
            vocab[new_token] = next_id
            self.merges[best_pair] = new_token
            self.merge_ids[next_id] = (vocab[best_pair[0]], vocab[best_pair[1]])
            next_id += 1
            
            # Clear pair counts to free memory
            pair_counts.clear()
            gc.collect()
        
        # Update the internal vocabulary
        self.vocab = vocab
        
        # Update the alphabet - rebuild the chars array
        base_chars = np.frombuffer(b"ATCG", dtype=np.uint8)
        new_chars = []
        
        # Collect unique bytes from all tokens
        all_bytes = set()
        for token in self.vocab.keys():
            for b in token:
                all_bytes.add(b)
        
        # Convert to sorted list and then to numpy array
        all_bytes_list = sorted(list(all_bytes))
        self.chars = np.array(all_bytes_list, dtype=np.uint8)
        
        self.size = len(self.vocab)
        self.encoding = np.zeros(256, dtype=np.uint8) + 255
        self.encoding[self.chars] = np.arange(len(self.chars))
        
        self.is_trained = True
        
        if verbose:
            print(f"Training complete. Final vocabulary size: {len(vocab)}")
            
            # Calculate compression statistics on a sample
            sample_genes = list(sequence_dict.keys())[:5]  # Take first 5 genes
            sample_sequences = []
            for gene in sample_genes:
                sample_sequences.extend(sequence_dict[gene][:20])  # Take up to 20 sequences per gene
            
            original_length = sum(len(seq) for seq in sample_sequences)
            tokenized_length = sum(len(self.tokenize(seq)) for seq in sample_sequences)
            compression_ratio = original_length / tokenized_length if tokenized_length > 0 else 0
            print(f"Sample compression ratio: {compression_ratio:.2f}x (on {len(sample_sequences)} sequences)")
            print(f"Sample original length: {original_length}, Tokenized length: {tokenized_length}")
            
        return self
    
    def tokenize(self, sequence):
        """
        Tokenize a DNA sequence using the learned BPE merges
        
        Args:
            sequence: A DNA sequence (string or byte string)
            
        Returns:
            List of token IDs
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before tokenization")
            
        if isinstance(sequence, str):
            sequence = sequence.encode('ascii')
        
        # Start with character-level tokens
        tokens = [bytes([c]) for c in sequence]
        
        # Apply merges in order of addition (not optimal but simpler implementation)
        for pair, replacement in self.merges.items():
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i] = replacement
                    del tokens[i + 1]
                else:
                    i += 1
        
        # Convert tokens to IDs
        token_ids = [self.vocab[token] for token in tokens]
        return token_ids
    
    def detokenize(self, token_ids):
        """
        Convert token IDs back to a DNA sequence
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Original DNA sequence as a byte string
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before detokenization")
            
        sequence = b''
        for token_id in token_ids:
            # For basic nucleotides (A, T, C, G)
            if token_id < 4:
                sequence += bytes([self.chars[token_id]])
            else:
                # For merged tokens, recursively expand using merge_ids
                def expand_token(tid):
                    if tid < 4:  # Basic nucleotide
                        return bytes([self.chars[tid]])
                    else:
                        left_id, right_id = self.merge_ids[tid]
                        return expand_token(left_id) + expand_token(right_id)
                
                sequence += expand_token(token_id)
        
        return sequence
    
    def encode_sequence(self, sequence):
        """
        Encode a DNA sequence to token IDs
        
        Args:
            sequence: DNA sequence as string or bytes
            
        Returns:
            NumPy array of token IDs
        """
        token_ids = self.tokenize(sequence)
        return np.array(token_ids, dtype=np.uint8)
    
    def decode_sequence(self, token_ids):
        """
        Decode token IDs to a DNA sequence
        
        Args:
            token_ids: NumPy array or list of token IDs
            
        Returns:
            DNA sequence as a byte string
        """
        return self.detokenize(token_ids)
    
    def save(self, path, format='json'):
        """
        Save the trained tokenizer to a file
        
        Args:
            path: File path to save the tokenizer
            format: 'json' or 'pickle' (pickle is more compact but less human-readable)
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before saving")
        
        if format == 'json':
            # Convert bytes to strings for JSON serialization
            serializable_vocab = {}
            for k, v in self.vocab.items():
                key = k.decode('ascii', errors='replace')
                serializable_vocab[key] = v
            
            # Convert byte pairs to strings for JSON serialization
            serializable_merges = {}
            for (pair_left, pair_right), replacement in self.merges.items():
                key = (pair_left.decode('ascii', errors='replace'), pair_right.decode('ascii', errors='replace'))
                serializable_merges[str(key)] = replacement.decode('ascii', errors='replace')
            
            # Convert numeric keys to strings for JSON serialization
            serializable_merge_ids = {str(k): v for k, v in self.merge_ids.items()}
            
            # Prepare data for saving
            data = {
                'vocab_size': self.vocab_size,
                'vocab': serializable_vocab,
                'merges': serializable_merges,
                'merge_ids': serializable_merge_ids,
                'chars': self.chars.tobytes().decode('ascii', errors='replace'),
                'size': self.size,
                'is_trained': self.is_trained
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump({
                    'vocab_size': self.vocab_size,
                    'vocab': self.vocab,
                    'merges': self.merges,
                    'merge_ids': self.merge_ids,
                    'chars': self.chars,
                    'size': self.size,
                    'encoding': self.encoding,
                    'is_trained': self.is_trained
                }, f)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")
        
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path, format='json'):
        """
        Load a trained tokenizer from a file
        
        Args:
            path: File path to load the tokenizer from
            format: 'json' or 'pickle' (must match the format used for saving)
            
        Returns:
            Loaded DNATokenizer instance
        """
        if format == 'json':
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Create a new tokenizer with the saved vocab size
            tokenizer = cls(vocab_size=data['vocab_size'])
            
            # Convert string keys back to bytes
            tokenizer.vocab = {k.encode('ascii'): v for k, v in data['vocab'].items()}
            
            # Convert string pairs back to byte pairs
            tokenizer.merges = {}
            for pair_str, replacement_str in data['merges'].items():
                # Extract the tuple from the string representation
                pair_str = pair_str.replace("('", "").replace("')", "").replace("', '", "','")
                left, right = pair_str.split(",")
                left = left.strip()[1:-1]  # Remove quotes
                right = right.strip()[1:-1]  # Remove quotes
                
                tokenizer.merges[(left.encode('ascii'), right.encode('ascii'))] = replacement_str.encode('ascii')
            
            # Convert string keys back to integers for merge_ids
            tokenizer.merge_ids = {}
            for k, v in data['merge_ids'].items():
                tokenizer.merge_ids[int(k)] = v
            
            # Restore chars array
            tokenizer.chars = np.frombuffer(data['chars'].encode('ascii'), dtype=np.uint8)
            tokenizer.size = data['size']
            
            # Reset encoding
            tokenizer.encoding = np.zeros(256, dtype=np.uint8) + 255
            tokenizer.encoding[tokenizer.chars] = np.arange(len(tokenizer.chars))
            
            tokenizer.is_trained = data['is_trained']
        
        elif format == 'pickle':
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Create a new tokenizer with the saved vocab size
            tokenizer = cls(vocab_size=data['vocab_size'])
            
            # Restore all properties
            tokenizer.vocab = data['vocab']
            tokenizer.merges = data['merges']
            tokenizer.merge_ids = data['merge_ids']
            tokenizer.chars = data['chars']
            tokenizer.size = data['size']
            tokenizer.encoding = data['encoding']
            tokenizer.is_trained = data['is_trained']
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")
        
        print(f"Tokenizer loaded from {path}")
        return tokenizer


def train_or_load_tokenizer(sequence_dict, vocab_size=100, tokenizer_path=None, format='pickle', 
                          force_retrain=False, batch_size=100, sample_size=None):
    """
    Utility function to either load a trained tokenizer or train a new one
    
    Args:
        sequence_dict: Dictionary where keys are genes and values are lists of sequences
        vocab_size: Target vocabulary size
        tokenizer_path: Path to save/load the tokenizer
        format: 'json' or 'pickle'
        force_retrain: If True, always train a new tokenizer even if one exists
        batch_size: Number of sequences to process at once
        sample_size: If provided, sample this many sequences from each gene (None for all)
        
    Returns:
        Trained DNATokenizer
    """
    if tokenizer_path and os.path.exists(tokenizer_path) and not force_retrain:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        return DNATokenizer.load(tokenizer_path, format=format)
    else:
        print(f"Training new tokenizer with vocab size {vocab_size}")
        tokenizer = DNATokenizer(vocab_size=vocab_size)
        tokenizer.train_on_dict(sequence_dict, verbose=False, batch_size=batch_size, sample_size=sample_size)
        
        if tokenizer_path:
            tokenizer.save(tokenizer_path, format=format)
        
        return tokenizer


# Example usage:
if __name__ == "__main__":
    # Example DNA sequence dictionary
    with open("data/hits.pkl", "rb") as f: 
        sequence_dict = pickle.load(f)
    
    # Train or load tokenizer
    tokenizer_path = "dna_tokenizer.pkl"
    tokenizer = train_or_load_tokenizer(
        sequence_dict=sequence_dict,
        vocab_size=30,
        tokenizer_path=tokenizer_path,
        format='pickle',
        force_retrain=False,  # Set to True to force retraining
        batch_size=2,
        sample_size=None  # Set to an integer to sample sequences
    )
    
    # Test tokenization and detokenization
    for gene, sequences in list(sequence_dict.items())[:1]:  # Test on first gene
        print(f"\nTesting gene: {gene}")
        for seq in sequences[:2]:  # Test on first two sequences
            print(f"\nOriginal: {seq}")
            tokens = tokenizer.tokenize(seq)
            print(f"Tokenized: {tokens}")
            reconstructed = tokenizer.detokenize(tokens).decode('ascii')
            print(f"Reconstructed: {reconstructed}")
            print(f"Correct reconstruction: {seq == reconstructed}")