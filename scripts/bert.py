from poet.alphabets import Alphabet
import numpy as np
from transformers import AutoTokenizer

# from train import ATCG

class DNA(Alphabet):
    """DNA-specific alphabet class compatible with DNABERT2 tokenizer"""
    
    def __init__(self, mask=False):
        # DNA alphabet: A, C, G, T
        chars = b"ACGT"
        
        # Initialize the base Alphabet class
        super().__init__(chars, encoding=None, mask=mask, missing=255)
        
        # Load the DNABERT2 tokenizer
        self.dnabert_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
        
        self.start_token = self.dnabert_tokenizer.cls_token_id
        self.stop_token = self.dnabert_tokenizer.sep_token_id
        self.size = max(self.dnabert_tokenizer.get_vocab().values()) + 1
        self.padding_token = self.dnabert_tokenizer.pad_token_id
        self.mask_token = self.dnabert_tokenizer.mask_token_id


    
    def tokenize_for_dnabert(self, sequence, add_special_tokens=False):
        """Tokenize a DNA sequence using DNABERT2 tokenizer
            add_special_tokens = True will add the start and stop automatically to the two ends
        """
       
        tokens = self.dnabert_tokenizer(
            sequence, 
            add_special_tokens=add_special_tokens,
            return_tensors="pt"
        )
        
        return tokens
    
    def encode(self, sequence, add_special_tokens=False):
        """Encode a DNA sequence for use with DNABERT2"""
        tokens = self.tokenize_for_dnabert(sequence, add_special_tokens=add_special_tokens)
        return tokens["input_ids"].squeeze()
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs back to a DNA sequence"""
        return self.dnabert_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    
    
alphabet = DNA(mask=True)
print(alphabet.size)
print(alphabet.padding_token)
# # a = ATCG()
# seq = alphabet.start_token + "AATTCCGG" + alphabet.stop_token
# print(seq)
# print(alphabet.encode_for_model(seq).shape)
# # print((a.encode(b"AATTCCGG").shape))