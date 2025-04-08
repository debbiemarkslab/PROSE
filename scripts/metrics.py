import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import precision_recall_curve, auc
import os

import logging

# Configure logger
logger = logging.getLogger("benchmark_utils")

class BenchmarkLogger:
    """
    Class to evaluate and log ProSE model performance on genomic benchmarks.
    Supports tracking of model performance throughout training on three benchmarks:
    - eQTL causal variant prediction
    - Ultra-rare variant prioritization
    - Transcription factor binding site (TFBS) disruption
    """
    
    def __init__(self,  data_dir, log_dir, device='cuda', log_freq=5000):
        """
        Initialize the benchmark logger
        
        Args:
            data_dir: Base directory containing benchmark data
            log_dir: Directory to save intermediate results
            device: Device to run inference on ('cuda' or 'cpu')
            log_freq: Frequency of logging in training steps
        """
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.device = device
        self.log_freq = log_freq
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize benchmark datasets
        self._load_datasets()
    
    def _load_datasets(self):
        """Load all benchmark datasets"""
        logger.info("Loading benchmark datasets...")
        # self._load_eqtl_data()
        # self._load_rare_variant_data()
        self._load_tfbs_data()
        logger.info("All benchmark datasets loaded successfully")
    
    def _load_eqtl_data(self):
        """Load eQTL benchmark data"""
        # Load the eQTL dataset
        eqtl_path = os.path.join(self.data_dir, 'eQTLs_causal/indels/data/eQTLS_indel_with_full_seqs_no_dups.csv')
        self.eqtl_df = pd.read_csv(eqtl_path)
        
        # Calculate indel length for filtering
        self.eqtl_df['REF_len'] = self.eqtl_df['REF'].apply(len)
        self.eqtl_df['ALT_len'] = self.eqtl_df['ALT'].apply(len)
        self.eqtl_df['indel_length'] = abs(self.eqtl_df['REF_len'] - self.eqtl_df['ALT_len'])
        
        # Filter the data based on PIP scores
        self.eqtl_df['pip_group'] = self.eqtl_df.apply(
            lambda row: 'causal' if row['pip'] >= 0.95 else 
                       ('background' if row['pip'] <= 0.01 else 'drop'), axis=1)
        self.eqtl_df = self.eqtl_df[self.eqtl_df['pip_group'] != 'drop']
        
        # Prepare sequences for scoring
        self.eqtl_wt_seqs = []
        self.eqtl_var_seqs = []
       
        
        for _, row in self.eqtl_df.iterrows():
            self.eqtl_wt_seqs.append(row['WT'])
            self.eqtl_var_seqs.append(row['VAR'])
           
        
        logger.info(f"Loaded eQTL dataset with {len(self.eqtl_df)} variants")
        logger.info(f"  Causal variants: {len(self.eqtl_df[self.eqtl_df['pip_group'] == 'causal'])}")
        logger.info(f"  Background variants: {len(self.eqtl_df[self.eqtl_df['pip_group'] == 'background'])}")
    
    def _load_rare_variant_data(self):
        """Load ultra-rare variant benchmark data"""
        # Load the gnomAD dataset with ultra-rare and common variants
        rare_path = os.path.join(self.data_dir, 'rare_variants/gnomad_promoter_indels.csv')
        self.rare_df = pd.read_csv(rare_path)
        
        # Classify variants as ultra-rare or common based on MAF
        self.rare_df['variant_class'] = self.rare_df['af'].apply(
            lambda x: 'ultra_rare' if x < 0.001 else 'common')
        
        # Prepare sequences for scoring
        self.rare_wt_seqs = []
        self.rare_var_seqs = []
      
        
        for _, row in self.rare_df.iterrows():
            self.rare_wt_seqs.append(row['WT'])
            self.rare_var_seqs.append(row['VAR'])
          
        
        logger.info(f"Loaded rare variant dataset with {len(self.rare_df)} variants")
        logger.info(f"  Ultra-rare variants: {len(self.rare_df[self.rare_df['variant_class'] == 'ultra_rare'])}")
        logger.info(f"  Common variants: {len(self.rare_df[self.rare_df['variant_class'] == 'common'])}")
    
    def _load_tfbs_data(self):
        """Load TFBS disruption benchmark data"""
        # Load the TFBS dataset with consistently and variably expressed genes
        # tfbs_path = os.path.join(self.data_dir, 'tfbs_disruption/tfbs_knockouts.csv')
        tfbs_path = os.path.join(self.data_dir, 'tfbs_with_expression_types.csv')
        self.tfbs_df = pd.read_csv(tfbs_path)
        
        # Prepare sequences for scoring
        self.tfbs_wt_seqs = []
        self.tfbs_var_seqs = []
        
        for _, row in self.tfbs_df.iterrows():
            self.tfbs_wt_seqs.append(row['WT'])
            self.tfbs_var_seqs.append(row['VAR'])
            
        
        logger.info(f"Loaded TFBS dataset with {len(self.tfbs_df)} variants")
        logger.info(f"  Consistently expressed: {len(self.tfbs_df[self.tfbs_df['expression'] == 'consistent'])}")
        logger.info(f"  Variably expressed: {len(self.tfbs_df[self.tfbs_df['expression'] == 'variable'])}")
    
    
    def _calculate_cohens_d(self, x, y):
        """Calculate Cohen's d effect size between two distributions
        
        Args:
            x: First distribution
            y: Second distribution
            
        Returns:
            Cohen's d value
        """
        nx = len(x)
        ny = len(y)
        
        # If either group is empty, return NaN
        if nx == 0 or ny == 0:
            return np.nan
        
        # Calculate means
        mean_x = np.nanmean(x)
        mean_y = np.nanmean(y)
        
        # Calculate variances
        var_x = np.nanvar(x, ddof=1)
        var_y = np.nanvar(y, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt((nx * var_x + ny * var_y) / (nx + ny))
        
        # Cohen's d
        d = (mean_x - mean_y) / pooled_std if pooled_std > 0 else np.nan
        
        return d
    
    def score_eqtl_benchmark(self, scores):
        """Score eQTL dataset and calculate metrics
        
        Returns:
            Dictionary of metrics
        """
        logger.info("Scoring eQTL benchmark...")
        
        # # Add scores to DataFrame
        self.eqtl_df['ProSE_avg_score'] = scores
        # self.eqtl_df['ProSE_forward_score'] = [score[1] for score in scores]
        
        # Calculate metrics
        causal_scores = np.abs(self.eqtl_df[self.eqtl_df['pip_group'] == 'causal']['ProSE_avg_score'])
        bg_scores = np.abs(self.eqtl_df[self.eqtl_df['pip_group'] == 'background']['ProSE_avg_score'])
        
        # Cohen's ds
        cohens_d = self._calculate_cohens_d(causal_scores, bg_scores)
        
        # AUPRC
        self.eqtl_df['binary_label'] = self.eqtl_df['pip_group'].apply(lambda x: 1 if x == 'causal' else 0)
        p, r, _ = precision_recall_curve(self.eqtl_df['binary_label'], np.abs(self.eqtl_df['ProSE_avg_score']))
        auprc = auc(r, p)
        
        # Normalized AUPRC
        baseline = len(self.eqtl_df[self.eqtl_df['pip_group'] == 'causal']) / len(self.eqtl_df)
        nauprc = auprc / baseline
        
        # Calculate metrics for indels > 5bp
        long_indels_mask = (self.eqtl_df['indel_length'] > 5)
        long_df = self.eqtl_df[long_indels_mask]
        
        if len(long_df) > 0:
            long_causal_scores = np.abs(long_df[long_df['pip_group'] == 'causal']['ProSE_avg_score'])
            long_bg_scores = np.abs(long_df[long_df['pip_group'] == 'background']['ProSE_avg_score'])
            
            if len(long_causal_scores) > 0 and len(long_bg_scores) > 0:
                cohens_d_long = self._calculate_cohens_d(long_causal_scores, long_bg_scores)
                
                p_long, r_long, _ = precision_recall_curve(long_df['binary_label'], np.abs(long_df['ProSE_avg_score']))
                auprc_long = auc(r_long, p_long)
                
                baseline_long = len(long_df[long_df['pip_group'] == 'causal']) / len(long_df)
                nauprc_long = auprc_long / baseline_long
            else:
                cohens_d_long = np.nan
                auprc_long = np.nan
                nauprc_long = np.nan
        else:
            cohens_d_long = np.nan
            auprc_long = np.nan
            nauprc_long = np.nan
        
        return {
            'eqtl_cohens_d': cohens_d,
            'eqtl_auprc': auprc,
            'eqtl_nauprc': nauprc,
            'eqtl_cohens_d_long': cohens_d_long,
            'eqtl_auprc_long': auprc_long,
            'eqtl_nauprc_long': nauprc_long
        }
    
    def score_rare_variant_benchmark(self, scores):
        """Score rare variant dataset and calculate enrichment metrics
        
        Returns:
            Dictionary of enrichment metrics
        """
        logger.info("Scoring rare variant benchmark...")
       
        
        # Add scores to DataFrame
        self.rare_df['ProSE_avg_score'] = scores
        self.rare_df.dropna(inplace=True)

        # self.rare_df['ProSE_forward_score'] = [score[1] for score in scores]
        
        # Calculate enrichment for multiple percentile thresholds
        percentiles = [0.001, 0.01, 0.1, 1.0]
        enrichment_results = {}
        
        for percentile in percentiles:
            # Calculate score threshold for this percentile
            threshold = np.nanpercentile(self.rare_df['ProSE_avg_score'], percentile)
            
            # Count ultra-rare and common variants at this threshold
            ultra_rare_count = len(self.rare_df[(self.rare_df['ProSE_avg_score'] <= threshold) & 
                                              (self.rare_df['variant_class'] == 'ultra_rare')])
            common_count = len(self.rare_df[(self.rare_df['ProSE_avg_score'] <= threshold) & 
                                          (self.rare_df['variant_class'] == 'common')])
            
            # Total variants in each class
            total_ultra_rare = len(self.rare_df[self.rare_df['variant_class'] == 'ultra_rare'])
            total_common = len(self.rare_df[self.rare_df['variant_class'] == 'common'])
            
            # Calculate percentages
            ultra_rare_pct = ultra_rare_count / total_ultra_rare if total_ultra_rare > 0 else 0
            common_pct = common_count / total_common if total_common > 0 else 0
            
            # Calculate enrichment ratio
            ratio = ultra_rare_pct / common_pct if common_pct > 0 else float('inf')
            
            enrichment_results[f'rare_enrichment_{percentile:.3f}'] = ratio
        
        return enrichment_results
    
    def score_tfbs_benchmark(self, scores):
        """Score TFBS disruption dataset and calculate accuracy
        
        Returns:
            Dictionary of TFBS disruption metrics
        """
        logger.info("Scoring TFBS benchmark...")
        
        # Add scores to DataFrame
        self.tfbs_df['ProSE_avg_score'] = scores
        # self.tfbs_df['ProSE_forward_score'] = [score[1] for score in scores]
        self.tfbs_df.dropna(inplace=True)
        # Group by TF and calculate accuracy for each
        delta_accuracies = []
        
        # For each TF, compare consistent vs variable expression genes
        for tf in self.tfbs_df['TF'].unique():
            tf_data = self.tfbs_df[self.tfbs_df['TF'] == tf]
            
            # Skip TFs with insufficient data
            if len(tf_data) < 2:
                continue
            
            consistent_scores = tf_data[tf_data['expression'] == 'consistent']['ProSE_avg_score']
            variable_scores = tf_data[tf_data['expression'] == 'variable']['ProSE_avg_score']
            
            # Skip if not enough data in either group
            if len(consistent_scores) == 0 or len(variable_scores) == 0:
                continue
            
            # Count pairs where consistent is more deleterious (lower score) than variable
            correct_pairs = 0
            total_pairs = 0
            
            for c_score in consistent_scores:
                for v_score in variable_scores:
                    total_pairs += 1
                    if c_score < v_score:  # Lower score = more deleterious
                        correct_pairs += 1
            
            accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0.5
            delta_accuracy = accuracy - 0.5  # Difference from random
            
            delta_accuracies.append(delta_accuracy)
        
        # Calculate overall metrics
        mean_delta_accuracy = np.nanmean(delta_accuracies) if delta_accuracies else 0
        median_delta_accuracy = np.nanmedian(delta_accuracies) if delta_accuracies else 0
        positive_fraction = np.nanmean([d > 0 for d in delta_accuracies]) if delta_accuracies else 0
        
        return {
            'tfbs_mean_delta_accuracy': mean_delta_accuracy,
            'tfbs_median_delta_accuracy': median_delta_accuracy,
            'tfbs_positive_fraction': positive_fraction
        }
   
