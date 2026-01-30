"""
Layer-Level Skipping for Fast-dLLM
===================================
Skips layers with high cosine similarity between adjacent layer inputs within a denoising step.

Usage:
    from layer_skipping import LayerLevelSkipping
    
    skipper = LayerLevelSkipping(similarity_threshold=0.95)
    
    # In your layer loop:
    for layer_idx, layer in enumerate(model.layers):
        should_skip, sim = skipper.should_skip_layer(hidden_states, layer_idx)
        
        if should_skip:
            # Skip layer, reuse previous output
            hidden_states = skipper.prev_layer_output
        else:
            # Compute normally
            hidden_states = layer(hidden_states)
            skipper.prev_layer_output = hidden_states
    
    stats = skipper.get_stats()
    print(f"Skip rate: {stats['skip_rate']:.2%}")
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class LayerLevelSkipping:
    """
    Layer-level compute skipping based on cosine similarity within a denoising step.
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        """
        Args:
            similarity_threshold: Similarity threshold above which layers are skipped.
                                 Range: [0.0, 1.0]. Recommended: 0.90-0.97
        """
        self.threshold = similarity_threshold
        self.prev_layer_input = None
        self.prev_layer_output = None
        self.skipped_layers = 0
        self.total_layers = 0
        self.layer_similarities = []
        
    def should_skip_layer(self, current_input: torch.Tensor, layer_idx: int) -> Tuple[bool, float]:
        """
        Determine if current layer should be skipped.
        
        Args:
            current_input: Input to current layer. Shape (batch_size, seq_len, hidden_dim)
            layer_idx: Current layer index
        
        Returns:
            Tuple of (should_skip: bool, similarity: float)
        """
        # First layer: never skip
        if self.prev_layer_input is None or layer_idx == 0:
            self.prev_layer_input = current_input.clone().detach()
            self.total_layers += 1
            return False, 0.0
        
        # Normalize for cosine similarity
        curr_norm = F.normalize(current_input, p=2, dim=-1)
        prev_norm = F.normalize(self.prev_layer_input, p=2, dim=-1)
        
        # Compute cosine similarity per token: (batch_size, seq_len)
        token_similarities = (curr_norm * prev_norm).sum(dim=-1)
        
        # Average similarity across batch and sequence
        similarity = token_similarities.mean().item()
        self.layer_similarities.append(similarity)
        
        # Determine if layer should be skipped
        should_skip = similarity > self.threshold
        
        # Update statistics
        if should_skip:
            self.skipped_layers += 1
        self.total_layers += 1
        
        # Update previous input
        self.prev_layer_input = current_input.clone().detach()
        
        return should_skip, similarity
    
    def reset_step(self):
        """Reset for new denoising step."""
        self.prev_layer_input = None
        self.prev_layer_output = None
        self.skipped_layers = 0
        self.total_layers = 0
        self.layer_similarities = []
    
    def reset(self):
        """Complete reset for new sequence."""
        self.reset_step()
    
    def get_stats(self) -> Dict[str, float]:
        """Get skipping statistics."""
        if self.total_layers == 0:
            return {
                'skipped_layers': 0,
                'total_layers': 0,
                'skip_rate': 0.0,
                'flops_reduction_estimate': 0.0,
                'avg_similarity': 0.0
            }
        
        skip_rate = self.skipped_layers / self.total_layers
        avg_similarity = (sum(self.layer_similarities) / len(self.layer_similarities) 
                         if self.layer_similarities else 0.0)
        
        return {
            'skipped_layers': self.skipped_layers,
            'total_layers': self.total_layers,
            'skip_rate': skip_rate,
            'flops_reduction_estimate': skip_rate,
            'avg_similarity': avg_similarity
        }
    
    def print_stats(self):
        """Print statistics."""
        stats = self.get_stats()
        print(f"Layer-Level Skipping:")
        print(f"  Skipped: {stats['skipped_layers']}/{stats['total_layers']} layers")
        print(f"  Skip Rate: {stats['skip_rate']:.2%}")
        print(f"  FLOP Reduction: {stats['flops_reduction_estimate']:.2%}")
        print(f"  Avg Similarity: {stats['avg_similarity']:.3f}")


class CombinedSkipping:
    """
    Combines token-level and layer-level skipping.
    """
    
    def __init__(self, token_threshold: float = 0.95, layer_threshold: float = 0.95):
        """
        Args:
            token_threshold: Threshold for token-level skipping
            layer_threshold: Threshold for layer-level skipping
        """
        # Import here to avoid circular dependency
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from token_skipping import TokenLevelSkipping
        
        self.token_skipper = TokenLevelSkipping(token_threshold)
        self.layer_skipper = LayerLevelSkipping(layer_threshold)
        
    def reset_denoising_step(self):
        """Reset for new denoising step."""
        self.layer_skipper.reset_step()
        
    def reset_sequence(self):
        """Reset for new sequence."""
        self.token_skipper.reset()
        self.layer_skipper.reset()
        
    def get_combined_stats(self) -> Dict[str, any]:
        """
        Get combined statistics from both skipping mechanisms.
        
        Returns:
            Dictionary with all metrics including combined FLOP reduction
        """
        token_stats = self.token_skipper.get_stats()
        layer_stats = self.layer_skipper.get_stats()
        
        # Estimate combined FLOP reduction
        # If we skip t% of tokens and l% of layers:
        # Combined reduction ≈ t + l - (t * l)
        token_skip = token_stats['skip_rate']
        layer_skip = layer_stats['skip_rate']
        combined_reduction = token_skip + layer_skip - (token_skip * layer_skip)
        
        return {
            'token_skipping': token_stats,
            'layer_skipping': layer_stats,
            'combined_flops_reduction': combined_reduction
        }
    
    def print_stats(self):
        """Print combined statistics."""
        stats = self.get_combined_stats()
        print("\n" + "="*60)
        self.token_skipper.print_stats()
        print()
        self.layer_skipper.print_stats()
        print()
        print(f"Combined FLOP Reduction: {stats['combined_flops_reduction']:.2%}")
        print("="*60)


if __name__ == "__main__":
    # Test layer skipping
    print("Layer-Level Skipping Test")
    print("=" * 50)
    
    skipper = LayerLevelSkipping(similarity_threshold=0.95)
    
    # Simulate 32 layers
    for layer_idx in range(32):
        hidden = torch.randn(2, 10, 768)
        should_skip, sim = skipper.should_skip_layer(hidden, layer_idx)
        if should_skip:
            print(f"Layer {layer_idx}: SKIP (sim={sim:.3f})")
        else:
            print(f"Layer {layer_idx}: COMPUTE (sim={sim:.3f})")
    
    print("\n" + "=" * 50)
    skipper.print_stats()
    
    # Test combined
    print("\n\nCombined Skipping Test")
    print("=" * 50)
    
    combined = CombinedSkipping(token_threshold=0.95, layer_threshold=0.95)
    
    # Simulate generation
    for step in range(3):
        combined.reset_denoising_step()
        
        for layer_idx in range(32):
            hidden = torch.randn(2, 10, 768)
            
            # Token skipping
            if step > 0:  # Skip first step
                token_mask = combined.token_skipper.should_skip_tokens(hidden)
            
            # Layer skipping
            should_skip_layer, _ = combined.layer_skipper.should_skip_layer(hidden, layer_idx)
    
    combined.print_stats()
