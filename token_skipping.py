"""
Token-Level Skipping for Fast-dLLM
===================================
Skips tokens with high cosine similarity across consecutive denoising steps.

Usage:
    from token_skipping import TokenLevelSkipping
    
    skipper = TokenLevelSkipping(similarity_threshold=0.95)
    
    # In your denoising loop:
    for step in range(num_steps):
        hidden_states = model.forward(...)
        compute_mask = skipper.should_skip_tokens(hidden_states)
        # Use mask to skip computation
        
    stats = skipper.get_stats()
    print(f"Skip rate: {stats['skip_rate']:.2%}")
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


class TokenLevelSkipping:
    """
    Token-level compute skipping based on cosine similarity across denoising steps.
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        """
        Args:
            similarity_threshold: Similarity threshold above which tokens are skipped.
                                 Range: [0.0, 1.0]. Recommended: 0.90-0.97
        """
        self.threshold = similarity_threshold
        self.prev_hidden_states = None
        self.skipped_tokens = 0
        self.total_tokens = 0
        self.current_step = 0
        
    def should_skip_tokens(self, current_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Determine which tokens should be skipped.
        
        Args:
            current_hidden_states: Shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            Boolean mask: True = compute, False = skip. Shape (batch_size, seq_len)
        """
        batch_size, seq_len, hidden_dim = current_hidden_states.shape
        
        # First step: compute everything
        if self.prev_hidden_states is None or self.current_step == 0:
            compute_mask = torch.ones(
                batch_size, seq_len, 
                dtype=torch.bool, 
                device=current_hidden_states.device
            )
            self.prev_hidden_states = current_hidden_states.clone().detach()
            self.total_tokens += batch_size * seq_len
            self.current_step += 1
            return compute_mask
        
        # Normalize for cosine similarity
        curr_norm = F.normalize(current_hidden_states, p=2, dim=-1)
        prev_norm = F.normalize(self.prev_hidden_states, p=2, dim=-1)
        
        # Compute cosine similarity: (batch_size, seq_len)
        similarity = (curr_norm * prev_norm).sum(dim=-1)
        
        # Create mask: True where similarity <= threshold (needs computation)
        compute_mask = similarity <= self.threshold
        
        # Update statistics
        self.skipped_tokens += (~compute_mask).sum().item()
        self.total_tokens += compute_mask.numel()
        
        # Store current for next comparison
        self.prev_hidden_states = current_hidden_states.clone().detach()
        self.current_step += 1
        
        return compute_mask
    
    def reset(self):
        """Reset for new sequence."""
        self.prev_hidden_states = None
        self.skipped_tokens = 0
        self.total_tokens = 0
        self.current_step = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get skipping statistics."""
        if self.total_tokens == 0:
            return {
                'skipped_tokens': 0,
                'total_tokens': 0,
                'skip_rate': 0.0,
                'flops_reduction_estimate': 0.0,
                'current_step': 0
            }
        
        skip_rate = self.skipped_tokens / self.total_tokens
        
        return {
            'skipped_tokens': self.skipped_tokens,
            'total_tokens': self.total_tokens,
            'skip_rate': skip_rate,
            'flops_reduction_estimate': skip_rate,
            'current_step': self.current_step
        }
    
    def print_stats(self):
        """Print statistics."""
        stats = self.get_stats()
        print(f"Token-Level Skipping:")
        print(f"  Skipped: {stats['skipped_tokens']:,}/{stats['total_tokens']:,}")
        print(f"  Skip Rate: {stats['skip_rate']:.2%}")
        print(f"  FLOP Reduction: {stats['flops_reduction_estimate']:.2%}")


if __name__ == "__main__":
    # Test
    skipper = TokenLevelSkipping(similarity_threshold=0.95)
    
    for step in range(5):
        hidden = torch.randn(2, 10, 768)
        mask = skipper.should_skip_tokens(hidden)
        print(f"Step {step}: {mask.sum().item()}/{mask.numel()} tokens to compute")
    
    skipper.print_stats()
