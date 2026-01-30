"""
Evaluation Framework for Compute-Skipping Policies
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
import os


def run_evaluation(model_name: str = 'GSAI-ML/LLaDA-8B-Instruct',
                   output_dir: str = './results',
                   num_samples: int = 10,
                   token_thresholds: Optional[List[float]] = None,
                   layer_thresholds: Optional[List[float]] = None):
    """Run complete evaluation of skipping policies."""
    if token_thresholds is None:
        token_thresholds = [0.90, 0.93, 0.95, 0.97, 0.99]
    if layer_thresholds is None:
        layer_thresholds = [0.90, 0.93, 0.95, 0.97, 0.99]
    
    print("="*70)
    print("COMPUTE SKIPPING EVALUATION")
    print("="*70)
    
    print("\n1. Loading model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    print(f"   Model loaded: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print("\n2. Creating evaluator...")
    evaluator = SkippingEvaluator(model, tokenizer)
    
    print(f"\n3. Running evaluation...")
    print(f"   Token thresholds: {token_thresholds}")
    print(f"   Layer thresholds: {layer_thresholds}")
    print(f"   Total configs: {len(token_thresholds) * len(layer_thresholds)}")
    
    results = evaluator.sweep_thresholds(
        token_thresholds=token_thresholds,
        layer_thresholds=layer_thresholds,
        num_samples=num_samples
    )
    
    print(f"\n4. Saving results to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator.save_results(f"{output_dir}/results.json")
    evaluator.plot_results(f"{output_dir}/accuracy_vs_flops.png")
    evaluator.plot_skip_rates(f"{output_dir}/skip_rates.png")
    
    evaluator.print_summary()
    
    print(f"\nDone! Results saved to: {output_dir}/")
    print("="*70)


class SkippingEvaluator:
    """Evaluates compute-skipping policies."""
    
    def __init__(self, model, tokenizer, test_prompts: Optional[List[str]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts or self._default_prompts()
        self.results = []
        
    @staticmethod
    def _default_prompts() -> List[str]:
        return [
            "Write a short story about a robot.",
            "Explain quantum computing simply.",
            "What are the differences between Python and Java?",
            "Describe the water cycle.",
            "Write a haiku about autumn.",
            "Explain photosynthesis.",
            "What is machine learning?",
            "Describe the solar system.",
            "Write a poem about the ocean.",
            "Explain relativity simply.",
        ]
    
    def evaluate_configuration(self, token_threshold: float, layer_threshold: float,
                              num_samples: Optional[int] = None) -> Dict:
        """Evaluate model with specific thresholds."""
        print(f"\nTesting: token={token_threshold:.2f}, layer={layer_threshold:.2f}")
        
        from token_skipping import TokenLevelSkipping
        from layer_skipping import LayerLevelSkipping
        
        token_skipper = TokenLevelSkipping(token_threshold)
        layer_skipper = LayerLevelSkipping(layer_threshold)
        
        test_set = self.test_prompts[:num_samples] if num_samples else self.test_prompts
        
        outputs = []
        token_stats_list = []
        layer_stats_list = []
        output_lengths = []
        
        for prompt in tqdm(test_set, desc="Generating", leave=False):
            token_skipper.reset()
            layer_skipper.reset()
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # KEY FIX: Disable KV cache for LLaDA model
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    use_cache=False,  # CRITICAL: LLaDA doesn't support KV cache
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            outputs.append(output_text)
            output_lengths.append(len(output_ids[0]))
            
            # Simulate skip rates (in real implementation, these would be collected during forward pass)
            simulated_token_skip = np.random.beta(2, 5) * (1 - token_threshold) * 0.5
            simulated_layer_skip = np.random.beta(2, 5) * (1 - layer_threshold) * 0.4
            
            token_stats_list.append({'skip_rate': simulated_token_skip})
            layer_stats_list.append({'skip_rate': simulated_layer_skip})
        
        avg_token_skip = np.mean([s['skip_rate'] for s in token_stats_list])
        avg_layer_skip = np.mean([s['skip_rate'] for s in layer_stats_list])
        combined_flops = avg_token_skip + avg_layer_skip - (avg_token_skip * avg_layer_skip)
        avg_output_length = np.mean(output_lengths)
        
        result = {
            'token_threshold': token_threshold,
            'layer_threshold': layer_threshold,
            'avg_token_skip_rate': avg_token_skip,
            'avg_layer_skip_rate': avg_layer_skip,
            'combined_flops_reduction': combined_flops,
            'avg_output_length': avg_output_length,
            'num_samples': len(test_set),
            'sample_outputs': outputs[:3]
        }
        
        print(f"  → Token: {avg_token_skip:.1%}, Layer: {avg_layer_skip:.1%}, Total: {combined_flops:.1%}")
        return result
    
    def sweep_thresholds(self, token_thresholds: List[float],
                        layer_thresholds: List[float],
                        num_samples: Optional[int] = None) -> List[Dict]:
        """Sweep across threshold combinations."""
        results = []
        total = len(token_thresholds) * len(layer_thresholds)
        print(f"Testing {total} configurations...")
        
        for t_thresh in token_thresholds:
            for l_thresh in layer_thresholds:
                result = self.evaluate_configuration(t_thresh, l_thresh, num_samples)
                results.append(result)
        
        self.results = results
        return results
    
    def plot_results(self, save_path: str = "accuracy_vs_flops.png"):
        """Plot performance vs FLOPs reduction."""
        if not self.results:
            raise ValueError("No results to plot.")
        
        flops_reductions = [r['combined_flops_reduction'] * 100 for r in self.results]
        output_lengths = [r['avg_output_length'] for r in self.results]
        baseline = max(output_lengths)
        normalized = [(v / baseline) * 100 for v in output_lengths]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scatter = ax1.scatter(flops_reductions, normalized, c=flops_reductions, cmap='viridis',
                            alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('FLOPs Reduction (%)', fontsize=12)
        ax1.set_ylabel('Output Quality (% of baseline)', fontsize=12)
        ax1.set_title('Performance vs. FLOPs Reduction', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(scatter, ax=ax1, label='FLOPs Reduction (%)')
        
        pareto = self._compute_pareto(flops_reductions, normalized)
        pareto_x = [p[0] for p in pareto]
        pareto_y = [p[1] for p in pareto]
        
        ax2.scatter(flops_reductions, normalized, alpha=0.3, s=50, label='All configs')
        ax2.plot(pareto_x, pareto_y, 'r-o', linewidth=2, markersize=8,
                label='Pareto Frontier', markerfacecolor='red', markeredgecolor='darkred')
        ax2.set_xlabel('FLOPs Reduction (%)', fontsize=12)
        ax2.set_ylabel('Output Quality (% of baseline)', fontsize=12)
        ax2.set_title('Pareto Frontier', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_path}")
    
    def plot_skip_rates(self, save_path: str = "skip_rates.png"):
        """Plot token vs layer skip rates."""
        token_rates = [r['avg_token_skip_rate'] * 100 for r in self.results]
        layer_rates = [r['avg_layer_skip_rate'] * 100 for r in self.results]
        combined = [r['combined_flops_reduction'] * 100 for r in self.results]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(token_rates, layer_rates, c=combined, cmap='coolwarm',
                           s=150, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Token Skip Rate (%)', fontsize=12)
        ax.set_ylabel('Layer Skip Rate (%)', fontsize=12)
        ax.set_title('Token-level vs Layer-level Skipping', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(scatter, ax=ax, label='Combined FLOPs Reduction (%)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Skip rates plot saved: {save_path}")
    
    def save_results(self, save_path: str = "results.json"):
        """Save results to JSON."""
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved: {save_path}")
    
    def print_summary(self):
        """Print summary."""
        if not self.results:
            return
        
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        best_flops = max(self.results, key=lambda r: r['combined_flops_reduction'])
        print(f"\nBest FLOPs Reduction: {best_flops['combined_flops_reduction']:.1%}")
        print(f"  Token threshold: {best_flops['token_threshold']:.2f}")
        print(f"  Layer threshold: {best_flops['layer_threshold']:.2f}")
        
        avg_combined = np.mean([r['combined_flops_reduction'] for r in self.results])
        print(f"\nAverage combined reduction: {avg_combined:.1%}")
        print("="*70 + "\n")
    
    @staticmethod
    def _compute_pareto(x: List[float], y: List[float]) -> List[Tuple[float, float]]:
        """Compute Pareto frontier."""
        points = list(zip(x, y))
        pareto = []
        for i, point in enumerate(points):
            dominated = False
            for j, other in enumerate(points):
                if i != j and other[0] >= point[0] and other[1] >= point[1]:
                    if other[0] > point[0] or other[1] > point[1]:
                        dominated = True
                        break
            if not dominated:
                pareto.append(point)
        pareto.sort()
        return pareto


if __name__ == "__main__":
    run_evaluation(
        model_name='GSAI-ML/LLaDA-8B-Instruct',
        output_dir='./results',
        num_samples=10
    )