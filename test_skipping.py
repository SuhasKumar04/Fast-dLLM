import sys
sys.path.append('./skipping')

from evaluation import run_evaluation

# Run evaluation
run_evaluation(
    model_name='GSAI-ML/LLaDA-8B-Instruct',
    output_dir='./results',
    num_samples=10  # Use 10 test prompts
)