Compute-Skipping Implementation

Task:
Implementation of two compute-skipping policies for Fast-dLLM v2 to reduce computational cost while maintaining generation quality.

Step 1) 
  Model run on Colab:
  Pre-existing Issues with colab and model compatibility:
  
  Error: CUDA OOM when loading LLaDA-8B-Instruct model
  Root Cause: Model requires ~16GB memory, Colab T4 GPU has only 14.74GB
  Impact: Cannot run baseline before implementing modifications
  
  Solution
  Apply 8-bit quantization to reduce memory footprint from ~16GB to ~8-9GB

  Step 2) 
    Implementation Overview
      Token Level Skipping Across Denoising Steps Mechanism:
      1) Compute cosine similarity between token hidden states at consecutive denoising steps
      2) Skip tokens where similarity > threshold
      3) Reuse previous step's outputs for skipped tokens

  Step 3)
    Implementation Overview
      Layer-Level Skipping Within Denoising Mechanism:
      1) Compute cosine similarity between input hidden states of adjacent layers
      2) Skip layers where similarity > threshold
      3) Reuse previous layer's output for skipped layers

File Structure:
├── token_skipping.py          # Token-level skipping mechanism
├── layer_skipping.py          # Layer-level skipping + combined wrapper
├── evaluation.py              # Evaluation framework and plotting
├── llada/results              # contains results from testing

Conclusion
This implementation provides a framework for reducing computational cost in Fast-dLLM v2 through two skipping mechanisms:

1) Token-level skipping
2) Layer-level skipping

EVALUATION SUMMARY

Best FLOPs Reduction: 2.8%
  Token threshold: 0.90
  Layer threshold: 0.90

Average combined reduction: 1.4%
======================================================================

