# Reading Review: b00001

## Metadata

- **Title**: Large Language Diffusion Models
- **Authors**: Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li
- **Year**: 2025
- **Venue**: NeurIPS 2025
- **arXiv**: 2502.09992v3 (18 Oct 2025)
- **BibKey**: nie2025llada

---

## Section-by-Section Summary

### 1 Introduction

The paper challenges the assumption that autoregressive models (ARMs) are the only viable path to scalable LLMs. The core argument is that the essential properties of LLMs (scalability, in-context learning, instruction-following) stem from generative modeling principles (Eq. 1), not from the left-to-right autoregressive factorization (Eq. 2). The authors introduce **LLaDA** (Large Language Diffusion with mAsking), a masked diffusion model (MDM) trained from scratch at 8B parameters using 2.3T tokens. LLaDA matches LLaMA3 8B Base on most standard benchmarks and outperforms GPT-4o on reversal poem completion.

### 2.1 Probabilistic Formulation

LLaDA defines a model distribution via a forward masking process and a reverse denoising process. The forward process independently masks each token with probability t ∈ U[0,1], so at t=1 all tokens are masked. The core trainable component is a **mask predictor** p_θ(·|x_t), a non-causal Transformer that predicts all masked tokens simultaneously. It is trained with cross-entropy on masked positions only (Eq. 3). This loss is provably an upper bound on the negative log-likelihood (Eq. 4), giving a principled maximum-likelihood objective. The key simplification (Eq. 11): the mask predictor is time-invariant—it only needs the set of unmasked tokens as conditioning, so t need not be fed as input.

### 2.2 Pre-training

LLaDA uses a standard Transformer (no causal mask) as the mask predictor. Two model sizes: 1B and 8B. The 8B model uses vanilla multi-head attention (not GQA, due to incompatibility with KV caching). Pre-training data: 2.3T tokens aligned with LLaMA-style data protocols (code, math, multilingual, Chinese). Fixed sequence length 4096. Total compute: 0.13M H800 GPU hours. Training schedule: Warmup-Stable-Decay with AdamW, batch size 1280.

### 2.3 Supervised Fine-Tuning (SFT)

SFT follows the same masking objective but applies masking only to the response tokens r_0 while keeping prompt p_0 unmasked (Eq. 5). Data: 4.5M pairs spanning code, math, multilingual, and instruction-following. Trained for 3 epochs. No RL alignment was applied (left for future work), which explains some gaps vs. instruction-tuned ARMs on certain metrics.

### 2.4 Inference

Sampling proceeds by simulating the reverse process from t=1 (fully masked) to t=0, over N discrete steps. At each step, the mask predictor predicts all masked tokens; then s/t of the predicted tokens are randomly re-masked (random remasking strategy, Algorithm 4). Low-confidence remasking is also analyzed (better for certain tasks). Generation length is a user-specified hyperparameter. Conditional likelihood evaluation uses a Monte Carlo estimator (Algorithm 3, Eq. 6) with lower variance than the direct ELBO form.

### 3.1 Scalability

LLaDA scales from 10^20 to 10^23 FLOPs and matches ARM baselines trained on the same data across six tasks (MMLU, ARC-C, CMMLU, PIQA, GSM8K, HumanEval). On MMLU and GSM8K, LLaDA shows slightly stronger scaling than the ARM baseline. The authors hypothesize this is due to bidirectional conditioning (multiple conditioning directions) rather than left-to-right only.

### 3.2 Benchmark Results

**Base model (Tab. 1)**: LLaDA 8B Base surpasses LLaMA2 7B Base on nearly all 15 tasks and is competitive with LLaMA3 8B Base overall. Advantages on math (GSM8K: 70.3 vs 48.7) and Chinese (CMMLU: 69.9 vs 50.7 vs LLaMA3). Weaker on PIQA (73.6 vs 80.6) and HumanEval (35.4 vs 34.8—comparable).

**Instruct model (Tab. 2)**: SFT-only LLaDA 8B Instruct lags behind SFT+RL models on some tasks (e.g., MMLU 65.5 vs 68.4 for LLaMA3 8B Instruct) but is competitive given no RL. GSM8K 69.4, Math 31.9.

### 3.3 Reversal Reasoning and Analyses

Using 496 Chinese poem sentence pairs, forward generation = predict next line; reversal = predict preceding line. LLaDA 8B Instruct scores **45.6** on reversal vs 34.3 for GPT-4o and 38.0 for Qwen2.5-7B, while scoring 51.8 on forward (vs 82.7 for GPT-4o). LLaDA treats forward and reversal uniformly because it has no inductive left-to-right bias. Different sampling strategies (autoregressive, block diffusion, pure diffusion) are compared; pure diffusion performs best overall.

### 3.4 Case Studies

Tab. 3 demonstrates LLaDA 8B Instruct's multi-turn dialogue capability (translation across English/Chinese/German, constrained poem generation with acrostic constraint). The sampling visualization shows tokens predicted in later diffusion steps (darker) tend to be content words, while earlier steps (lighter) fill function words—an emergent coarse-to-fine pattern.

### 4 Related Work

Places LLaDA in the landscape of: (a) continuous diffusion for text (Li et al., Gong et al.—limited scalability); (b) discrete diffusion language models (Austin et al. [16], Lou et al. [17], Shi et al. [18], Sahoo et al. [19], Ou et al. [20]—theoretical foundations); (c) MDM scaling laws (Nie et al. [27]—predecessor work by same group at GPT-2 scale); (d) image diffusion transformers (DiT, MaskGIT). LLaDA is the first to scale MDMs to 8B with full pre-training+SFT pipeline.

### 5 Conclusion and Discussion

LLaDA demonstrates that diffusion models can match ARM capabilities at scale. Key open limitations: (1) generation length is a hyperparameter; (2) compute budget capped at 10^23 FLOPs, so ARM baseline not scaled equally large; (3) no KV cache / attention mechanism specialization; (4) no RL alignment; (5) no multimodal capability explored; (6) integration into agent/O1-like systems unexplored.

### Appendix A: Formulation of Masked Diffusion Models

Provides the full mathematical derivation. The forward process factorizes over tokens (Eq. 7–8). The reverse process also factorizes (Eq. 9–10), enabling parallel token prediction. The time-free parameterization (Eq. 11) shows that the mask predictor only needs to condition on unmasked tokens, not on t explicitly—this is the key practical simplification that makes LLaDA implementable as a standard non-causal Transformer.

---

## Result-Tracking List

### Key Equations (treated as formal claims)

- **Eq. (3) — Training objective**: Cross-entropy loss on masked tokens only, averaged over t ~ U[0,1] and masked positions. This is the ELBO surrogate used for all training.
  - *Used in*: Algorithms 1 and 2; upper-bounds negative log-likelihood via Eq. (4).

- **Eq. (4) — ELBO inequality**: −E[log p_θ(x_0)] ≤ L(θ). Proves Eq. (3) is a valid maximum-likelihood objective.
  - *Used in*: Motivation for the training objective; distinguishes LLaDA from MaskGIT which lacks this guarantee.

- **Eq. (5) — SFT loss**: Same as Eq. (3) but masking applied only to response tokens r_0, conditioning on prompt p_0 unmasked.
  - *Used in*: Algorithm 2.

- **Eq. (6) — Conditional likelihood (low-variance form)**: Averages cross-entropy over uniformly sampled mask positions l ∈ {1,...,L} from r_0. Lower variance than Eq. (5) for evaluation.
  - *Used in*: Algorithm 3 (evaluation).

- **Eq. (7–8) — Forward process**: q_{t|0}(x_t|x_0) factorizes per token; each token is masked independently with probability t.
  - *Used in*: Training data generation; reverse process derivation.

- **Eq. (9–10) — Reverse process**: Also fully factorized per token; enables parallel prediction of all masked tokens.
  - *Used in*: Inference (Algorithm 4).

- **Eq. (11) — Time-free parameterization**: q_{0|t}(x^i_t|x_t) = p_data(x^i_0 | x^{UM}_t). The mask predictor only needs unmasked tokens as input, not t.
  - *Used in*: Practical implementation—removes need to input t to the Transformer.

### Algorithms

- **Algorithm 1 — Pre-training**: Sample x_0, sample t ~ U[0,1], mask to get x_t, compute L, backprop. Repeat.
- **Algorithm 2 — SFT**: Same but only response r_0 is masked; prompt p_0 is always visible.
- **Algorithm 3 — Conditional log-likelihood evaluation**: Monte Carlo estimator with n_mc samples; each sample picks a random masking level l and computes cross-entropy.
- **Algorithm 4 — Random remasking (inference)**: Iterates t from 1 to 0 in steps of 1/N; at each step predicts all masks, then re-masks s/t fraction randomly.

---

## Overall Assessment

**Main contributions**:
1. First demonstration that a masked diffusion model can match ARM capabilities when scaled to 8B parameters with full pre-training + SFT.
2. Clean theoretical foundation: training objective is a valid ELBO; time-free parameterization simplifies implementation.
3. Empirical evidence that reversal reasoning is naturally handled by bidirectional models without special training.

**Strengths**:
- Rigorous theoretical grounding (ELBO, time-free parameterization).
- Comprehensive benchmarking across 15+ tasks with transparent protocol.
- Honest about limitations and comparison gaps (no RL, compute asymmetry).
- Open-source code and model weights.

**Limitations / Assumptions**:
- Generation length must be specified in advance (no EOS-like stopping natively).
- No RL/RLHF alignment—instruct model is SFT-only.
- Compute budget comparison is unequal at large scale (ARM baseline not scaled to >10^23 FLOPs).
- Inference is slower than ARMs (no KV cache; multiple forward passes for sampling).
- No multimodal or long-context specialization studied.

**Relevance to this project (memory + distillation)**:
- Directly relevant to `exp0406_distil`: LLaDA is an alternative to ARM-based student models. If distilling to a small model, a diffusion student could better handle bidirectional tasks.
- The time-free parameterization and parallel token prediction are relevant to any experiment where response generation is non-sequential.
- For memory experiments (`exp0404_three_world`): LLaDA's bidirectional nature means it would not suffer from the left-to-right position bias that may affect Qwen-based tested agents in reversal or recall tasks.

---

## Follow-up Notes

- **Open question**: Can LLaDA be used as a tested agent in memory experiments? Its lack of KV cache makes multi-turn conversation expensive, but as a stateless agent (our setting already is stateless per round), this is not a bottleneck.
- **Idea to reuse**: The low-confidence remasking strategy (Appendix B.4) is analogous to the memory stabilization logic in `memory_io.py`—both try to avoid overwriting high-confidence content.
- **To read next**: Nie et al. [27] (arXiv:2410.18514) — scaling laws for MDMs, predecessor to LLaDA. Ou et al. [20] (arXiv:2406.03736) — theoretical foundations cited heavily here.
