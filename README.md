# 📑 μP Papers

![status](https://img.shields.io/badge/status-active-green)

μP is an influential, theoretically grounded prescription for how to scale 
various neural network architectures such that the layer activations (and other 
quantities such as the learning rate) remain stable during training (neither 
shrink nor explode) with the model size (i.e. width and depth).


## Overview
* [Key papers](#key-papers-width-only-μp)
* [Depth extensions](#depth-extensions)
* [Understanding hyperparameter transfer](#understanding-hyperparameter-transfer)
* [Other optimisers](#other-optimisers)
* [Other architectures](#other-architectures)
* [On weight decay](#on-weight-decay)
* [Miscellaneous](#miscellaneous)
* [Further resources](#further-resources)


## Key papers (width-only μP)
* [Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522): original paper introducing μP for SGD building on the "Tensor Programs" formalism. The main motivation was to find a parameterisation that both (i) allows for as much feature learning as possible (μP is maximal in this sense) unlike the NTK, and (ii) remains stable with respect to the model width, unlike the standard parameterisation.
* [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466): building on the previous paper, this shows that under μP, many optimal hyperparameters such as the learning rate also remain stable across models (including GPT-3) of different width, allowing for zero-shot hyperparameter transfer without tuning at large scale. It also extends μP for Adam beyond SGD.
* [Tensor Programs IVb: Adaptive Optimization in the Infinite-Width Limit](https://arxiv.org/abs/2308.01814): fully works out the μP theory for adaptive optimisers including Adam.
* [A Spectral Condition for Feature Learning](https://arxiv.org/abs/2310.17813): shows an interesting equivalence between μP and a certain scaling of the spectral norm of weight matrices and their updates. This partly inspired the [Muon optimiser](https://jeremybernste.in/writing/deriving-muon).


## Depth extensions
* [Depth Dependence of μP Learning Rates in ReLU MLPs](https://arxiv.org/abs/2305.07810)
* [Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks](https://arxiv.org/abs/2310.02244): concurrently with the following paper, this proposed an extension of μP to model depth for ResNets (with unit block depth) by rescaling each residual block and parameter update by the square root of the depth. Experiments with fully connected ResNets on CIFAR10.
* [Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit](https://arxiv.org/abs/2309.16620): concurrently with the previous paper, this proposed a slightly different depth-extension of μP using dynamical mean field theory (DMFT). Unlike Yang et al. (2024), they do not rescale the learning rate of Adam, but this rescaling is reintroduced in the next paper. Experiments with both CNNs and ViTs (with and without LayerNorm) on both CIFAR10 and ImageNet.
* [Infinite Limits of Multi-head Transformer Dynamics](https://arxiv.org/abs/2405.15712): also relying on DMFT, derives width and depth limits for multi-head attention transformers, providing principled scalings for SGD and heuristic scalings for Adam.
* [Don’t be lazy: CompleteP enables compute-efficient deep transformers](https://arxiv.org/abs/2505.01618): in contrast to previous depth extensions of μP, this proposes rescaling the residual transformer blocks by the depth (rather than its square root) based on both empirical results and a theoretical notion of non-lazy learning of all model layers. This parameterisation requires rescaling other quantities such as LayerNorm and Adam's weight decay parameter. Experiments with large-scale LLMs, also revealing new compute-optimal regimes.


## Understanding hyperparameter transfer
* [Super Consistency of Neural Network Landscapes and Learning Rate Transfer](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ba1d33849b963efc6b5d3082ad68f480-Abstract-Conference.html): investigates the phenomenon of learning rate transfer from an optimisation perspective, showing that under μP and its depth extension (but not the NTK), certain quantities including the largest eigenvalue of the loss Hessian (aka sharpness) remain consistent across different model scales (i.e. widths and depths). Comprehensive experiments with ResNets, ViTs and GPT-2.
* [On the Provable Separation of Scales in Maximal Update Parameterization](https://openreview.net/forum?id=csB1njlpjM)
* [A Proof of Learning Rate Transfer under μP](https://arxiv.org/abs/2511.01734)
* [Understanding the Mechanisms of Fast Hyperparameter Transfer](https://openreview.net/forum?id=Q7mLKxQ8qk)


## Other optimisers
* [On the Parameterization of Second-Order Optimization Effective Towards the Infinite Width](https://arxiv.org/abs/2312.12226): derives a feature-learning (μP-like) infinite-width limit parameterisation for second-order methods including K-FAC and Shampoo. Experiments with MLPs, CNNs, ResNets and a simplified language model.
* [Effective Sharpness Aware Minimization Requires Layerwise Perturbation Scaling](https://openreview.net/forum?id=Qo6KUhQkPw): derives a μP extension for sharpness aware minimisation (SAM) with stable learning rate and perturbation radius across model widths. Experiments with MLPs, ResNets & ViTs.
* [Extending μP: Spectral Conditions for Feature Learning Across Optimizers](https://openreview.net/forum?id=TfJ67nPJl2)
* [Towards a Principled Muon under μ𝖯: Ensuring Spectral Conditions throughout Training](https://arxiv.org/abs/2601.01306)
* [Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning](https://www.arxiv.org/abs/2602.06204)
* [Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales](https://arxiv.org/abs/2512.05620)


## Other architectures
* [u-μP: The Unit-Scaled Maximal Update Parametrization](https://arxiv.org/abs/2407.17465): proposes a simple extension of μP for LLMs such that activations, weights and gradients have unit (rather than just constant) variance (u-μP) with respect to model width, showing that it helps with low-precision training.
* [μnit Scaling: Simple and Scalable FP8 LLM Training](https://arxiv.org/abs/2502.05967): proposes a more efficient unit-scaled μP parameterisation for low-precision LLM training.
* [Sparse maximal update parameterization: A holistic approach to sparse training dynamics](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3b6aaffec941f98930753fa6d6de7263-Abstract-Conference.html): derives a μP extension for random unstructured static (weight) sparsity with stable feature learning across both model widths and sparsity level. Experiments with LLMs.
* [Scaling Diffusion Transformers Efficiently via μP](https://arxiv.org/abs/2505.15270)
* [Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size](https://arxiv.org/abs/2506.15025)
* [On Feature Learning in Structured State Space Models](https://openreview.net/forum?id=aQv5AbN1wF)
* [Local Loss Optimization in the Infinite Width: Stable Parameterization of Predictive Coding Networks and Target Propagation](https://arxiv.org/abs/2411.02001)
* [μPC: Scaling Predictive Coding to 100+ Layer Networks](https://arxiv.org/abs/2505.13124)
* [On the Infinite Width and Depth Limits of Predictive Coding Networks](https://arxiv.org/abs/2602.07697)
* [μ-Parametrization for Mixture of Experts](https://arxiv.org/abs/2508.09752)
* [Transfer Paramatters: Optimal per-Module Hyperparameters Across All Scaling Axes](https://openreview.net/forum?id=elB9k4nTL1)
* [GQA-μP: The Maximal Parameterization Update for Grouped Query Attention and Fully Sharded Data Parallel](https://openreview.net/forum?id=UJB2uOS9MR)
* [μLO: Compute-Efficient Meta-Generalization of Learned Optimizers](https://openreview.net/forum?id=f8z2bzOLK2)
* [Arithmetic-Mean μP for Modern Architectures: A Unified Learning-Rate Scale for CNNs and ResNets](https://arxiv.org/abs/2510.04327)
* [Hyperparameter Transfer with Mixture-of-Experts Layers](https://arxiv.org/abs/2601.20205)
* [Hyperparameter Transfer Laws for Non-Recurrent Multi-Path Neural Networks](https://www.arxiv.org/abs/2602.07494)
* [μpscaling small models: Principled warm starts and hyperparameter transfer](https://arxiv.org/abs/2602.10545)


## On weight decay
The role of weight decay with respect to depth-transfer is discussed in the 
[CompleteP work](https://arxiv.org/abs/2505.01618) (Dey et al., 2025).
* [How to set AdamW's weight decay as you scale model and dataset size](https://arxiv.org/abs/2405.13698)
* [Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](https://arxiv.org/abs/2505.13738)
* [Weight Decay may matter more than muP for Learning Rate Transfer in Practice](https://arxiv.org/abs/2510.19093)
* [Robust Layerwise Scaling Rules by Proper Weight Decay Tuning](https://arxiv.org/abs/2510.15262)


## Miscellaneous
* [Lecture Notes on Infinite-Width Limits of Neural Networks](https://mlschool.princeton.edu/sites/g/files/toruqf5946/files/documents/Princeton___Lecture_Notes_0.pdf): these notes (see especially Section 4) provide a detailed and pedagogical derivation of (width-only) μP for MLPs.
* [Feature-Learning Networks Are Consistent Across Widths At Realistic Scales](https://proceedings.neurips.cc/paper_files/paper/2023/hash/03600ae6c3392fd65ad7c3a90c6f7ce8-Abstract-Conference.html): empirically shows that the behaviour of finite-width networks is remarkably consistent across model widths used in practice, validating μP. See the next paper for a similar study also considering depth.
* [Function-Space Learning Rates](https://arxiv.org/abs/2502.17405): impressive paper developing an efficient method (requiring only a few extra backward passes) to measure the change in the network function induced by parameter updates to achieve hyperparameter transfer across width, depth and even LoRA rank for many architectures including transformers. The empirical nature of this approach has the advantage, over μP, of not needing to derive scalings on a case-by-case basis.
* [The Optimization Landscape of SGD Across the Feature Learning Strength](https://arxiv.org/abs/2410.04642)
* [Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization](https://arxiv.org/abs/2602.00827)
* [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872)
* [Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks](https://arxiv.org/abs/2507.02119)
* [On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling](https://arxiv.org/abs/2505.22491)
* [A thorough reproduction and evaluation of μP](https://openreview.net/forum?id=AFxEdJwQcp)
* [The lazy (NTK) & rich (μP) regimes: A gentle tutorial](https://arxiv.org/abs/2404.19719)
* [An Empirical Study of μP Learning Rate Transfer](https://arxiv.org/abs/2404.05728)
* [μP for RL: Mitigating Feature Inconsistencies During Reinforcement Learning](https://openreview.net/forum?id=Wuy631kHwH)
* [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://arxiv.org/abs/2404.06395)


## Further resources

### 📝 Blogs
* my high-level [blog post](https://francesco-innocenti.github.io/posts/2025/04/09/Infinite-Widths-&-Depths-Part-III-The-Maximal-Update-Parameterisation/) on μP and its extensions
* Microsoft's [post](https://www.microsoft.com/en-us/research/blog/on-infinitely-wide-neural-networks-that-exhibit-feature-learning/) 
introducing μP
* Microsoft's [post on the original hyperparameter transfer results](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/)
* [this post](https://blog.speechmatics.com/mup) by Speechmatics
* [this post](https://cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization) 
by Cerebras

### 🎙️ Talks
* [this conversation](https://www.youtube.com/watch?v=1aXOXHA7Jcw&t=2723s&ab_channel=TimothyNguyen) 
with Greg Yang focused on "Tensor Programs"
* [this talk](https://www.youtube.com/watch?v=CnAfD7aVzLg&ab_channel=AutoMLSeminars) on the scaling exponents of different parameterisations

### 💻 Code
* the original [`mup`](https://github.com/microsoft/mup?tab=readme-ov-file#coord-check) github repo (PyTorch)
* the [`nanoGPT-mup`](https://github.com/EleutherAI/nanoGPT-mup?tab=readme-ov-file) repo (PyTorch).


## Contributing
Contributions are welcome! To add a paper or submit a correction, please open an 
issue or submit a pull request.