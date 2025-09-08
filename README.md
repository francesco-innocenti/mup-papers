# 📑 μP Papers

![status](https://img.shields.io/badge/status-active-green)

A curated repository of papers on the maximal update parameterisation (μP) and 
related ideas.

> μP is an influential, theoretically grounded prescription for how to scale 
various neural network architectures such that the layer activations (and other 
quantities such as the learning rate) remain stable during training (neither 
shrink nor explode) with the model size (i.e. width and depth).


## Key original papers (width-only μP)
* [Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522)
* [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
* [Tensor Programs IVb: Adaptive Optimization in the Infinite-Width Limit](https://arxiv.org/abs/2308.01814)
* [A Spectral Condition for Feature Learning](https://arxiv.org/abs/2310.17813)


## Depth extensions (depth-μP)
* [Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522)
* [Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit](https://arxiv.org/abs/2309.16620)
* [Super Consistency of Neural Network Landscapes and Learning Rate Transfer](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ba1d33849b963efc6b5d3082ad68f480-Abstract-Conference.html)
* [Don’t be lazy: CompleteP enables compute-efficient deep transformers](https://arxiv.org/abs/2505.01618)


## Other extensions
* [The Optimization Landscape of SGD Across the Feature Learning Strength](https://arxiv.org/abs/2410.04642)
* [On the Parameterization of Second-Order Optimization Effective Towards the Infinite Width](https://arxiv.org/abs/2312.12226)
* [How to set AdamW's weight decay as you scale model and dataset size](https://arxiv.org/abs/2405.13698)
* [Scaling Diffusion Transformers Efficiently via μP](https://arxiv.org/abs/2505.15270)
* [Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size](https://arxiv.org/abs/2506.15025)
* [Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](https://arxiv.org/abs/2505.13738)
* [Sparse maximal update parameterization: A holistic approach to sparse training dynamics](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3b6aaffec941f98930753fa6d6de7263-Abstract-Conference.html)
* [μnit Scaling: Simple and Scalable FP8 LLM Training](https://arxiv.org/abs/2502.05967)
* [Local Loss Optimization in the Infinite Width: Stable Parameterization of Predictive Coding Networks and Target Propagation](https://arxiv.org/abs/2411.02001)
* [μPC: Scaling Predictive Coding to 100+ Layer Networks](https://arxiv.org/abs/2505.13124)


## Miscellaneous
* [Feature-Learning Networks Are Consistent Across Widths At Realistic Scales](https://proceedings.neurips.cc/paper_files/paper/2023/hash/03600ae6c3392fd65ad7c3a90c6f7ce8-Abstract-Conference.html)
* [On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling](https://arxiv.org/abs/2505.22491)
* [A thorough reproduction and evaluation of μP](https://openreview.net/forum?id=AFxEdJwQcp)
* [Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks](https://arxiv.org/abs/2507.02119)


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
* the [`nanoGPT-mup`](https://github.com/EleutherAI/nanoGPT-mup?tab=readme-ov-file) 
github repo (PyTorch).


## Contributing
Contributions are welcome! To add a paper or submit a correction, please open an 
issue or submit a pull request.