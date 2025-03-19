# biogen

(For educational purposes) generative model for bioRxiv titles.

**Roadmap**

- [x] Implement attention https://arxiv.org/abs/1706.03762
- [x] Load data from https://huggingface.co/datasets/laion/biorXiv_metadata
- [x] tokenize data
- [x] Pytorch data loader
- [x] Token embedder and position embedder
- [x] generate mode
- [x] Implement train step and eval step (Overfit on one batch with a small model)
- [x] Train small v0 model on CPU
- [x] Scale up model and train v1 on Colab GPU
