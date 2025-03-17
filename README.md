# biogen

(For educational purposes) generative model for bioRxiv titles.

**Roadmap**

- [x] Implement attention https://arxiv.org/abs/1706.03762
- [x] Load data from https://huggingface.co/datasets/laion/biorXiv_metadata
- [x] tokenize data
- [ ] Pytorch data loader
- [ ] Token embedder and position embedder
- [ ] Overfit on one batch with a small model
- [ ] Switch to FlashAttention
- [ ] Train v0 model on CPU
- [ ] Scale up model and train v1 on Colab GPU
- [ ] Scale up model and train v2 on multiple GPUs (TBD where, but likely Kaggle or Lambda)
- [ ] Embed all the titles and visualize UMAP
