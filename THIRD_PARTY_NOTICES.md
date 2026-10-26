# Attribution and distribution scope

Frieren builds on the following methods:

- **EASE** — *Embarrassingly Shallow Autoencoders for Sparse Data*. The package implements its closed-form mathematical objective.
- **NativE** — supplies pretrained multimodal knowledge embeddings to Frieren-K. Its trainer, datasets, embeddings, and checkpoints are not distributed here.
- **FREEDOM** — *A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation*. The package contains the single-graph M0 consumer used by Frieren, including degree-dependent interaction-edge sampling and user–item propagation.
- **MENTOR, SMORE, and MSCA** — their model implementations are not copied here. Integration adapters replace only the specified item-graph views. Use the original implementations under their respective licenses.

This repository contains framework source, configurations, documentation, tests, and synthetic-data utilities. It excludes benchmark data, pretrained assets, credentials, remote administration scripts, and experiment logs.

No additional license grant is made by this notice. Applicable third-party rights remain with their respective holders.
