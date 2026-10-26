# Data and graph artifacts

Each dataset has a JSON manifest, as illustrated in `configs/dataset.example.json`. Paths are resolved relative to that manifest.

The interaction file is TSV with `userID`, `itemID`, and `x_label` columns. Users and items have contiguous zero-based integer indices. Labels 0, 1, and 2 denote training, validation, and test. Duplicate user–item pairs, including overlaps between splits, are rejected. The loader preserves the supplied splits.

The item mapping is TSV with columns `item_id` (external identifier) and `itemID` (internal index). It covers the full catalog, including items without training interactions. Features, graph nodes, and checkpoints share this order.

Image and text features are two-dimensional numeric NumPy arrays without Python objects. Row `i` belongs to `itemID=i`. Frieren-L normalizes each modality, concatenates the features, and divides by the square root of the number of modalities. Raw features are frozen; the content encoder is trainable.

Knowledge embeddings may already follow item order. Otherwise, provide entity embeddings, a JSON list of entity identifiers in embedding-row order, and a JSON dictionary mapping external item identifiers to entity identifiers. Frieren-K extracts and reorders the items explicitly.

A graph consists of two files:

- `graph.pt`: a CPU sparse COO tensor.
- `graph.pt.json`: item order, construction settings, tensor SHA256, and provenance metadata.

`A[receiver, sender]` weights the message received by the row item from the column item. Loading checks dimensions, item order, finite nonnegative weights, and hashes. Adapted graphs record their target model and transformation and cannot be adapted twice.

Training checkpoints contain optimizer and random states and must come from a trusted source. Graph tensors are loaded with PyTorch's `weights_only=True`.
