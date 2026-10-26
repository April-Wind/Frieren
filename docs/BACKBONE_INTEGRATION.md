# Frozen-graph consumers and transfer interfaces

Frieren estimates an item operator before downstream recommendation training.
Its row `i`, column `j` entry is the weight of the message **received by item
`i` from item `j`**. Item identifiers and their order are carried with every
graph artifact. A graph is not automatically treated as an undirected graph.

## Evaluated model interfaces

| Model | Replaced view | Conversion before insertion |
| --- | --- | --- |
| FREEDOM M0 | `mm_adj` | Identity: use the saved operator directly |
| MENTOR | `mm_adj` | Symmetrize with `A + A.T`, then symmetric degree normalization |
| MSCA | `struct_original_adj` | The same explicit transfer conversion |
| SMORE | `fusion_adj` | The same explicit transfer conversion |

For the three transferred models, the conversion is

$$
\widetilde A=A+A^\top,\qquad
D_{ii}=\max\left(\sum_j\widetilde A_{ij},10^{-12}\right),\qquad
A_{\mathrm{transfer}}=D^{-1/2}\widetilde A D^{-1/2}.
$$

The transferred operator can have a larger neighbor set than the original
directed operator. This is the **evaluated adapter**, not a claim that these
models consume a bit-identical FREEDOM operator. No second Top-K selection,
loop removal, or weight learning is performed by the adapter. In particular,
canonical Frieren-K's self-including neighbor policy is retained.

## FREEDOM: an executable clean consumer

`FreedomConsumer` implements the graph-only M0 setting used in the final
FREEDOM experiments. Learnable parameters consist of user and item ID
embeddings initialized with Xavier uniform. The UI branch averages its input
embedding and all propagation layers; the item branch uses its last layer:

$$
E_{\mathrm{UI}}=\frac{1}{L_{\mathrm{UI}}+1}
\sum_{\ell=0}^{L_{\mathrm{UI}}}\widehat A_{\mathrm{UI}}^\ell E^{(0)},
\qquad
H_{\mathrm{item}}=A^{L_{\mathrm{item}}}Q.
$$

Final item vectors add the UI item output and the item-graph output. A BPR
loss trains these vectors from training positives and one sampled negative.
Degree-dependent UI edge dropout is sampled once per epoch. Its independent
generator uses `seed + 1000003`; changing the item graph does not change that
edge stream. No auxiliary feature BPR, graph gate, content-message experiment,
or new loss is added in the clean implementation.

```python
from frieren.training import TrainConfig, train_freedom

result = train_freedom(
    data, graph,
    TrainConfig(epochs=500, seed=999),
    output_dir="outputs/freedom_frieren_l",
    device="cuda",
)
```

The trainer shuffles training interaction rows and draws negatives from
training-active items, excluding each user's training history. This matches
the source consumer's negative pool; it differs from the estimator's
full-catalog candidate sampling. The default learning rate is constant Adam
at 0.001. The saved `last.pt` includes optimizer state, training row order,
the item sampling pool, Python/NumPy/Torch RNG states, and UI dropout RNG.
`best.pt` is selected only by validation Recall@20, using strict improvement.
Test data never selects a checkpoint. The clean trainer evaluates test once
after restoring the selected checkpoint; historical per-epoch test logging
is not required to choose the model.

An explicit `resume=True` accepts a larger fixed epoch budget with all other
settings, inputs, device, and library versions unchanged. It resumes the last
training state, not the selected model. A process lock prevents overlapping
writers to the same output directory. Files are written atomically.

Forward values, BPR gradients, UI dropout streams, and the training-row and
negative-sampling streams have CPU fixture parity tests against the frozen
experimental M0 source when that source is present in the development
workspace. These historical-source comparisons are explicitly skipped in a
standalone checkout that does not include the private experiment snapshots.
The standalone mathematical, adapter, resume, and replay tests still run.
Full historical GPU trajectories still need
benchmark-data verification before asserting result-level reproducibility.
The release does not claim backend-independent bitwise training equivalence.

## MENTOR, SMORE, and MSCA: integration hooks

The clean package provides graph adapters for these upstream implementations;
it does not substitute a simplified approximation of their architectures.
Obtain the corresponding upstream code and dependencies, then insert the
hook **after its native item graph is constructed and before training**:

```python
from frieren.adapters import apply_graph
from frieren.contracts import GraphArtifact

artifact = GraphArtifact.load("artifacts/frieren_l.pt")
applied = apply_graph(model, artifact, backbone="MSCA")
applied.save("outputs/msca/replacement_operator.pt")
```

The evaluated MSCA replacement targets its behavior co-occurrence structural
view, leaving image and text views intact. The SMORE replacement targets the
fusion view, leaving its modality-specific views intact. MENTOR replaces its
multimodal item view, retaining its original interaction and alignment
mechanisms. Other views are not automatically replaced or added.

`apply_graph` validates item dimensions and the named attribute and clears
common stored prediction embeddings. If the upstream model exposes
`item_ids`, it checks their complete order. Otherwise the caller must ensure
the model uses the exact mapping supplied with the artifact: matching only
the item count is insufficient to prove alignment. The saved adapted artifact
records the source operator signature, output signature, backbone, and view.
An already adapted artifact cannot be adapted a second time.

For native controls, construct the model normally and omit the hook. Use
the same model configuration, split, seed, training budget, and model
selection policy for a paired native-versus-replaced comparison. Published
transfer results may use different validated configurations across models;
the graph adapter does not overwrite or retune those configurations.

## Fixed-parameter graph evaluation

```python
from frieren.training import replay_freedom

comparison = replay_freedom(
    data,
    {"native": native_graph, "Frieren-K": kg_graph,
     "Frieren-B": behavior_graph, "Frieren-L": learned_graph},
    checkpoint_path="outputs/freedom_frieren_l/best.pt",
    device="cuda",
)
```

The routine freezes the checkpoint parameters and exchanges the item operator
without an optimizer step. Repeat it for checkpoints trained with each graph
to obtain a graph-by-parameter matrix. Both validation and test metrics are
reported. For selecting diagnostic examples, use validation results first;
test results describe the corresponding fixed cases. The routine accepts
release-framework checkpoints; historical consumer checkpoints require an
explicit parameter-name conversion and separate native-cell parity check.

This analysis separates the observed effect of changing the graph under
fixed parameters from the diagonal comparison of independently trained
systems. It does not identify a causal mechanism for the optimizer's behavior
or prove that a graph is superior under every possible parameter setting.
