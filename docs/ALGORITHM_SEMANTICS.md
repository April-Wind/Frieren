# Algorithm semantics

Frieren exposes one artifact interface for three alternative relation estimators. Every graph uses **receiver rows and sender columns**: propagation is `H_next = A @ H`. The estimators have different graph-conversion recipes. Sharing an interface does not make their normalization, self-edge policy, or tie handling identical.

## Frieren-K

Knowledge estimation starts from externally pretrained knowledge embeddings. `align_entity_embeddings` extracts recommendation items in the catalogue's exact order; non-item entities participate upstream but do not become downstream graph nodes. The package does not substitute repaired modality-ablation vectors for the canonical strong-graph vectors.

Item vectors are L2-normalized in float32, cosine neighbours are selected with `torch.topk`, and each selected edge receives `1/k`. **The canonical knowledge recipe includes self edges**. Set `include_self=False` only for the explicitly named self-removal ablation. Top-k is row-wise; the output is generally directed and is not automatically symmetrized.

Equal-score selection can depend on the numerical library/device. Published frozen graph assets with their hashes are the exact reference when reproducing a particular historical experiment. Regenerating from embeddings is a separately auditable construction step.

## Frieren-B

For binary training interactions $X$, EASE solves

$$
\min_{\operatorname{diag}(B)=0}\|X-XB\|_F^2+\lambda\|B\|_F^2.
$$

With $P=(X^\top X+\lambda I)^{-1}$, its solution is

$$
B_{ij}=-P_{ij}/P_{jj},\qquad B_{jj}=0.
$$

`coefficients_[source, target]` follows EASE's prediction convention. `predict` returns $XB$, using the complete signed matrix. Direct EASE prediction does not discard negative coefficients.

The frozen graph uses `B.T[receiver, sender]`. Its **clean-v2** recipe is:

1. Remove self and retain positive incoming coefficients.
2. Select up to `k` in descending coefficient order, breaking exact ties by ascending internal item index.
3. If a row has fewer than `k` candidates, supplement it from the explicitly supplied multimodal fallback graph, in descending existing edge weight then ascending item index. Exclude self and already selected neighbours.
4. Normalize all selected positive edge weights by their row sum.

No KG fallback is silently introduced. A limited fallback pool may leave fewer than `k` neighbours; the implementation records this instead of inventing neighbours. Historical legacy-EASE graphs used different fallback/completion rules and must not be relabelled clean-v2.

The closed-form solve uses float64 and a Cholesky factorization. Building a full coefficient matrix has quadratic storage and a dense factorization cost. The graph alone does not carry that upstream cost away.

## Frieren-L

The default estimator reproduces U2: 64-dimensional user/item ID embeddings initialized with normal standard deviation 0.08; a fixed-feature encoder with a 256-dimensional hidden layer, LeakyReLU slope 0.2, and a 64-dimensional output. Encoder parameters are trainable; raw features are frozen. Image and text features are each L2-normalized, concatenated, and divided by $\sqrt{2}$ before entering the encoder.

Let $a_u,v_i,z_i$ be normalized user, item-ID, and encoded-content vectors. Warm items use

$$
e_i=\operatorname{normalize}(v_i+z_i),
$$

while items with no training interactions use $e_i=z_i$. This same rule is used for direct prediction and graph export.

### Training tasks

Each epoch contains `X.nnz` tasks, not one pass through observed pairs. Users are uniformly sampled with replacement; a positive is sampled uniformly from each chosen user's training history. The default task includes 128 negatives sampled from training-unseen items with replacement. Negatives can repeat. Validation/test interactions are not used to remove negatives. The original 160-item draw blocks and random-stream construction are retained.

With probability 0.5, an entire task's candidate block uses content vectors alone. Otherwise it uses the fused vectors. The positive and all negatives share this mask. Candidate scores are $a_u^\top e_i/\tau$, with $\tau=0.2$, and recommendation loss is candidate cross-entropy.

Unique positive items within the batch additionally receive bidirectional content–behavior alignment:

$$
\mathcal L_{\mathrm{align}}=
\tfrac12\big(\operatorname{CE}(ZV^\top/\tau,I)+
\operatorname{CE}(VZ^\top/\tau,I)\big).
$$

The total objective is

$$
\mathcal L=\mathcal L_{\mathrm{rec}}+0.1\mathcal L_{\mathrm{align}}
+10^{-4}\mathcal L_{\mathrm{reg}},
$$

where regularization sums mean squared raw user embeddings, candidate item embeddings, and each encoder parameter tensor. It is not an optimizer weight-decay substitute.

Adam uses learning rate 0.001, batch size 256, and at most 200 epochs. Side-only and direct validation Recall@20 are tracked separately; training stops after both have been stale for 20 epochs. **Only the direct-validation-selected checkpoint is eligible for canonical graph export.** Test evaluation occurs once after selection. The side checkpoint is retained for diagnostics, not selected using test performance.

### Export

The selected model exports warm fused / cold content vectors. Per-row float32 inner products are converted to float64 for the archived NumPy `argpartition` selection rule; self scores are set to negative infinity. Top-10 neighbours receive equal weight 0.1. They are not weighted by cosine score. K and L deliberately preserve their original, different tie-selection rules.

Resume restores model, Adam state, epoch, best/stale bookkeeping, and random states. It checks configuration, item order, features, training interactions, and split identities. Resume is exact at the saved epoch boundary; a partially completed epoch is replayed. PyTorch checkpoints containing optimizer/RNG objects are loaded as trusted local artifacts; do not load arbitrary untrusted checkpoint files.

## Optional BGNW

BGNW refines **weights on existing Frieren-L neighbours**; it does not select new neighbours. For receiver $i$, let $Z_i=X[:,\mathcal N(i)]$, $c_i=Z_i^\top X[:,i]$, and $d_i=\sum_uX_{ui}$. The diagonal objective is

$$
\min_w\ \tfrac12w^\top
\left(\frac{\operatorname{diag}(\operatorname{diag}(Z_i^\top Z_i))}{d_i}
+\alpha I\right)w-\frac{c_i^\top w}{d_i}.
$$

Weights sum to one. Warm-source weights are at least $10^{-5}$; cold-source weights retain their original values. Cold receivers, rows with fewer than two warm neighbours, and rows without target–neighbour cooccurrence are left unchanged. The default is $\alpha=1$. The active-set solver checks KKT residuals; graph support remains exactly fixed.

This package's optional BGNW is the historical DIAG method, not JOINT. JOINT uses the full neighbourhood Gram matrix; it is not silently substituted here. Sports' ordinary-U2 choice, if used in a paper configuration, is an explicit dataset setting and not an automatic quality detector.

## Downstream adapters

FREEDOM consumes the frozen propagation operator directly, with no repeated normalization. The item branch starts from the downstream model's own item ID embeddings, propagates on the fixed graph, and combines its result with user–item collaborative propagation.

The paper's MENTOR, SMORE, and MSCA transfers first convert a source graph to

$$
\widetilde A=D^{-1/2}(A+A^\top)D^{-1/2},\qquad
D_{ii}=\sum_j(A+A^\top)_{ij}.
$$

Their replacement views are respectively `mm_adj`, `fusion_adj`, and `struct_original_adj`. Other views are retained. Consequently a transferred operator is an adapter-transformed source graph, not a byte-identical FREEDOM operator. Record both source and transformed signatures when reporting transfer results.

## Research-source provenance

The clean implementation was extracted from the following frozen research snapshots. These are provenance references, not runtime dependencies on an unpublished workspace:

- `paper_complete_20261010/main_stage/build_graphs.py`: clean-v2 EASE, K/L sensitivity construction.
- `paper_complete_20261010/est_stage/train_estimator.py`: U2 initialization, masking, losses, sampler streams, selection, direct export.
- `si29_fused_repair_20260930/collected/export_graph.py`: corrected fused-direct export.
- `u2_conditional_weights_20261006/prepare.py`: diagonal neighborhood objective, cold protections, simplex solver.
- `paper_complete_20261010/repair_stage/payload/sources/{mentor,smore,msca}/src/models/behavior_graph.py`: transfer normalization and view replacement.
- `paper_complete_20261010/main_stage/payload/src/models/freedom.py`: direct graph consumption and graph-only M0 downstream configuration.

Tiny CPU parity tests compare the archived implementations when run in the research workspace. Portable tests also independently verify the constrained EASE solution, graph orientation, mapping/self policies, BGNW constrained objective, cold protections, and exact resume behavior. These checks establish implementation fidelity; they do not replace full dataset experiments or prove a universal performance gain.
