# Package organization

The package separates relation estimation, graph storage, and graph consumption.

1. **Estimation:** Frieren-K reads knowledge embeddings, Frieren-B solves behavioral reconstruction, and Frieren-L learns joint content–behavior representations. Optional BNW refines weights within Frieren-L's selected neighborhoods.
2. **Artifacts:** `GraphArtifact` binds an operator to item identifiers, construction settings, and signatures. Different estimators retain their self-loop, weighting, and tie-selection rules.
3. **Consumption:** FREEDOM uses the operator directly. Transfer adapters perform explicit view replacement for MENTOR, SMORE, and MSCA.

Variants are expressed through configurations rather than duplicated model files. The core package has no dependency on remote instances or experiment directories. Historical-source parity tests are optional; standalone tests cover the mathematical objectives, mapping, adapters, resume, and replay.

Exact benchmark reproduction requires the selected data splits, input features, pretrained embeddings, graph settings, and model configurations. Graph hashes identify a frozen operator when top-k ties depend on the numerical backend.
