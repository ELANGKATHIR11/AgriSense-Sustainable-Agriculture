# Patent Novelty Claim #04: Visual Retrieval-Augmented Generation (VRAG) for Agricultural Explanations

## Title
System and Method for Vision-Embedding-Based Retrieval-Augmented Generation Constraining Large Language Model Outputs to Evidence-Grounded Agricultural Explanations

## Mechanism

VRAG (Visual Retrieval-Augmented Generation) extends traditional text-based RAG by operating on vision embeddings rather than text embeddings:

1. **Index Construction**: DeiT-extracted 384-dim embeddings from thousands of validated crop disease images are indexed in a FAISS inner-product index with associated metadata (crop type, disease name, severity, treatment outcome, geographic region).

2. **Visual Retrieval**: When a new crop image is analyzed, its DeiT embedding is used to query the FAISS index, retrieving the K most similar historical cases with similarity scores and full metadata.

3. **Evidence Assembly**: Retrieved cases are formatted as structured evidence documents containing:
   - Similarity score (quantified visual similarity)
   - Disease identification from the matched case
   - Treatment that was applied and its outcome
   - Environmental conditions from the matched case

4. **LLM Constraint**: The assembled evidence is passed to Phi-3-mini with a strict system prompt that enforces:
   - The LLM can ONLY explain the decision using provided evidence
   - The LLM NEVER diagnoses diseases (the vision pipeline did that)
   - The LLM NEVER prescribes treatments (the action templates provide those)
   - The LLM NEVER overrides or contradicts the Governor's decision

5. **Tone Modulation**: The LLM's language style is automatically adjusted based on the Governor's confidence level — HIGH confidence produces direct, actionable language; LOW confidence produces cautious, hedging language.

## Why Non-Obvious

1. **Text RAG vs. Visual RAG**: Existing RAG systems retrieve text documents based on text queries. VRAG retrieves visual examples based on image similarity. This is non-obvious because:
   - The embedding space is learned for classification, not retrieval — adapting it for retrieval requires L2 normalization and inner-product indexing
   - The metadata associated with visual examples provides different information than text documents — treatment outcomes, severity progressions, geographic specificity

2. **Evidence constrains generation, not informs it**: In traditional RAG, retrieved documents inform the LLM's response. In VRAG, retrieved evidence CONSTRAINS the LLM — it cannot go beyond what the evidence supports. This is non-obvious because it reverses the typical RAG paradigm from "assistive context" to "strict grounding."

3. **Confidence-aware tone**: Adjusting the linguistic register based on decision confidence is non-obvious because:
   - Most LLM systems use a fixed tone regardless of certainty
   - Agricultural users need to distinguish confident recommendations from uncertain observations
   - The tone modulation is driven by the Decision Governor's confidence band, not the LLM's own uncertainty

4. **Visual similarity as semantic evidence**: Using image similarity scores as evidence for natural-language explanations bridges the modality gap — a visual match of 0.95 is translated into "this case is very similar to a confirmed case of late blight in tomatoes," providing evidence a farmer can understand without seeing the retrieved image.

## System Claim

A computer-implemented system for vision-embedding-based retrieval-augmented generation comprising:
- A FAISS inner-product index storing L2-normalized DeiT vision embeddings with associated agricultural metadata
- A retrieval engine that queries the index with a new image's embedding and returns K most similar cases with similarity scores
- An evidence assembly module that formats retrieved cases into structured evidence documents
- A constrained LLM (Phi-3-mini) with a strict system prompt preventing autonomous diagnosis, prescription, or decision override
- A tone modulation layer that adjusts linguistic register based on the Decision Governor's confidence band

## Method Claim

A method for evidence-grounded agricultural explanation generation comprising:
1. Extracting a DeiT vision embedding from an input crop image
2. Querying a FAISS inner-product index of validated disease embeddings to retrieve K most similar historical cases
3. Assembling structured evidence documents from retrieved cases including similarity scores, disease identifications, treatment outcomes, and environmental conditions
4. Passing the evidence documents to a constrained LLM with a system prompt that enforces strict evidence grounding
5. Modulating the LLM's linguistic tone based on the Decision Governor's confidence band (HIGH → direct; LOW → cautious)
6. Generating a structured explanation containing summary, evidence used, confidence level, next steps, and explicit warnings

## Dependent Claims

1. The system of the main claim wherein the FAISS index supports optional post-retrieval filtering by crop type and geographic region.
2. The method of the main claim wherein the LLM output is structured as JSON with fields: summary, evidence_used, confidence_level, what_to_do_next, what_NOT_to_do.
3. The system of the main claim wherein the LLM falls back to template-based explanation generation when the LLM service is unavailable, preserving evidence grounding.
4. The method of the main claim wherein the system supports multilingual explanation generation by passing a language parameter to the LLM, enabling farmer-facing output in local languages.
