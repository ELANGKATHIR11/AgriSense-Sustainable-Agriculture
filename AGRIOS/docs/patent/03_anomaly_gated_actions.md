# Patent Novelty Claim #03: Isolation Forest Anomaly Gate for Agricultural AI Action Prevention

## Title
System and Method for Distribution-Aware Anomaly Gating Preventing AI-Driven Agricultural Actions on Out-of-Distribution Inputs

## Mechanism

An Isolation Forest model trained on the embedding space of known-good crop disease images acts as a hard gate on the Decision Governor. The mechanism operates as follows:

1. **Training**: The Isolation Forest (100 estimators, 5% contamination rate) is trained on all 384-dimensional DeiT embeddings from the validated disease dataset. This learns the distribution boundary of "known" agricultural conditions.

2. **Inference**: Each new image embedding is passed through the Isolation Forest before reaching the Decision Governor.

3. **Gating Logic**:
   - If `prediction == -1` (anomaly): The Decision Governor's maximum allowed action is capped at **OBSERVE** — it can never ACT or WAIT on anomalous inputs
   - If `prediction == 1` (inlier): The gate is open; the Decision Governor decides freely based on confidence bands and regret scores

4. **Score Propagation**: The anomaly score (decision function value) is propagated to the Decision Governor's regret computation, increasing the expected loss of ACT for inputs near the distribution boundary.

## Why Non-Obvious

1. **Confidence thresholds are insufficient**: Most agricultural AI systems use a confidence threshold (e.g., "if confidence > 0.8, recommend treatment"). However, neural networks are known to produce high-confidence predictions on OOD inputs. An image of a car could receive 0.95 confidence for "tomato blight." The Isolation Forest operates in embedding space, detecting distributional anomalies regardless of downstream confidence.

2. **Embedding-space vs. pixel-space**: Operating the anomaly detector in the 384-dimensional DeiT embedding space rather than pixel space is non-obvious because:
   - Pixel-space anomaly detection would require much higher dimensionality
   - DeiT embeddings capture semantic similarity, not visual artifacts
   - The same Isolation Forest works across all crop types because the embedding space is shared

3. **Hard gate vs. soft penalty**: The hard gate (cap at OBSERVE) rather than a soft confidence penalty is non-obvious because soft approaches allow the system to "reason past" anomalies through high-scoring other signals. The hard gate ensures that when the system encounters something it has never seen, it ALWAYS defers to human judgment.

4. **Composition with regret scoring**: The anomaly score is additionally used in regret computation, creating a dual-layer defense: the hard gate prevents action, while the score increases regret of aggressive actions even for borderline cases.

## System Claim

A computer-implemented system for preventing AI-driven agricultural actions on out-of-distribution inputs comprising:
- An Isolation Forest anomaly detector trained on validated crop disease embeddings in a 384-dimensional DeiT feature space
- A hard action gate that caps the maximum allowed action at OBSERVE when anomalous inputs are detected
- A score propagation pathway that increases regret scores for aggressive actions on borderline inputs
- Integration with the Decision Governor such that anomaly gating occurs before confidence-based decision logic

## Method Claim

A method for anomaly-gated agricultural decision-making comprising:
1. Training an Isolation Forest model on DeiT embeddings of validated crop disease images with a contamination parameter calibrated to the expected rate of anomalous field images
2. For each new input image, extracting a DeiT embedding and evaluating it through the trained Isolation Forest
3. If the Isolation Forest predicts the input as anomalous (prediction == -1): setting a hard gate that prevents the Decision Governor from recommending any action more aggressive than OBSERVE
4. Propagating the anomaly decision function score to the Governor's regret computation, increasing the expected loss of ACT and WAIT actions proportionally to the anomaly severity
5. Annotating the decision output with the anomaly flag, score, and human-readable reason for transparency

## Dependent Claims

1. The system of the main claim wherein the Isolation Forest uses 100 estimators and a contamination rate of 0.05, calibrated for agricultural imaging conditions.
2. The method of the main claim wherein the anomaly gate model is retrained periodically as new validated disease images are added to the dataset, adapting the distribution boundary.
3. The system of the main claim wherein the anomaly detection runs in under 5 milliseconds on CPU, enabling real-time gating on edge devices.
4. The method of the main claim wherein anomalous inputs trigger an automatic request for additional context (e.g., prompting the user to capture additional images or provide manual description).
