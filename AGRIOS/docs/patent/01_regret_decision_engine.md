# Patent Novelty Claim #01: Minimax Regret-Based Decision Engine for Agricultural AI

## Title
Method and System for Minimax Regret-Based Action Gating in Autonomous Agricultural Decision-Making

## Mechanism

The Decision Governor employs minimax regret ranking rather than traditional argmax-confidence selection to determine agricultural interventions. For each candidate action (ACT, WAIT, OBSERVE, DO_NOTHING), the system computes:

```
regret(action) = max(0, expected_loss(action) - expected_loss(best_alternative))
```

Where expected losses are computed using crop-specific loss weight matrices that encode the relative costs of:
- False positive action (treating when unnecessary)
- Missed detection (not treating when necessary)  
- Unnecessary waiting (delay when action is safe)
- Observation cost (monitoring overhead)

The system then selects the action that minimizes worst-case regret across all possible states of nature, rather than the action with highest expected utility.

## Why Non-Obvious

Traditional agricultural AI systems use **argmax confidence** — they select the action associated with the highest classification confidence score. This approach fails in three critical ways that the regret-based method solves:

1. **Asymmetric loss landscapes**: In agriculture, the cost of a false positive treatment (e.g., unnecessary pesticide application) differs dramatically from a missed detection (e.g., untreated blight spreading). Confidence scores treat these symmetrically.

2. **State uncertainty**: When confidence is moderate (0.5-0.7), argmax may recommend action, but the regret framework considers what happens if that confidence is wrong — potentially recommending WAIT when the cost of being wrong about ACT exceeds the cost of delayed action.

3. **Multi-signal fusion**: The regret computation integrates heterogeneous signals (vision confidence, sensor readings, historical VRAG evidence, anomaly scores) through a unified loss framework rather than simple averaging or voting.

## System Claim

A computer-implemented system for autonomous agricultural decision-making comprising:
- A Decision Governor module that receives multi-modal inputs (vision embeddings, sensor readings, retrieval evidence, anomaly flags)
- A crop-specific loss weight matrix encoding asymmetric costs of agricultural interventions
- A minimax regret computation engine that ranks candidate actions by worst-case regret
- A bootstrap percentile confidence band generator that provides calibrated uncertainty estimates
- An anomaly gate that caps maximum action severity when out-of-distribution inputs are detected

## Method Claim

A method for gating agricultural AI actions comprising:
1. Receiving multi-modal agricultural signals including at least one of: visual crop analysis, soil sensor readings, and historical case retrieval
2. Computing bootstrap percentile confidence bands from the plurality of signal confidence scores
3. For each candidate action in a predefined action set, computing expected loss using crop-specific loss weight matrices
4. Computing minimax regret as the difference between each action's expected loss and the minimum expected loss across all candidates
5. Selecting the action with minimum worst-case regret, subject to anomaly gating constraints
6. Annotating the selected action with regret score, confidence band, and evidence chain

## Dependent Claims

1. The system of the main claim wherein the anomaly gate forces the action to OBSERVE when embedding-space anomaly detection indicates out-of-distribution inputs.
2. The method of the main claim wherein crop-specific loss weights are dynamically adjusted based on growth stage, seasonal factors, and historical outcome feedback.
3. The system of the main claim wherein the confidence band is computed using bootstrap resampling with at least 1000 resamples from the plurality of signal confidence scores.
4. The method of the main claim wherein the regret score and evidence chain are provided to a constrained language model that generates farmer-facing explanations grounded only in the provided evidence.
