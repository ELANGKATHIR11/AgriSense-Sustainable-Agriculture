# Causal Counterfactual Yield Simulator (CCYS)

## System Overview
The **CCYS** moves beyond predictive correlation to prescriptive causation. It allows farmers to ask "What If" questions (Interventions) and get answers derived from a **Structural Causal Model (SCM)**, ensuring that recommendations are not just based on what "rich farmers do" but on the actual chemical/biological effect of inputs.

## Components
1.  **`causal_graph.py`**:
    *   **Class**: `CausalAgriculturalGraph`
    *   **Function**: Defines the Directed Acyclic Graph (DAG) representing domain knowledge (e.g., Rain -> Yield, Rain -> Fertilizer Usage).
    *   **Patent Novelty**: "Explicit encoding of agricultural causal assumptions into a graph structure to identify backdoor adjustment sets."

2.  **`counterfactual_engine.py`**:
    *   **Class**: `TLearnerSimulator`
    *   **Function**: Implements a Meta-Learner (T-Learner) architecture to estimate the Conditional Average Treatment Effect (CATE).
    *   **Mechanism**: $CATE(x) = E[Y|do(T=1), X=x] - E[Y|do(T=0), X=x]$

## Usage
```python
from causal_graph import CausalAgriculturalGraph
from counterfactual_engine import TLearnerSimulator

# 1. Define Causal Assumptions
dag = CausalAgriculturalGraph()
print("Confounders to control:", dag.get_backdoor_sets())

# 2. Simulate Intervention
sim = TLearnerSimulator(dag)
field_ctx = {'Soil_Organic_Carbon': 1.2, 'Rainfall_mm': 600}
intervention = {'Nitrogen_Applied': 45} # Apply 45kg Urea

result = sim.simulate_counterfactual(field_ctx, intervention)
print(f"Benefit of Intervention: {result['treatment_effect']} kg/ha")
```
