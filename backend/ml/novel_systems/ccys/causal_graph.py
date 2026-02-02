import networkx as nx


class CausalAgriculturalGraph:
    """
    Structural Causal Model (SCM) for Indian Agriculture.

    Novelty:
    Explicitly models the causal graph DAG(Treatment, Confounders, Outcome)
    to allow for 'do-calculus' interventions, distinguishing causality from correlation.
    """

    def __init__(self):
        self.dag = nx.DiGraph()
        self.define_structure()

    def define_structure(self):
        """
        Defines the Nodes and Edges of the Causal Graph.

        N -> Nitrogen (Treatment)
        S -> Soil Quality (Confounder)
        W -> Weather/Rainfall (Confounder)
        Y -> Yield (Outcome)
        M -> Management Cost (Proxy for Wealth/Ability)
        """
        # Confounders affect both Treatment and Outcome
        # e.g., Better Soil (S) -> Higher Yield (Y) AND likelihood of more Fertilizer usage (N)
        self.dag.add_edge("Soil_Organic_Carbon", "Yield_Kg_Ha")
        self.dag.add_edge("Soil_Organic_Carbon", "Nitrogen_Applied")

        self.dag.add_edge("Rainfall_mm", "Yield_Kg_Ha")
        self.dag.add_edge(
            "Rainfall_mm", "Nitrogen_Applied"
        )  # Farmers apply less if no rain

        # Treatment effects
        self.dag.add_edge("Nitrogen_Applied", "Yield_Kg_Ha")
        self.dag.add_edge("Irrigation_Hours", "Yield_Kg_Ha")

        # Wealth Confounder (Unobserved usually, but we have proxy)
        self.dag.add_edge("Farmer_Credit_Score", "Nitrogen_Applied")
        self.dag.add_edge("Farmer_Credit_Score", "Irrigation_Hours")
        # Credit score doesn't directly cause yield, but causes ability to buy inputs

    def get_backdoor_sets(self, treatment="Nitrogen_Applied", outcome="Yield_Kg_Ha"):
        """
        Identifies variables that must be controlled for to estimate
        the true causal effect (Backdoor Criterion).
        """
        # In a real system, we'd use 'dowhy' logic here.
        # For prototype, we manually identify open backdoor paths.
        # Path: N <- S -> Y
        # Path: N <- W -> Y
        return ["Soil_Organic_Carbon", "Rainfall_mm", "Farmer_Credit_Score"]

    def export_gml(self):
        return "\n".join(nx.generate_gml(self.dag))
