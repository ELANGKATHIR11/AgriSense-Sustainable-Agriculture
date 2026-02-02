import torch
import copy
from client_node import SoilLSTM


class FederatedAggregator:
    """
    Central Server for F-SHTF.
    Orchestrates the Federated Averaging (FedAvg) process.
    """

    def __init__(self):
        self.global_model = SoilLSTM()
        self.global_weights = self.global_model.state_dict()

    def aggregate_updates(self, client_weights_list):
        """
        FedAvg Algorithm.
        w_global = sum(w_client_i) / N

        Args:
           client_weights_list: List of state_dicts from clients.
        """
        num_clients = len(client_weights_list)
        if num_clients == 0:
            return

        # Initialize with zeros
        averaged_weights = copy.deepcopy(client_weights_list[0])
        for key in averaged_weights.keys():
            averaged_weights[key] = torch.zeros_like(averaged_weights[key])

        # Summation
        for weights in client_weights_list:
            for key in weights.keys():
                averaged_weights[key] += weights[key]

        # Averaging
        for key in averaged_weights.keys():
            averaged_weights[key] = averaged_weights[key] / num_clients

        self.global_weights = averaged_weights
        self.global_model.load_state_dict(self.global_weights)
        print(f"Aggregated updates from {num_clients} clients.")

    def get_global_weights(self):
        return copy.deepcopy(self.global_weights)


# Simulation Script
if __name__ == "__main__":
    from client_node import FederatedClient

    print("Initializing F-SHTF Federation...")
    aggregator = FederatedAggregator()

    # Simulate 3 Clients with Dummy Private Data
    clients = []
    for i in range(3):
        # random data (Batch=5, Time=12, Feat=5)
        x = torch.randn(5, 12, 5)
        y = torch.randn(5, 2)
        clients.append(FederatedClient(client_id=i, private_data=(x, y)))

    # Round 1
    print("\n--- Round 1 ---")
    current_global = aggregator.get_global_weights()
    updates = []

    for client in clients:
        client.set_weights(current_global)
        # Train locally (with privacy noise)
        w_update = client.train_locally(epochs=1, dp_sigma=0.01)
        updates.append(w_update)
        print(f"Client {client.client_id} finished training.")

    aggregator.aggregate_updates(updates)
    print("Global Model Updated.")
