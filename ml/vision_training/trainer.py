import os
import time

class ModelTrainer:
    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
        checkpoint_dir: str = "ml/models/checkpoints"
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int, train_data: list) -> float:
        """Simulates single epoch training forward/backward steps."""
        time.sleep(0.1) # Simulate computation delay
        loss = 0.85 / (epoch + 1)
        return float(loss)

    def fit(self, train_data: list, val_data: list) -> dict:
        """Full training loop execution with simulated validation early stopping."""
        history = {"loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0
        
        print(f"Beginning training on RTX 5060 for {self.epochs} epochs...")
        for epoch in range(1, self.epochs + 1):
            loss = self.train_epoch(epoch, train_data)
            val_loss = loss + 0.05
            
            history["loss"].append(loss)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch}/{self.epochs} - Loss: {loss:.4f} - Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if epoch % 2 == 0:
                self.save_checkpoint(epoch, loss)
                
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                praise_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                    
        return history

    def save_checkpoint(self, epoch: int, loss: float):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.bin")
        # Dummy checkpoint save
        with open(path, "w") as f:
            f.write(f"epoch={epoch}\nloss={loss}\n")
        print(f"Saved model checkpoint to {path}")
        
    def resume_training(self, checkpoint_path: str):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        # Parse checkpoint values
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                lines = f.readlines()
            print(f"Restored epoch state: {lines}")
