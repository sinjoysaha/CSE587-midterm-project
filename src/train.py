import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ModelTrainer:
    """
    A class to handle model training and evaluation.
    """

    def __init__(self, model, train_loader, test_loader, optimizer, params):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        # self.criterion = criterion
        self.params = params

        # TensorBoard writer
        self.out_dir = os.path.abspath(
            os.path.join(os.path.curdir, "runs", str(int(time.time())))
        )
        self.train_summary_writer = SummaryWriter(os.path.join(self.out_dir, "train"))
        self.val_summary_writer = SummaryWriter(os.path.join(self.out_dir, "val"))

        # Checkpoint directory
        self.checkpoint_dir = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        """
        Train the model using the provided training and validation data loaders.
        """
        global_step = 0
        for epoch in range(self.params["num_epochs"]):
            print(f"\nEpoch {epoch + 1}/{self.params['num_epochs']}")
            for x_batch, y_batch in self.train_loader:
                self.model.train()
                self.optimizer.zero_grad()

                # Forward pass
                scores, _ = self.model(x_batch)
                loss = self.model.loss(scores, y_batch)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Log training metrics

                self.train_summary_writer.add_scalar("loss", loss.item(), global_step)
                if global_step % 10 == 0:
                    print(f"global_step: {global_step} | train_loss: {loss.item():.4f}")

                if global_step % self.params["evaluate_every"] == 0:
                    self.evaluate(global_step)

                if global_step % self.params["checkpoint_every"] == 0:
                    self.save_checkpoint(global_step)

                global_step += 1

        print("Training complete.")

    def evaluate(self, global_step):
        """
        Evaluate the model on the validation set.
        """
        print("\nEvaluating model...")
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_dev_batch, y_dev_batch in self.test_loader:
                dev_scores, _ = self.model(x_dev_batch)
                val_loss += self.model.loss(dev_scores, y_dev_batch).item()

        val_loss /= len(self.test_loader)
        self.val_summary_writer.add_scalar("loss", val_loss, global_step)
        print(f"global_step: {global_step} | val_Loss: {val_loss:.4f}\n")

    def save_checkpoint(self, global_step):
        """
        Save model checkpoint.
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"model_step_{global_step}.pt"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}\n")
