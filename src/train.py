import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix

from config import NUM_CLASSES


class ModelTrainer:
    """
    A class to handle model training and evaluation.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, params):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        # self.criterion = criterion
        self.params = params

        # TensorBoard writer
        self.out_dir = os.path.abspath(
            os.path.join(
                os.path.curdir,
                "runs",
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
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
                logits, preds = self.model(x_batch)
                loss = self.model.loss(logits, y_batch)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Log training metrics
                accuracy = (preds == y_batch).float().mean()
                self.train_summary_writer.add_scalar("loss", loss.item(), global_step)
                self.train_summary_writer.add_scalar(
                    "accuracy", accuracy.item(), global_step
                )

                if global_step % 10 == 0:
                    print(
                        f"global_step: {global_step} | train_loss: {loss.item():.4f} | train_acc: {accuracy.item():.4f}"
                    )

                if global_step % self.params["evaluate_every"] == 0:
                    self.evaluate(global_step)

                if global_step % self.params["checkpoint_every"] == 0:
                    self.save_checkpoint(global_step)

                global_step += 1

        print("Training complete.")
        self.save_checkpoint(global_step)

    def evaluate(self, global_step):
        """
        Evaluate the model on the validation set.
        """
        print("\nEvaluating model...")
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val_batch, y_val_batch in self.val_loader:
                val_logits, val_preds = self.model(x_val_batch)
                val_loss += self.model.loss(val_logits, y_val_batch).item()

        val_loss /= len(self.val_loader)
        val_acc = (val_preds == y_val_batch).float().mean()
        self.val_summary_writer.add_scalar("loss", val_loss, global_step)
        self.val_summary_writer.add_scalar("accuracy", val_acc.item(), global_step)
        print(
            f"global_step: {global_step} | val_loss: {val_loss:.4f} | val_acc : {val_acc:.4f}\n"
        )

    def save_checkpoint(self, global_step):
        """
        Save model checkpoint.
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"model_step_{global_step}.pt"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}\n")

    def generate_report(self, test_loader, cm_fname=""):
        """
        Generate a report for the model.
        """
        print("\nTesting model...")
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_test_batch, y_test_batch in test_loader:
                test_logits, test_preds = self.model(x_test_batch)
                test_loss += self.model.loss(test_logits, y_test_batch).item()
                all_preds.extend(test_preds.tolist())
                all_labels.extend(y_test_batch.tolist())

        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss:.4f}")

        # classification report
        # all_preds = np.array(all_preds)
        # all_labels = np.array(all_labels)
        # get test dataset from test dataloader object
        target_names = [
            test_loader.dataset.index_to_category[i] for i in range(NUM_CLASSES)
        ]

        print(classification_report(all_labels, all_preds, target_names=target_names))

        # confusion matrix
        # import confusion_matrix
        cm = confusion_matrix(
            list(map(lambda x: test_loader.dataset.index_to_category[x], all_labels)),
            list(map(lambda x: test_loader.dataset.index_to_category[x], all_preds)),
            labels=target_names,
        )
        print(cm)
        # visualize confusion matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        cfm_plot = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # save plt image
        cfm_plot.figure.savefig(f"{self.out_dir}/{cm_fname}_cm.png")
        print("Testing complete.")
