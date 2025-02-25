import os
import json
import zipfile
import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.vocab import GloVe
import pandas as pd

class ArxivDataset(Dataset):
    def __init__(self, zip_path, json_file_name, max_lines=1000, glove_name="6B", glove_dim=100, max_length=300):
        """
        Initialize the ArxivDataset.

        Args:
            zip_path (str): Path to the ZIP archive containing the JSON dataset.
            json_file_name (str): Name of the JSON file inside the ZIP archive.
            max_lines (int, optional): Maximum number of lines to load from the JSON file (default: 1000).
            glove_name (str, optional): Pre-trained GloVe model name (default: "6B").
            glove_dim (int, optional): Dimension of the GloVe word vectors (default: 100).
            max_length (int, optional): Maximum sequence length for text embeddings (default: 300).
        """
        self.zip_path = zip_path
        self.json_file_name = json_file_name
        self.max_length = max_length

        # Load data from ZIP archive
        self.data = self._load_arxiv_dataset_from_zip(zip_path, json_file_name, max_lines)

        # Load GloVe embeddings
        self.glove = GloVe(name=glove_name, dim=glove_dim)

        # Build category-to-index mapping
        self.category_to_index = self._build_category_index()

    def _load_arxiv_dataset_from_zip(self, zip_path, json_file_name, max_lines):
        """
        Load a subset of the arXiv dataset from a ZIP archive.

        Args:
            zip_path (str): Path to the ZIP file.
            json_file_name (str): Name of the JSON file inside the ZIP archive.
            max_lines (int): Maximum number of records to load.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the loaded dataset.
        """
        abs_zip_path = os.path.abspath(zip_path)
        data = []
        
        with zipfile.ZipFile(abs_zip_path, 'r') as zip_ref:
            with zip_ref.open(json_file_name) as json_file:
                for i, line in enumerate(json_file):
                    if i >= max_lines:
                        break
                    data.append(json.loads(line))

        return pd.DataFrame(data)

    def _build_category_index(self):
        """
        Create a dictionary mapping each category to a unique index.

        Returns:
            dict: A dictionary mapping category names to integer indices.
        """
        categories = set(self.data.get("categories", "unknown"))
        return {cat: idx for idx, cat in enumerate(sorted(categories))}

    def _text_to_embedding(self, text):
        """
        Convert text into GloVe embeddings.

        Args:
            text (str): The input text (title + abstract).

        Returns:
            torch.Tensor: A tensor of shape (max_length, glove_dim).
        """
        tokens = text.lower().split()
        embeddings = [self.glove[token] for token in tokens if token in self.glove.stoi]

        # Pad or truncate the sequence to max_length
        if len(embeddings) > self.max_length:
            embeddings = embeddings[:self.max_length]
        else:
            padding = [torch.zeros(self.glove.dim)] * (self.max_length - len(embeddings))
            embeddings.extend(padding)

        return torch.stack(embeddings)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (text_embedding, label)
        """
        row = self.data.iloc[idx]
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        category = row.get("categories", "unknown")

        # Convert text to GloVe embeddings
        text_embedding = self._text_to_embedding(title + " " + abstract)

        # Convert category to index
        label = torch.tensor(self.category_to_index.get(category, 0), dtype=torch.long)

        return text_embedding, label

    def get_dataloaders(self, train_ratio=0.8, batch_size=32, shuffle=True):
        """
        Split the dataset into training and testing sets and return DataLoaders.

        Args:
            train_ratio (float, optional): The ratio of training data (default: 0.8).
            batch_size (int, optional): The batch size for DataLoader (default: 32).
            shuffle (bool, optional): Whether to shuffle the training data (default: True).

        Returns:
            tuple: (train_loader, test_loader)
        """
        train_size = int(len(self) * train_ratio)
        test_size = len(self) - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader



"""
############ Example Usage ############
# Define the path to the ZIP file and JSON file
zip_path = "./data/archive.zip"
json_file_name = "arxiv-metadata-oai-snapshot.json"

# Load the dataset from the ZIP archive
df = load_arxiv_dataset_from_zip(zip_path, json_file_name)
print(df.head(1))
"""