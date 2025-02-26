import os
import json
import zipfile
import torch

# import torchtext

# torchtext.disable_torchtext_deprecation_warning()
from torch.utils.data import Dataset, DataLoader, random_split

# from torchtext.vocab import GloVe
import pandas as pd
from config import *
from gensim.utils import simple_preprocess


class ArxivDataset(Dataset):
    def __init__(
        self,
        data_file_path,
        gensim_model,
        max_length=MAX_LENGTH,
    ):
        """
        Initialize the ArxivDataset.

        Args:
            data_file_path (str): Path to the ZIP archive containing the JSON dataset.
            max_lines (int, optional): Maximum number of lines to load from the JSON file (default: 1000).
            glove_name (str, optional): Pre-trained GloVe model name (default: "6B").
            glove_dim (int, optional): Dimension of the GloVe word vectors (default: 100).
            max_length (int, optional): Maximum sequence length for text embeddings (default: 300).
        """
        self.data_file_path = data_file_path
        # self.json_file_name = json_file_name
        self.max_length = max_length

        # Load data from ZIP archive
        # self.data = self._load_arxiv_dataset_from_zip(data_file_path, json_file_name, max_lines)
        data = torch.load(self.data_file_path)
        self.data = data["data"]
        self.labels = data["labels"]
        assert len(self.data) == len(
            self.labels
        ), "Data and labels must have the same length."

        # Load GloVe embeddings
        self.embedding_model = gensim_model  # api.load(gensim_model)
        self.padding_token_id = self.embedding_model.key_to_index["-"]

        # Build category-to-index mapping
        self.category_to_index, self.index_to_category = self._build_category_index()

    @DeprecationWarning
    def _load_arxiv_dataset_from_zip(self, data_file_path, json_file_name, max_lines):
        """
        Load a subset of the arXiv dataset from a ZIP archive.

        Args:
            data_file_path (str): Path to the ZIP file.
            json_file_name (str): Name of the JSON file inside the ZIP archive.
            max_lines (int): Maximum number of records to load.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the loaded dataset.
        """
        abs_data_file_path = os.path.abspath(data_file_path)
        data = []

        with zipfile.ZipFile(abs_data_file_path, "r") as zip_ref:
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
        categories = sorted(set(self.labels))
        return {cat: idx for idx, cat in enumerate(categories)}, {
            idx: cat for cat, idx in enumerate(categories)
        }

    def _text_to_tokens(self, text):
        """
        Convert text into GloVe embeddings.

        Args:
            text (str): The input text (title + abstract).

        Returns:
            torch.Tensor: A tensor of shape (max_length, glove_dim).
        """

        # print(self.padding_token_id)
        # print(text)
        tokens = [
            self.embedding_model.key_to_index[word]
            for word in simple_preprocess(text)
            if word in self.embedding_model.key_to_index
        ]
        # # embeddings = [self.glove[token] for token in tokens if token in self.glove.stoi]

        # print(tokens)

        # Pad or truncate the sequence to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            padding = [self.padding_token_id] * (self.max_length - len(tokens))
            tokens.extend(padding)

        # print(tokens)

        return torch.tensor(tokens, dtype=torch.long)

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
        # row = self.data[idx]
        # title = row.get("title", "")
        # abstract = row.get("abstract", "")
        # category = row.get("categories", "unknown")

        # Convert text to GloVe embeddings
        tokens = self._text_to_tokens(self.data[idx])
        # print(tokens)
        # Convert category to index
        label = torch.tensor(
            self.category_to_index.get(self.labels[idx], 0), dtype=torch.long
        )

        return tokens, label

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
data_file_path = "./data/archive.zip"
json_file_name = "arxiv-metadata-oai-snapshot.json"

# Load the dataset from the ZIP archive
df = load_arxiv_dataset_from_zip(data_file_path, json_file_name)
print(df.head(1))
"""
