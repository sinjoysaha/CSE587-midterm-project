import torch
import torch.nn as nn
import torchtext

# torchtext.disable_torchtext_deprecation_warning()
import torch.nn.functional as F
import numpy as np

# import gensim.downloader as api
# from torchtext.vocab import GloVe
from config import *

# glove_embeddings = GloVe(name="6B", dim=300)  # 840B
# model = api.load("word2vec-google-news-300")


class GloveCNN(nn.Module):
    """
    A CNN for text classification.
    Uses GloVe pretrained embeddings, followed by a convolutional, max-pooling, and softmax layer.
    """

    def __init__(
        self,
        num_classes,
        embedding_dim,
        filter_sizes,
        num_filters,
        gensim_model,
        embedding_freeze=True,
        l2_reg_lambda=0.0,
        dropout_rate=0.5,
    ):
        super(GloveCNN, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.embedding_freeze = embedding_freeze
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.dropout_rate = dropout_rate

        self.embedding_model = gensim_model
        print("GloVe embeddings loaded...")
        # print(f"Dimensions: {self.embedding_model.parameters["dimensions"]}")
        print(f"Dimensions: {self.embedding_dim}")

        # Embedding layer with GloVe pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            # torch.tensor(GloVe(name="6B", dim=self.embedding_dim), dtype=torch.float32),
            torch.FloatTensor(self.embedding_model.vectors),
            embedding_freeze,
        )

        # Convolutional layers
        # self.convs = nn.ModuleList(
        #     [
        #         nn.Conv2d(1, self.num_filters, kernel_size=filter_size)
        #         for filter_size in filter_sizes
        #     ]
        # )

        self.conv1 = nn.Conv1d(
            self.embedding_dim, self.num_filters, kernel_size=3, padding="same"
        )  # self.embedding_dim
        self.maxpool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(
            self.num_filters, self.num_filters, kernel_size=3, padding="same"
        )
        self.maxpool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(
            self.num_filters, self.num_filters, kernel_size=3, padding="same"
        )
        self.maxpool3 = nn.MaxPool1d(2)

        # Fully connected layer
        self.fc = nn.Linear(num_filters * (num_filters // (2 ** (3 - 1))), num_classes)

        # Dropout
        # self.dropout = nn.Dropout(dropout_rate)

        # L2 regularization loss
        self.total_loss = 0.0

    def forward(self, x):
        # Embedding lookup
        # print(x.size())
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_size)
        # x = x.unsqueeze(1)  # (batch_size, 1, sequence_length, embedding_size)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_size, sequence_length)
        # print(x.size())

        # Convolution + maxpool layers
        # pooled_outputs = []
        # for conv in self.convs:

        #     conv_out = F.relu(conv(x))
        #     print(conv_out.size())
        #     # (batch_size, num_filters, sequence_length - filter_size + 1, 1)
        #     pooled_out = F.max_pool1d(conv_out, (conv_out.size(2), 1))
        #     print(pooled_out.size())
        #     # (batch_size, num_filters, 1, 1)
        #     pooled_outputs.append(pooled_out)

        # # Combine all pooled features
        # num_filters_total = self.num_filters * len(self.filter_sizes)
        # h_pool = torch.cat(pooled_outputs, 1)  # (batch_size, num_filters_total, 1, 1)
        # h_pool_flat = h_pool.view(
        #     -1, num_filters_total
        # )  # (batch_size, num_filters_total)

        # print(x.size())
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1 = self.maxpool1(x1)
        # print(x1.size())

        x2 = self.conv2(x1)
        x2 = F.relu(x2)
        x2 = self.maxpool2(x2)
        # print(x2.size())

        x3 = self.conv3(x2)
        x3 = F.relu(x3)
        x3 = self.maxpool3(x3)
        # print(x3.size())

        # outs = torch.cat((x1, x2, x3), 1)
        # print(outs.size())

        flatten_layer = torch.flatten(x3, 1)
        # print(flatten_layer.size())

        # Dropout
        # h_drop = self.dropout(h_pool_flat)

        # Final scores and predictions
        # scores = self.fc(h_pool_flat)  # (batch_size, num_classes)
        logits = self.fc(flatten_layer)
        # print(logits.size())

        predictions = torch.argmax(logits, 1)  # (batch_size)

        return logits, predictions

    def loss(self, scores, labels):
        # Calculate mean cross-entropy loss
        losses = F.cross_entropy(scores, labels)
        self.l2_loss = sum(p.pow(2.0).sum() for p in self.parameters())
        self.total_loss = losses + self.l2_reg_lambda * self.l2_loss
        return self.total_loss

    def print_model_architecture(self):
        """
        Print model summary and architecture details.
        """
        print("\nModel Summary:")
        print(self)
        print("\nLayer Information:")
        print(f"Embedding Layer: {self.embedding}")
        for i, conv in enumerate(self.convs):
            print(f"Conv Layer {i}: {conv}")
        print(f"Fully Connected Layer: {self.fc}")
        print(f"Dropout Layer: {self.dropout}")
