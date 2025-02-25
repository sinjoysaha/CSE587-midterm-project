import torch
import torch.nn as nn
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch.nn.functional as F
import numpy as np

from torchtext.vocab import GloVe
glove_embeddings = GloVe(name='6B', dim=300) # 840B

# import gensim.downloader as api
# model = api.load("word2vec-google-news-300")

class GloveCNN(nn.Module):
    """
    A CNN for text classification.
    Uses GloVe pretrained embeddings, followed by a convolutional, max-pooling, and softmax layer.
    """
    def __init__(self, num_classes, embedding_size, filter_sizes, num_filters, glove_embeddings = GloVe(name='6B', dim=300), embedding_freeze = True, l2_reg_lambda=0.0, dropout_rate = 0.5):
        super(GloveCNN, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.embedding_freeze = embedding_freeze
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.dropout_rate = dropout_rate 
        

        # Embedding layer with GloVe pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(glove_embeddings, dtype=torch.float32), embedding_freeze)
        
        # Convolutional layers
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (filter_size, embedding_size)) for filter_size in filter_sizes])
        
        # Fully connected layer
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # L2 regularization loss
        self.total_loss = 0.0

    def forward(self, x):
        # Embedding lookup
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_size)
        x = x.unsqueeze(1)  # (batch_size, 1, sequence_length, embedding_size)
        
        # Convolution + maxpool layers
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, sequence_length - filter_size + 1, 1)
            pooled_out = F.max_pool2d(conv_out, (conv_out.size(2), 1))  # (batch_size, num_filters, 1, 1)
            pooled_outputs.append(pooled_out)
        
        # Combine all pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = torch.cat(pooled_outputs, 1)  # (batch_size, num_filters_total, 1, 1)
        h_pool_flat = h_pool.view(-1, num_filters_total)  # (batch_size, num_filters_total)
        
        # Dropout
        h_drop = self.dropout(h_pool_flat)
        
        # Final scores and predictions
        scores = self.fc(h_drop)  # (batch_size, num_classes)
        predictions = torch.argmax(scores, 1)  # (batch_size)
        
        return scores, predictions

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