import torch
from train import ModelTrainer
from torch.utils.data import DataLoader

from ArxivDataset import ArxivDataset
from GloveCNN import GloveCNN
from GloveRNN import GloveRNN

# %%
from config import *
import gensim.downloader as api

gensim_model = api.load(f"glove-wiki-gigaword-{EMBEDDING_DIM}")
print("GloVe embeddings loaded...")
print(f"Dimensions: {EMBEDDING_DIM}")

print("Loading data")
train_loader = DataLoader(
    ArxivDataset(TRAIN_FPATH, gensim_model=gensim_model),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
print("Loaded train data")
val_loader = DataLoader(
    ArxivDataset(VAL_FPATH, gensim_model=gensim_model), batch_size=BATCH_SIZE
)
print("Loaded validation data")
test_same_year_loader = DataLoader(
    ArxivDataset(TEST_SAME_YEAR_FPATH, gensim_model=gensim_model),
    batch_size=BATCH_SIZE,
)
print("Loaded test (same year) data")
test_diff_year_loader = DataLoader(
    ArxivDataset(TEST_DIFF_YEAR_FPATH, gensim_model=gensim_model),
    batch_size=BATCH_SIZE,
)
print("Loaded test (diff year) data")


def run(model):
    print("Loaded model")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    params = {
        "num_epochs": NUM_EPOCHS,
        "evaluate_every": EVALUATE_EVERY,
        "checkpoint_every": CHECKPOINT_EVERY,
    }

    print("Initializing model trainer")
    model_trainer = ModelTrainer(model, train_loader, val_loader, optimizer, params)
    print("Traning model")
    model_trainer.train()

    print("Generating report")
    print("------- Test on same year range data (2015 - 2020) -------")
    model_trainer.generate_report(test_same_year_loader, cm_fname="test_same")

    print("------- Test on different year range data (2023) -------")
    model_trainer.generate_report(test_diff_year_loader, cm_fname="test_diff")


print("Loading GloveCNN model")
cnn_model = GloveCNN(
    num_classes=NUM_CLASSES,
    gensim_model=gensim_model,
    embedding_dim=EMBEDDING_DIM,
    filter_sizes=FILTER_SIZES,
    num_filters=NUM_FILTERS,
    l2_reg_lambda=L2_REG_LAMBDA,
)
cnn_model.print_model_summary()
run(cnn_model)

# print("Loading GloveRNN model")
# rnn_model = GloveRNN(
#     num_classes=NUM_CLASSES,
#     num_rnn_layers=NUM_RNN_LAYERS,
#     gensim_model=gensim_model,
#     embedding_dim=EMBEDDING_DIM,
#     hidden_dim=RNN_HIDDEN_DIM,
# )

# run(rnn_model)
