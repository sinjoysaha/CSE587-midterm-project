# Percentage of the training data to use for validation
DEV_SAMPLE_PERCENTAGE = 0.2

# CNN Model Hyperparameters
EMBEDDING_DIM = 50  # Dimensionality of character embedding
FILTER_SIZES = [3, 4, 5]  # List of filter sizes
NUM_FILTERS = 128  # Number of filters per filter size
DROPOUT_KEEP_PROB = 0.5  # Dropout keep probability
L2_REG_LAMBDA = 0.0  # L2 regularization lambda
NUM_CLASSES = 5  # Number of classes
MAX_LENGTH = 256  # Maximum length of a sentence


# Training parameters
BATCH_SIZE = 512  # Batch Size
NUM_EPOCHS = 10  # Number of training epochs
EVALUATE_EVERY = 50  # Evaluate model on dev set after this many steps
CHECKPOINT_EVERY = 1000  # Save model after this many steps
NUM_CHECKPOINTS = 3  # Number of checkpoints to store

# Misc Parameters
ALLOW_SOFT_PLACEMENT = True  # Allow device soft device placement
LOG_DEVICE_PLACEMENT = False  # Log placement of ops on devices

# Paths
DATA_PATH = "./data/archive.zip"
JSON_FILE_NAME = "arxiv-metadata-oai-snapshot.json"

DATA_FOLDER = "../data/"
DATASET_FPATH = f"{DATA_FOLDER}/arxiv-metadata-oai-snapshot.json"
TRAIN_FPATH = f"{DATA_FOLDER}/train.pt"
TEST_SAME_YEAR_FPATH = f"{DATA_FOLDER}/test_same_year.pt"
TEST_DIFF_YEAR_FPATH = f"{DATA_FOLDER}/test_diff_year.pt"
