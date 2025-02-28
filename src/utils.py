from gensim.parsing.preprocessing import (
    strip_tags,
    strip_multiple_whitespaces,
    strip_punctuation,
    strip_non_alphanum,
)


def clean_text(text):
    text = strip_tags(text)
    text = strip_non_alphanum(text)
    text = strip_multiple_whitespaces(text)
    text = text.lower()

    return text


# %% Confusion Matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_cm(conf_matrix):
    # Labels for the classes
    labels = ["cs.CL", "cs.CR", "cs.CV", "cs.LG", "cs.RO"]

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix Heatmap")

    # Show the plot
    plt.show()


# Confusion matrix data 1
conf_matrix = np.array(
    [
        [2119, 3, 51, 19, 0],
        [43, 956, 74, 41, 3],
        [93, 6, 5466, 102, 24],
        [57, 7, 155, 356, 13],
        [4, 3, 146, 16, 697],
    ]
)
create_cm(conf_matrix)

# Confusion matrix data 2
conf_matrix = np.array(
    [
        [4213, 9, 139, 33, 2],
        [99, 1147, 132, 61, 11],
        [343, 24, 9790, 115, 44],
        [597, 53, 1373, 1969, 46],
        [32, 9, 471, 32, 1951],
    ]
)
create_cm(conf_matrix)
