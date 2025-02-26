# %%

import pandas as pd

from config import *


# %%
def load_dataset():
    df = pd.read_json(DATASET_FPATH, lines=True)

    # put comma in len() to make it more readable
    print(f"Dataset loaded with {len(df)} records")
    print(df)

    return df


df = load_dataset()
print(df["categories"].value_counts())

# %% EDA

df["primary_category"] = df["categories"].apply(lambda x: x.split(" ")[0])
print(df["primary_category"].value_counts())

# %% Category CS as the first category
cs_df = df[df["primary_category"].map(lambda x: x.startswith("cs"))]
print(cs_df.columns)
print(cs_df["primary_category"].value_counts())

# %% Category CS as the only category
cs_only_df = df[
    df["categories"].apply(lambda x: x.startswith("cs") and len(x.split(" ")) == 1)
]
cs_only_df["update_date"] = pd.to_datetime(cs_only_df["update_date"])

print(cs_only_df["categories"].value_counts())
cs_only_df["categories"].value_counts().sort_values().plot(kind="barh", figsize=(5, 8))

# %%
cs_only_df["update_date"].dt.year.value_counts().sort_index().plot(kind="bar")

# %% Top 5 categories
top5_categories = cs_only_df["categories"].value_counts().head(5)
print(top5_categories)
print(top5_categories.index)

cs_top5_df = cs_only_df[cs_only_df["categories"].isin(top5_categories.index)]
print(cs_top5_df)


# %%
import plotly.express as px


def plot_categories(df):
    fig = px.bar(
        df["categories"].value_counts().sort_values(),
        orientation="h",
        title="CS Categories",
    )
    fig.update_layout(width=400, height=800)
    fig.show()


# %%
plot_categories(cs_only_df)
plot_categories(cs_top5_df)


# %%
# by year and stack by category
def plot_year_category(df, type="line"):

    if type == "line":
        fig = px.line(
            df.groupby([df["update_date"].dt.year, "categories"]).size().unstack()
        )
    else:
        # colors should not be repeated
        fig = px.bar(
            df.groupby([df["update_date"].dt.year, "categories"]).size().unstack(),
            color="categories",
        )
    return fig


# %%
plot_year_category(cs_only_df)
# cs_only_df.groupby([cs_only_df["update_date"].dt.year, "primary_category"]).size().unstack().plot(kind="bar", stacked=True)


# %%
import torch
from sklearn.model_selection import train_test_split
from utils import clean_text


def get_train_test(df, keep_cols=["title", "abstract", "categories", "update_date"]):
    df = df[keep_cols]
    df["clean_title"] = df["title"].map(clean_text)
    df["clean_abstract"] = df["abstract"].map(clean_text)

    before_2020_df = df[
        (df["update_date"].dt.year >= 2015) & (df["update_date"].dt.year <= 2020)
    ]

    train_df, test_same_year_df = train_test_split(
        before_2020_df, test_size=0.4, random_state=42
    )

    test_diff_year_df = df[(df["update_date"].dt.year == 2023)]

    print(train_df.shape, test_same_year_df.shape, test_diff_year_df.shape)

    # save to torch file .pt
    fnames = ["train.pt", "test_same_year.pt", "test_diff_year.pt"]
    for dataset, fn in zip([train_df, test_same_year_df, test_diff_year_df], fnames):
        temp_dict = {
            "data": dataset["clean_title"].tolist(),
            "labels": dataset["categories"].tolist(),
        }

        torch.save(temp_dict, f"{DATA_FOLDER}/{fn}")

    # torch.save(train_df, "train_df.pt")

    return train_df, test_same_year_df, test_diff_year_df


train_df, test_same_year_df, test_diff_year_df = get_train_test(cs_top5_df)

# %%
plot_year_category(train_df, type="bar")
# %%
plot_year_category(test_same_year_df, type="bar")
# %%
plot_year_category(test_diff_year_df, type="bar")
# %%

# %% load the torch file

train_dataset = torch.load(f"{DATA_FOLDER}/train.pt")

# %%

print(train_dataset["data"][:5])
print(train_dataset["labels"][:5])
# %%
