from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RANDOM_STATE = 42


# Read a MovieLens .dat file using the provided column names.
def _read_dat(name, columns, data_dir):
    path = Path(data_dir) / name

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. "
            "Place the MovieLens 1M files in the data/ folder."
        )

    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=columns,
        encoding="latin-1",
    )


# Load ratings and remove missing or duplicate user-movie entries.
def load_ratings(data_dir=DATA_DIR):
    ratings = _read_dat(
        "ratings.dat",
        ["user_id", "movie_id", "rating", "timestamp"],
        data_dir,
    )

    ratings = ratings.dropna()
    ratings = ratings.drop_duplicates(["user_id", "movie_id"])
    ratings = ratings.astype({
        "user_id": int,
        "movie_id": int,
        "rating": float,
    })

    return ratings.reset_index(drop=True)


# Load movie information and remove missing or duplicate movies.
def load_movies(data_dir=DATA_DIR):
    movies = _read_dat(
        "movies.dat",
        ["movie_id", "title", "genres"],
        data_dir,
    )

    return (
        movies.dropna()
        .drop_duplicates("movie_id")
        .reset_index(drop=True)
    )


# Load user information and remove rows containing missing values.
def load_users(data_dir=DATA_DIR):
    columns = [
        "user_id",
        "gender",
        "age",
        "occupation",
        "zip_code",
    ]

    users = _read_dat("users.dat", columns, data_dir)

    return users.dropna().reset_index(drop=True)


# Create sorted arrays containing all unique user and movie IDs.
def build_index(ratings):
    user_ids = np.sort(ratings["user_id"].unique())
    movie_ids = np.sort(ratings["movie_id"].unique())

    return user_ids, movie_ids


# Convert original user and movie IDs into numerical matrix indices.
def encode(df, user_ids, movie_ids):
    encoded = df.copy()

    encoded["u"] = np.searchsorted(
        user_ids,
        encoded["user_id"].values,
    )

    encoded["i"] = np.searchsorted(
        movie_ids,
        encoded["movie_id"].values,
    )

    return encoded


# Randomly split ratings into training and testing datasets.
def split_ratings(df, test_size=0.2, random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(len(df))
    n_test = int(len(df) * test_size)

    test = (
        df.iloc[indices[:n_test]]
        .sort_index()
        .reset_index(drop=True)
    )

    train = (
        df.iloc[indices[n_test:]]
        .sort_index()
        .reset_index(drop=True)
    )

    return train, test