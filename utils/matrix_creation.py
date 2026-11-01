import numpy as np
import pandas as pd
from scipy import sparse


# Build a sparse user-item matrix
def to_csr(df, n_users, n_items):
    data = (
        df["rating"].values,
        (df["u"].values, df["i"].values),
    )

    return sparse.csr_matrix(
        data,
        shape=(n_users, n_items),
    )


# Get the user index for each stored rating
def row_index(R):
    return np.repeat(
        np.arange(R.shape[0]),
        np.diff(R.indptr),
    )


# Calculate the average rating for each user
def user_means(R):
    counts = np.diff(R.indptr)
    sums = np.asarray(R.sum(axis=1)).ravel()

    return sums / np.maximum(counts, 1)


# Subtract each user's average from their ratings
def center(R, means):
    centered = R.copy().astype(float)
    centered.data -= means[row_index(R)]

    return centered


def save_normalized(Rc, user_ids, movie_ids, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.DataFrame(
        Rc.toarray(),
        index=user_ids,
        columns=movie_ids,
    )

    frame.rename_axis("user_id").to_csv(
        path,
        float_format="%.4g",
    )