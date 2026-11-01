# Matrix Factorization Recommender System

Movie recommender on MovieLens 1M using **SVD** (`scipy.sparse.linalg.svds`) and **Probabilistic Matrix
Factorization (PMF)**, with an interactive **Streamlit** dashboard.

## Setup

```bash
pip install -r requirements.txt
python -c "import numpy, pandas, scipy, streamlit, matplotlib"
```

Download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and copy `ratings.dat`, `users.dat`
and `movies.dat` into `data/` (the GroupLens license forbids redistributing the data, so it is not included).

## Run

```bash
python train.py            # trains, evaluates, writes processed/ and reports/
streamlit run app.py       # dashboard
jupyter notebook Movie_Recommender_System.ipynb   # EDA + results walkthrough
```

## Approach

1. **Data**: nulls and duplicates removed, ids indexed, 80/20 random split with `random_state = 42`
   (a further 90/10 split of the training part is used for validation only).
2. **Matrix**: sparse user-item matrix; the user-mean-centered version is saved to
   `processed/user_item_matrix.csv`.
3. **SVD**: `svds` on the centered training matrix; `k` chosen on validation; predictions saved to
   `reports/svd_predictions.npy`.
4. **PMF**: global mean + shrunk user/item biases, then Gaussian-prior latent factors fitted by alternating
   least squares (MAP estimate). MSE is logged per iteration, convergence saved to
   `reports/pmf_convergence.png`, factors `U`, `V` (plus biases) saved in `reports/pmf_factors/`.
5. **Benchmark**: item-based k-NN collaborative filtering (`models/knn_baseline.py`).
6. **Evaluation**: test RMSE for all models in `reports/model_metrics.json` (required fields) and
   `reports/extra_metrics.json` (benchmark, MAE, chosen hyperparameters, threshold checks).

### Overfitting control and stopping rule

* SVD: `k` selected where validation RMSE is lowest; larger `k` keeps lowering train RMSE but raises
  validation RMSE (`reports/learning_curves.png`).
* PMF: L2 regularization (`lambda`), shrunk biases and early stopping: training stops after 3 iterations
  without validation improvement and the best iteration is kept; the final model is refit on the full
  training set for that many iterations.

### Interpretability

* Global: `reports/latent_factors.csv` (poles and genres of every factor, ranked by importance),
  `reports/item_similarity.png`, `reports/user_factors.png`.
* Local: `utils.recommendation.explain_recommendation` gives per-factor contributions and the most similar
  already-rated movies; plots saved as `reports/why_recommended_user_<id>.png` and shown in the dashboard.
* Three users (2 evaluated on training ratings, 1 on held-out ratings): recommendations in
  `reports/user_<id>_recommendations.csv`, analysis in `reports/user_analysis.md`.

## Structure

```
data/ models/ utils/ reports/ processed/
app.py  train.py  requirements.txt  Movie_Recommender_System.ipynb  README.md
```

## Dataset citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5, 4,
Article 19. https://doi.org/10.1145/2827872
