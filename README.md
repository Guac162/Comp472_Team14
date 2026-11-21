# COMP 472 Project - Team 14

Mohammed, Elena and Yousra (Kaggle team COMP472_Group14)

We compare k-NN, a Decision Tree, a Random Forest and a Neural Network on two tasks: image
recognition (50 classes) and sentiment analysis (3 classes). The data is given as feature
vectors. k-NN, the tree, the forest and k-Means are written from scratch in NumPy, and the
neural network uses PyTorch. Our final Kaggle score is 0.650.

## Setup

Python 3.13. Install the libraries with:

    pip install -r requirements.txt

Download the competition data from Kaggle and put the 5 CSV files in a folder called "data"
at the root of the repo. Run all commands from the root of the repo.

## Files

src/
- "data.py" loads the data and makes the 80/20 train/validation split used by every model.
- "preprocess.py" feature scaling (standardisation, L2, sqrt).
- "utils.py" helper functions: seeds, accuracy, timing, saving results and predictions.
- "sanity.py" checks that results make sense (for example, random labels must give chance accuracy).
- "knn.py" k-NN.
- "tree.py" Decision Tree.
- "forest.py" Random Forest, built on the Decision Tree.
- "pca.py" PCA, used in one forest experiment.
- "NeuralNetwork.py" the neural network and its training.
- "Kmeans.py" k-Means.
- "Clustering.py" the clustering experiments.

experiments/
- "eda.py" a first look at the data.
- "baseline.py" always predicts the most common class, to test the pipeline.
- "knn_first.py", "knn_sweep.py", "knn_final.py", "knn_soft_final.py" k-NN: first run, tuning, final models.
- "tree_first.py", "tree_sweep.py" Decision Tree: first run, tuning and depth plots.
- "forest_first.py", "forest_sweep.py", "forest_final.py" Random Forest: first run, tuning, final model.
- "forest_pca.py" tests whether PCA helps the forest on images.
- "nn_final_sentiment.py" final neural network for sentiment.
- "ensemble.py" tests votes between models.
- "error_analysis.py" confusion matrices and per-class results.
- "NeuralNetwork" an early version of the neural network, not used.

Other
- "tests/" small tests for k-NN, the tree and the forest.
- "make_submission.py" builds and checks the Kaggle submission file.
- "splits/" the saved train/validation split.
- "results/" all results, tuning tables and chosen settings.
- "preds/" saved predictions of each model.
- "figures/" all plots.
- "submissions/" every file sent to Kaggle.

## Pre-processing

The train/validation split is already saved in "splits/", so there is nothing to run.
Feature scaling is done automatically inside each script, and it is always fitted on the
training rows only.

## Training, evaluating and applying the models

Each script trains its model, prints the validation accuracy, and saves its results to
"results/" and its predictions to "preds/". The "final" scripts retrain the chosen model on
all the training data and predict the test set.

    python -m experiments.knn_sweep
    python -m experiments.knn_soft_final
    python -m experiments.tree_sweep
    python -m experiments.forest_sweep
    python -m experiments.forest_final
    python -m src.NeuralNetwork
    python -m experiments.nn_final_sentiment
    python -m src.Clustering
    python -m experiments.error_analysis

The tree and forest tuning take about an hour each.

To make the final Kaggle file:

    python make_submission.py --image preds/test/knn_soft_full_bal_image.npy --sentiment preds/test/nn_full_sentiment.npy --note "final"

To run the tests:

    python -m tests.test_knn
    python -m tests.test_tree
    python -m tests.test_forest

## Reproducibility

The random seed is 42 in every script, and every model uses the same saved split. Library
versions are in "requirements.txt".
