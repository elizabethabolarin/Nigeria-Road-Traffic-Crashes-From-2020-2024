"""
train.py
========
This script does the actual "teaching" of our model:

    1. Load the data.
    2. Clean it.
    3. Split it into three parts: TRAINING, VALIDATION, and TEST data.
    4. Teach (train) two different models on the training data.
    5. Check both models on the validation data, and pick the better one.
    6. Do one final honest check on the test data (which neither model
       has ever seen at all).
    7. Save the winning model to a file, so the FastAPI app and
       Streamlit app can load and reuse it later.

Run it from the terminal like this:
    python train.py
"""

import os                      # lets us work with folders/file paths
import json                    # lets us save simple info (numbers, text) into a .json file
import numpy as np             # numpy gives us math tools, like square root
import pandas as pd            # pandas lets us work with tables of data
from sklearn.model_selection import train_test_split           # splits a table into two random pieces
from sklearn.linear_model import LinearRegression               # Model 1: draws a straight-line pattern
from sklearn.ensemble import RandomForestRegressor               # Model 2: many small decision trees averaged together
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # tools that score how good a model is

from data_helper import load_data, clean_data, DataPreparer, FEATURE_COLUMNS, TARGET_COLUMN
# ^ our own file: load_data reads the CSV, clean_data fixes bad values,
#   DataPreparer turns data into numbers, and the last two are our default
#   column choices.

RANDOM_SEED = 42  # a fixed "shuffle setting" so we get the same random
                   # split every time we run this script (makes results
                   # repeatable instead of different each run)


def score_model(model, X, y):
    """Calculate 4 simple numbers that tell us how good a model's guesses are."""
    guesses = model.predict(X)                       # ask the model to guess Total_Crashes for every row in X
    mae = mean_absolute_error(y, guesses)             # average size of mistake (real answer vs. guess)
    mse = mean_squared_error(y, guesses)              # mistakes squared first, then averaged (punishes big misses more)
    rmse = np.sqrt(mse)                                # square-root MSE, brings the number back to normal "crashes" units
    r2 = r2_score(y, guesses)                          # % of the real pattern the model managed to capture (0 to 1)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}   # package all 4 numbers together in one dictionary


def train_and_save_model(
    csv_file_path="data/Nigerian_Road_Traffic_Crashes_2020_2024.csv",   # where to find the raw data file
    feature_columns=None,                                                # which columns to use as clues (default if not given)
    target_column=TARGET_COLUMN,                                         # which column to predict
    save_folder="saved_model",                                           # where to save the finished model
    show_progress=True,                                                   # whether to print updates as we go
):
    """
    Runs the full training process and saves the winning model to
    `save_folder` (a normal folder on the computer -- nothing fancy).
    """
    feature_columns = feature_columns if feature_columns else list(FEATURE_COLUMNS)
    # ^ if no feature list was given, fall back to our default list of clues

    # ---- Step 1 & 2: Load and clean the data ----------------------------
    df = load_data(csv_file_path)     # read the CSV file into a table
    df = clean_data(df)                # fix negative numbers, remove duplicate rows

    # ---- Step 3: Split into Training / Validation / Test ----------------
    # We set aside a TEST portion (20%) that we will NOT look at until the
    # very end. The remaining 80% is split again into TRAINING (60% of the
    # total) and VALIDATION (20% of the total).
    train_and_val_data, test_data = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    # ^ this line cuts off 20% of the rows (at random) to become test_data;
    #   the other 80% stays in train_and_val_data for now

    train_data, val_data = train_test_split(train_and_val_data, test_size=0.25, random_state=RANDOM_SEED)
    # ^ now cut train_and_val_data again: 25% of THIS 80% becomes val_data,
    #   which works out to 20% of the ORIGINAL total (0.25 x 0.8 = 0.2)
    # (0.25 of the remaining 80% works out to 20% of the original total)

    # ---- Step 4: Prepare the data (learn only from TRAINING data) -------
    preparer = DataPreparer(feature_columns=feature_columns, target_column=target_column)
    # ^ create our own data-preparing tool, telling it which columns matter

    X_train, y_train = preparer.learn_and_prepare(train_data)
    # ^ let it LEARN the state numbers from train_data only, then convert train_data into X/y

    X_val, y_val = preparer.prepare(val_data), val_data[target_column]
    # ^ convert val_data using what was already learned (no new learning happens here)

    X_test, y_test = preparer.prepare(test_data), test_data[target_column]
    # ^ same thing for test_data -- again, no new learning, just reusing what we already know

    # ---- Step 5: Teach two different models ------------------------------
    candidate_models = {
        "Linear Regression": LinearRegression(),                                   # Model 1: simple straight-line model
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED),
        # ^ Model 2: 200 small decision trees, all voting together, averaged into one final guess
    }

    validation_scores = {}   # we'll store each model's validation results here, by name
    for model_name, model in candidate_models.items():     # go through each of our two models, one at a time
        model.fit(X_train, y_train)                         # the model studies the training clues (X) and real answers (y)
        validation_scores[model_name] = score_model(model, X_val, y_val)
        # ^ check how well this model guesses on the validation data (data it wasn't taught with)

    # ---- Step 6: Pick the winner automatically ----------------------------
    # We pick whichever model has the LOWEST RMSE on the validation data
    # (lower RMSE = smaller average mistakes = better model).
    winner_name = min(validation_scores, key=lambda name: validation_scores[name]["RMSE"])
    # ^ min() looks through all the model names and picks whichever one has the smallest RMSE value

    winner_model = candidate_models[winner_name]   # grab the actual trained model object that won

    # ---- Step 7: One final, honest check on the TEST data -----------------
    # This data was never used to teach OR to pick the winner, so this score
    # is an honest estimate of how the model would do on brand-new data.
    test_score = score_model(winner_model, X_test, y_test)   # check the winning model on the untouched test data

    # ---- Step 8: Save the winning model + the DataPreparer to files -------
    os.makedirs(save_folder, exist_ok=True)
    # ^ create the save folder if it doesn't already exist (exist_ok=True means "don't complain if it's already there")

    joblib_model_path = os.path.join(save_folder, "trained_model.joblib")   # build the full file path for the model
    preparer_path = os.path.join(save_folder, "data_preparer.joblib")        # build the full file path for the DataPreparer

    import joblib                              # joblib: our tool for saving/loading Python objects to/from a file
    joblib.dump(winner_model, joblib_model_path)   # save the winning model to its file
    preparer.save(preparer_path)                    # save the DataPreparer (which remembers the state numbers) to its file

    info_to_remember = {                          # a small summary of what happened, saved as plain text (JSON)
        "winner_model_name": winner_name,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "validation_scores": validation_scores,
        "test_score": test_score,
        "random_seed": RANDOM_SEED,
        "split_used": "60% training / 20% validation / 20% test",
    }
    with open(os.path.join(save_folder, "model_info.json"), "w") as f:   # open a new file for writing
        json.dump(info_to_remember, f, indent=2)                          # write our summary into it, nicely formatted

    if show_progress:   # only print all this if the caller wants to see progress messages
        print("Validation scores (used to choose the winner):")
        print(pd.DataFrame(validation_scores).T.round(3))          # turn the scores into a neat table and print it
        print(f"\nWinner (lowest validation RMSE): {winner_name}")
        print("\nFinal TEST score (an honest, unbiased check):")
        print({k: round(v, 3) for k, v in test_score.items()})      # round each test number to 3 decimal places
        print(f"\nSaved model and helper files to: {save_folder}/")

    return {   # send back everything a caller (like the notebook) might want to look at afterward
        "validation_scores": validation_scores,
        "winner_name": winner_name,
        "winner_model": winner_model,
        "preparer": preparer,
        "test_score": test_score,
        "train_data": train_data, "val_data": val_data, "test_data": test_data,
    }


if __name__ == "__main__":
    # This part only runs when you type "python train.py" directly in the
    # terminal -- it does NOT run if this file is just imported by another
    # file (like the notebook does).
    train_and_save_model()
