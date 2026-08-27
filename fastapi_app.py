"""
fastapi_app.py
================
This is our WEB API for the Nigerian Road Traffic Crashes project.

Unlike our earlier version, this API TRAINS its own models -- you call
a /train address first, and the API teaches two models (Linear
Regression and Random Forest) right there and then, and keeps them in
memory ready to use. This follows the same overall pattern as
iris_main.py: train first, then predict.

Everything lives in this ONE file, organized top to bottom:

    1. IMPORTS
    2. APP SETUP & CONSTANTS
    3. MEMORY STORE (where the trained models live while the app is running)
    4. INPUT SHAPES (Pydantic models describing valid requests)
    5. HELPER FUNCTIONS (data loading/cleaning/checking)
    6. THE WEB ADDRESSES (ROUTES): /train, /predict/single, /predict/batch

To start this API, run this in your terminal:
    uvicorn fastapi_app:app --reload

Then open this in your browser to see and test it interactively:
    http://127.0.0.1:8000/docs
"""

# ----------------------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, status
# ^ FastAPI: turns our Python functions into a real web API.
#   HTTPException: lets us send back a clear error message when something goes wrong.
#   status: gives us readable names for web response codes, like status.HTTP_400_BAD_REQUEST.

from typing import List, Optional        # "List" means a list of things; "Optional" means a value that can be left out
from pydantic import BaseModel, Field     # BaseModel/Field describe exactly what a valid request must look like

import pandas as pd                       # for working with data as tables
import numpy as np                        # for math tools, like square root
from sklearn.model_selection import train_test_split          # splits data into random pieces
from sklearn.preprocessing import LabelEncoder                  # turns state names (text) into numbers
from sklearn.linear_model import LinearRegression                # Model 1: straight-line model
from sklearn.ensemble import RandomForestRegressor                # Model 2: many small trees averaged together
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # tools that score model accuracy


# ----------------------------------------------------------------------
# 2. APP SETUP & CONSTANTS
# ----------------------------------------------------------------------
app = FastAPI(
    title="Nigerian Traffic Crashes - Prediction API",
    description="Train Linear Regression and Random Forest models on Nigerian road "
                "crash data, then predict Total_Crashes with either one.",
    version="1.0.0",
)

DATA_FILE = "Nigerian_Road_Traffic_Crashes_2020_2024.csv"   # where the raw data lives

FEATURE_NAMES = ["State", "Year", "Quarter_Num", "SPV", "DAD", "PWR", "FTQ", "Other_Factors"]
# ^ the columns the models use as clues

TARGET_NAME = "Total_Crashes"   # the column we're trying to predict


# ----------------------------------------------------------------------
# 3. MEMORY STORE
#    This is where our trained models live WHILE THE APP IS RUNNING.
#    It starts out empty (all None) and only gets filled in once /train
#    has been called successfully.
# ----------------------------------------------------------------------
model_store = {
    "linear_model": None,          # will hold the trained Linear Regression model
    "forest_model": None,          # will hold the trained Random Forest model
    "state_encoder": None,         # will hold the tool that turns state names into numbers
    "winner_model_name": None,     # will hold whichever model name made smaller mistakes
    "feature_names": FEATURE_NAMES,
}


# ----------------------------------------------------------------------
# 4. INPUT SHAPES
#    These describe exactly what a valid request must contain. FastAPI
#    automatically checks every incoming request against these, and
#    sends back a clear error if something doesn't match.
# ----------------------------------------------------------------------

class TrainingConfig(BaseModel):
    """Optional settings for the /train step. Sensible defaults are used if you don't set these."""
    test_size: float = Field(default=0.2, gt=0, lt=0.9, description="Fraction of data held out for testing, e.g. 0.2 = 20%")
    random_forest_trees: int = Field(default=200, ge=10, le=1000, description="How many trees the Random Forest uses")

    model_config = {
        "json_schema_extra": {
            "example": {
                "test_size": 0.2,
                "random_forest_trees": 200,
            }
        }
    }


class CrashObservation(BaseModel):
    """The information needed to predict Total_Crashes for ONE state/quarter."""
    State: str = Field(..., description="Nigerian state name, e.g. 'Lagos'")
    Year: int = Field(..., description="e.g. 2023")
    Quarter_Num: int = Field(..., ge=1, le=4, description="1, 2, 3, or 4")
    SPV: int = Field(..., ge=0, description="Speed Violation incidents")
    DAD: int = Field(..., ge=0, description="Dangerous Driving incidents")
    PWR: int = Field(..., ge=0, description="Poor Weather/Road incidents")
    FTQ: int = Field(..., ge=0)
    Other_Factors: int = Field(..., ge=0)
    model: Optional[str] = Field(
        default=None,
        description="Which model to use: 'Linear Regression' or 'Random Forest'. "
                     "Leave blank to automatically use whichever model won during training.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "State": "Lagos", "Year": 2023, "Quarter_Num": 2,
                "SPV": 20, "DAD": 5, "PWR": 2, "FTQ": 1, "Other_Factors": 10,
                "model": "Random Forest",
            }
        }
    }


class BatchCrashRequest(BaseModel):
    """A list of MULTIPLE records, for predicting many at once."""
    observations: List[CrashObservation] = Field(..., min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "observations": [
                    {"State": "Lagos", "Year": 2023, "Quarter_Num": 2, "SPV": 20, "DAD": 5, "PWR": 2, "FTQ": 1, "Other_Factors": 10},
                    {"State": "Abia", "Year": 2021, "Quarter_Num": 1, "SPV": 3, "DAD": 1, "PWR": 0, "FTQ": 0, "Other_Factors": 2},
                ]
            }
        }
    }


# ----------------------------------------------------------------------
# 5. HELPER FUNCTIONS
# ----------------------------------------------------------------------

def verify_model_trained():
    """Checks that /train has already been called successfully before letting a prediction happen."""
    if model_store["linear_model"] is None or model_store["forest_model"] is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Models are not trained yet. Please call the /train endpoint first.",
        )


def load_and_clean_data():
    """Reads the CSV file and fixes the two obvious problems in it."""
    df = pd.read_csv(DATA_FILE)                    # read the raw file into a table
    df = df.drop_duplicates()                        # remove any exact duplicate rows
    df["Other_Factors"] = df["Other_Factors"].clip(lower=0)   # a crash count can never be negative, so fix any that are
    return df


def add_time_columns(df):
    """Splits the text column 'Quarter' (e.g. "Q4 2020") into two NUMBER columns: Quarter_Num and Year."""
    df = df.copy()
    pieces = df["Quarter"].str.split(" ", expand=True)                     # "Q4 2020" -> ["Q4", "2020"]
    df["Quarter_Num"] = pieces[0].str.replace("Q", "", regex=False).astype(int)   # "Q4" -> 4
    df["Year"] = pieces[1].astype(int)                                       # "2020" -> 2020
    return df


def encode_state_column(df, state_encoder, is_training):
    """
    Turns the State column (text, e.g. "Lagos") into numbers the model
    can use. If is_training=True, the encoder LEARNS the state names
    first. If is_training=False, it just reuses what it already learned
    (used for predictions, so we never "cheat" by learning from new data).
    """
    df = df.copy()
    if is_training:
        df["State"] = state_encoder.fit_transform(df["State"])   # learn each state's number, then convert
    else:
        known_states = set(state_encoder.classes_)                 # the state names the encoder already knows
        df["State"] = df["State"].apply(
            lambda name: int(state_encoder.transform([name])[0]) if name in known_states else -1
            # ^ known state -> its learned number; unknown/typo state -> placeholder -1, instead of crashing
        )
    return df


def score_model(model, X, y):
    """Calculates 4 standard numbers that describe how accurate a model's guesses are."""
    guesses = model.predict(X)
    mae = mean_absolute_error(y, guesses)              # average size of mistake
    mse = mean_squared_error(y, guesses)                 # mistakes squared, then averaged (punishes big misses more)
    rmse = np.sqrt(mse)                                    # brought back to normal "crashes" units
    r2 = r2_score(y, guesses)                               # % of the real pattern the model explains
    return {"mae": round(mae, 3), "mse": round(mse, 3), "rmse": round(rmse, 3), "r2": round(r2, 3)}


def choose_model(requested_model_name):
    """Picks which trained model to use: the one requested, or the automatic winner if none was asked for."""
    if requested_model_name is None:
        chosen_name = model_store["winner_model_name"]
    elif requested_model_name in ("Linear Regression", "Random Forest"):
        chosen_name = requested_model_name
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model '{requested_model_name}'. Choose 'Linear Regression' or 'Random Forest'.",
        )

    model = model_store["linear_model"] if chosen_name == "Linear Regression" else model_store["forest_model"]
    return model, chosen_name


def prepare_one_row(observation: CrashObservation):
    """Turns ONE validated request into a one-row table the model can predict from."""
    row_dict = observation.model_dump(exclude={"model"})   # every field EXCEPT "model" (that's not a real data column)
    row_df = pd.DataFrame([row_dict])                         # wrap it in a one-row table
    row_df = encode_state_column(row_df, model_store["state_encoder"], is_training=False)   # convert State to a number
    return row_df[FEATURE_NAMES]                                # keep only the columns the model expects, in order


# ----------------------------------------------------------------------
# 6. THE WEB ADDRESSES (ROUTES)
# ----------------------------------------------------------------------

@app.get("/")
def home():
    """A simple 'is it working?' check."""
    return {
        "status": "The API is running!",
        "models_trained": model_store["linear_model"] is not None,
        "steps": ["1. POST /train", "2. POST /predict/single or /predict/batch"],
        "docs": "/docs",
    }


@app.post("/train", status_code=status.HTTP_200_OK, summary="1. Train Both Models")
async def train(config: TrainingConfig):
    """
    Loads the crash data, cleans it, trains Linear Regression AND Random
    Forest, scores both on held-out test data, and remembers whichever
    one made the smaller mistakes as the "winner" (used by default when
    a prediction request doesn't specify a model).
    """
    # ---- Load and prepare the data -------------------------------------
    df = load_and_clean_data()               # read the CSV and fix obvious problems
    df = add_time_columns(df)                  # split "Quarter" into Quarter_Num and Year

    # ---- Split into training data and test data -------------------------
    train_df, test_df = train_test_split(df, test_size=config.test_size, random_state=42)

    # ---- Turn State into numbers, LEARNING only from training data ------
    state_encoder = LabelEncoder()
    train_df = encode_state_column(train_df, state_encoder, is_training=True)     # learns + converts
    test_df = encode_state_column(test_df, state_encoder, is_training=False)        # only converts, using what was learned

    X_train, y_train = train_df[FEATURE_NAMES], train_df[TARGET_NAME]
    X_test, y_test = test_df[FEATURE_NAMES], test_df[TARGET_NAME]

    # ---- Train Model 1: Linear Regression --------------------------------
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)                       # the model studies the training clues and answers
    linear_scores = score_model(linear_model, X_test, y_test)   # check its accuracy on the held-out test data

    # ---- Train Model 2: Random Forest -------------------------------------
    forest_model = RandomForestRegressor(n_estimators=config.random_forest_trees, random_state=42)
    forest_model.fit(X_train, y_train)
    forest_scores = score_model(forest_model, X_test, y_test)

    # ---- Automatically decide the winner (smaller RMSE = better) -----------
    winner_name = "Linear Regression" if linear_scores["rmse"] <= forest_scores["rmse"] else "Random Forest"

    # ---- Save everything into memory, ready for /predict calls -------------
    model_store["linear_model"] = linear_model
    model_store["forest_model"] = forest_model
    model_store["state_encoder"] = state_encoder
    model_store["winner_model_name"] = winner_name

    return {
        "status": "success",
        "message": "Both models were trained successfully.",
        "rows_used_for_training": len(train_df),
        "rows_used_for_testing": len(test_df),
        "test_metrics": {
            "Linear Regression": linear_scores,
            "Random Forest": forest_scores,
        },
        "automatically_chosen_winner": winner_name,
    }


@app.post("/predict/single", summary="2. Predict Single Observation")
async def predict_single(observation: CrashObservation):
    """Send ONE record here, get back ONE predicted number of crashes."""
    verify_model_trained()                                  # make sure /train has been called already

    chosen_model, model_name_used = choose_model(observation.model)   # pick the requested model, or the winner
    row = prepare_one_row(observation)                          # turn the request into a model-ready table
    prediction = float(chosen_model.predict(row)[0])              # ask the model to guess; [0] grabs that one answer

    return {
        "predicted_total_crashes": round(prediction, 2),
        "model_used": model_name_used,
    }


@app.post("/predict/batch", summary="3. Predict Batch Observation")
async def predict_batch(request: BatchCrashRequest):
    """Send a LIST of records here, get back a LIST of predicted numbers."""
    verify_model_trained()

    results = []                                              # we'll build up one result per observation here
    for index, observation in enumerate(request.observations):  # go through each observation, one at a time
        chosen_model, model_name_used = choose_model(observation.model)
        row = prepare_one_row(observation)
        prediction = float(chosen_model.predict(row)[0])

        results.append({
            "sample_index": index,
            "predicted_total_crashes": round(prediction, 2),
            "model_used": model_name_used,
        })

    return {
        "total_samples": len(results),
        "predictions": results,
    }
