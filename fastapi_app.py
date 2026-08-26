"""
fastapi_app.py
================
This is our WEB API. A "web API" is just a program that other apps (or
you, using a browser or a tool like curl) can send a request to, and it
sends back an answer -- in our case, a predicted number of crashes.

Everything for the API lives in this ONE file, organized top to bottom:

    1. IMPORTS               -- tools we need
    2. INPUT SHAPES           -- what a valid request must look like
    3. LOADING THE SAVED MODEL -- happens once, when the app starts
    4. THE ACTUAL PREDICTING FUNCTIONS
    5. THE WEB ADDRESSES (ROUTES) people can send requests to

To start this API, run this in your terminal:
    uvicorn fastapi_app:app --reload

Then open this in your browser to see and test it interactively:
    http://127.0.0.1:8000/docs
"""

# ----------------------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
# ^ FastAPI: the tool that turns our Python functions into a real web API.
#   HTTPException: lets us send back a clear error message when something goes wrong.

from pydantic import BaseModel, Field
# ^ BaseModel: lets us describe exactly what a valid request should look like.
#   Field: lets us add extra details (like an example value) to each piece of the request.

from typing import List             # lets us say "a list of things" when describing our data shapes
import pandas as pd                 # turns incoming data into a table, the same format our model expects
import joblib                       # loads our saved model and DataPreparer back from their files
import json                         # reads the small text file that tells us which model won
import os                           # helps us build file paths and check if files exist

from data_helper import DataPreparer, FEATURE_COLUMNS  # our own shared helper file (see data_helper.py)


# ----------------------------------------------------------------------
# 2. INPUT SHAPES (these describe exactly what a valid request looks
#    like -- FastAPI automatically checks every incoming request against
#    these, and sends back a clear error message if something is wrong,
#    like a missing field or text typed where a number was expected)
# ----------------------------------------------------------------------

class OneCrashRecord(BaseModel):
    """The information needed to predict Total_Crashes for ONE state/quarter."""
    # Every line below says: "this piece of information must be present,
    # and must be this type of value (text or whole number)."
    State: str = Field(..., example="Lagos")                                    # must be text, e.g. "Lagos"
    Year: int = Field(..., example=2023)                                         # must be a whole number, e.g. 2023
    Quarter_Num: int = Field(..., example=2, description="1, 2, 3, or 4")        # must be a whole number, 1 to 4
    SPV: int = Field(..., example=20, description="Speed Violation incidents")   # must be a whole number
    DAD: int = Field(..., example=5, description="Dangerous Driving incidents")  # must be a whole number
    PWR: int = Field(..., example=2, description="Poor Weather/Road incidents")  # must be a whole number
    FTQ: int = Field(..., example=1)                                              # must be a whole number
    Other_Factors: int = Field(..., example=10)                                   # must be a whole number
    # (the "..." means this field is REQUIRED -- the request will be rejected if it's missing)


class ManyCrashRecords(BaseModel):
    """A list of MULTIPLE records, for predicting many at once."""
    records: List[OneCrashRecord]   # a list where EVERY item must follow the OneCrashRecord shape above


# ----------------------------------------------------------------------
# 3. LOADING THE SAVED MODEL
#    This only runs ONCE, when the API first starts up -- not every
#    single time someone asks for a prediction, which keeps things fast.
# ----------------------------------------------------------------------

SAVE_FOLDER = "saved_model"   # the folder where train.py saved everything

model_file = os.path.join(SAVE_FOLDER, "trained_model.joblib")       # full path to the trained model file
preparer_file = os.path.join(SAVE_FOLDER, "data_preparer.joblib")     # full path to the saved DataPreparer file
info_file = os.path.join(SAVE_FOLDER, "model_info.json")               # full path to the small info file

if not (os.path.exists(model_file) and os.path.exists(preparer_file)):
    # ^ os.path.exists() checks whether a file is actually there on disk
    raise FileNotFoundError(
        "No trained model found. Please run `python train.py` first, "
        "then start this API again."
    )
    # ^ if either file is missing, stop immediately with a clear error message,
    #   instead of the app crashing later with a confusing error

trained_model = joblib.load(model_file)           # load the trained model itself back into memory
data_preparer = joblib.load(preparer_file)         # load the DataPreparer (it remembers the state numbers)

model_name_in_use = "Unknown Model"                 # a default value, in case the info file is missing
if os.path.exists(info_file):                        # only try to read it if the file actually exists
    with open(info_file) as f:                        # open the file for reading
        model_name_in_use = json.load(f).get("winner_model_name", "Unknown Model")
        # ^ read the JSON file, then grab the "winner_model_name" value from inside it


# ----------------------------------------------------------------------
# 4. THE ACTUAL PREDICTING FUNCTIONS
#    Kept separate from the web-address code below, so the logic is
#    easy to find and easy to test on its own.
# ----------------------------------------------------------------------

def predict_one(record_dict):
    """Turn ONE record (a dictionary of values) into ONE predicted number."""
    one_row_table = pd.DataFrame([record_dict])          # wrap the single record in a one-row table
    prepared_data = data_preparer.prepare(one_row_table)   # convert it into the number format the model needs
    prediction = trained_model.predict(prepared_data)[0]    # ask the model to guess; [0] grabs that one answer
    return float(prediction)                                 # convert to a plain number before sending it back


def predict_many(list_of_record_dicts):
    """Turn a LIST of records into a LIST of predicted numbers."""
    many_rows_table = pd.DataFrame(list_of_record_dicts)     # wrap the whole list into a multi-row table
    prepared_data = data_preparer.prepare(many_rows_table)     # convert ALL rows into the number format at once
    predictions = trained_model.predict(prepared_data)          # ask the model to guess for every row at once
    return [float(p) for p in predictions]                        # turn each answer into a plain number for the list


# ----------------------------------------------------------------------
# 5. THE WEB ADDRESSES (ROUTES)
#    Each function below answers one specific web address. FastAPI reads
#    the input shapes from Section 2 automatically.
# ----------------------------------------------------------------------

app = FastAPI(title="Nigerian Traffic Crashes - Prediction API")
# ^ this creates our actual API application; everything below "attaches" to it


@app.get("/")
# ^ @app.get(...) means: "when someone visits this web address using a normal
#   browser visit (a GET request), run the function right below this line"
def home():
    """A simple 'is it working?' check. Visit http://127.0.0.1:8000/ to see this."""
    return {   # FastAPI automatically turns this Python dictionary into JSON, the standard web-answer format
        "status": "The API is running!",
        "model_being_used": model_name_in_use,
        "try_these_addresses": ["/predict", "/predict/batch", "/docs"],
    }


@app.post("/predict")
# ^ @app.post(...) means: "when someone SENDS DATA to this address (a POST
#   request), run the function below, and check their data against OneCrashRecord"
def predict_single_record(record: OneCrashRecord):
    """Send ONE record here, get back ONE predicted number of crashes."""
    try:                                                    # "try" means: attempt this, and watch for errors
        prediction = predict_one(record.model_dump())
        # ^ record.model_dump() turns the validated request back into a plain Python dictionary
    except Exception as error:                              # if anything went wrong in the "try" block above...
        raise HTTPException(status_code=400, detail=f"Something went wrong: {error}")
        # ^ ...send back a clear error message instead of crashing the whole app

    return {
        "predicted_total_crashes": round(prediction, 2),    # round to 2 decimal places for a clean answer
        "model_used": model_name_in_use,
    }


@app.post("/predict/batch")
def predict_multiple_records(records: ManyCrashRecords):
    """Send a LIST of records here, get back a LIST of predicted numbers."""
    if len(records.records) == 0:                            # check: did they actually send any records at all?
        raise HTTPException(status_code=400, detail="You sent an empty list. Please include at least one record.")

    try:
        record_dicts = [r.model_dump() for r in records.records]
        # ^ go through every record in the list, and turn each one into a plain dictionary
        predictions = predict_many(record_dicts)               # get a prediction for every record, all at once
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Something went wrong: {error}")

    return {
        "predictions": [round(p, 2) for p in predictions],    # round every prediction to 2 decimal places
        "model_used": model_name_in_use,
        "how_many": len(predictions),                            # how many predictions we're sending back
    }
