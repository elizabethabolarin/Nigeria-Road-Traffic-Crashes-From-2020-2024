"""
data_helper.py
================
This file holds the data-cleaning steps that BOTH the training code
and the FastAPI app need to use. Keeping it in one place means the
same cleaning steps happen every time, whether we're training the
model or making a new prediction later.

Think of this file as a recipe card that both the "kitchen" (training)
and the "waiter" (the API that serves predictions) follow exactly the
same way.
"""

import pandas as pd                          # pandas lets us work with data as a table (rows and columns)
from sklearn.preprocessing import LabelEncoder  # a small tool that turns text (like state names) into numbers
import joblib   # joblib is just a tool for saving a Python object to a
                 # file on the computer, and loading it back later --
                 # similar to how you save a Word document and reopen it.


# These columns are RESULTS of a crash (you only know them AFTER a crash
# has already happened), so we must NOT use them to predict Total_Crashes.
# Using them would be like cheating on a test by looking at the answer key.
COLUMNS_TO_AVOID = ["Num_Injured", "Num_Killed", "Total_Vehicles_Involved"]

# The columns we WILL use to make predictions.
FEATURE_COLUMNS = ["State", "Year", "Quarter_Num", "SPV", "DAD", "PWR", "FTQ", "Other_Factors"]

# The column we are trying to predict.
TARGET_COLUMN = "Total_Crashes"


def load_data(file_path):
    """Read the CSV file from the computer into a table (DataFrame)."""
    return pd.read_csv(file_path)   # pd.read_csv opens the file and turns it into a pandas table


def clean_data(df):
    """
    Fix simple, obvious problems in the data:
    - Remove exact duplicate rows.
    - Fix impossible negative numbers in 'Other_Factors'
      (a crash count can never be negative).
    """
    df = df.copy()                    # make a fresh copy, so we don't accidentally change the original table
    df = df.drop_duplicates()         # remove any row that is an exact copy of another row

    if "Other_Factors" in df.columns:   # only do this if the column actually exists in the table
        # Any negative number becomes 0. Everything else stays the same.
        df["Other_Factors"] = df["Other_Factors"].clip(lower=0)   # .clip(lower=0) pushes any value below 0 up to 0

    return df   # send back the cleaned table


def add_time_columns(df):
    """
    The 'Quarter' column looks like "Q4 2020" (text + year together).
    This splits it into two separate NUMBER columns:
      - Quarter_Num  (e.g. 4)
      - Year         (e.g. 2020)
    Models work with numbers, not mixed text, so this makes the data usable.
    """
    df = df.copy()                              # again, work on a copy so the original table is untouched
    if "Quarter" in df.columns:                 # only run this if the 'Quarter' column exists
        pieces = df["Quarter"].str.split(" ", expand=True)   # split "Q4 2020" into two pieces: "Q4" and "2020"
        df["Quarter_Num"] = pieces[0].str.replace("Q", "", regex=False).astype(int)
        # ^ pieces[0] is "Q4" -> remove the letter "Q" -> "4" -> turn the text "4" into the real number 4
        df["Year"] = pieces[1].astype(int)       # pieces[1] is the text "2020" -> turn it into the real number 2020
    return df   # send back the table with the two new number columns added


class DataPreparer:
    """
    This class turns raw data into the exact number format the model
    needs. It has two simple jobs:

      1. LEARN  -- look at the training data once, and remember how to
                   turn state names (e.g. "Lagos") into numbers.
      2. PREPARE -- apply that same number-conversion to any new data
                    (training data, test data, or a brand new prediction
                    request), WITHOUT learning anything new from it.

    Why split it like this? If we let the model "peek" at new data while
    learning, it would get an unfair advantage and the results would be
    misleading. So we only ever LEARN once, from the training data.
    """

    def __init__(self, feature_columns=None, target_column=TARGET_COLUMN):
        # __init__ runs automatically whenever we create a new DataPreparer.
        # It just sets up the starting values this object will remember.
        self.feature_columns = feature_columns if feature_columns else list(FEATURE_COLUMNS)
        # ^ use the columns we were given, or fall back to the default list if none were given

        self.target_column = target_column       # remember which column we're trying to predict

        self.state_to_number = LabelEncoder()   # a simple tool that turns
                                                 # each state name into its
                                                 # own number, e.g. Lagos -> 12

        self.has_learned = False   # a flag (True/False switch) to track whether learn() has been called yet

    def learn(self, df):
        """Look at the TRAINING data once and remember each state's number."""
        df = add_time_columns(df)                       # make sure Quarter_Num/Year exist before we continue
        if "State" in self.feature_columns:              # only do this if 'State' is one of our chosen features
            self.state_to_number.fit(df["State"])         # .fit() studies the data and remembers each state's number
        self.has_learned = True                           # flip the flag to True, so prepare() is now allowed to run
        return self   # returning "self" lets us chain calls together if we want, e.g. DataPreparer().learn(df)

    def prepare(self, df):
        """
        Turn any data (old or new) into the number format the model needs,
        using what was already learned. Never learns anything new here.
        """
        if not self.has_learned:   # safety check: make sure learn() was called first
            raise RuntimeError("Call learn() on training data before calling prepare().")

        df = add_time_columns(df)     # again, make sure Quarter_Num/Year exist
        df = df.copy()                # work on a copy so we don't change the caller's original table

        if "State" in self.feature_columns:                        # only convert State if we're actually using it
            known_states = set(self.state_to_number.classes_)       # the list of state names the model already knows
            numbers = []                                              # we'll build up the converted numbers here, one by one
            for state_name in df["State"].tolist():                  # go through every state name in this data, one row at a time
                if state_name in known_states:                       # if we've seen this state name before...
                    numbers.append(int(self.state_to_number.transform([state_name])[0]))
                    # ^ convert this one state name into its number, and add it to our list
                else:
                    # A state name we've never seen before (e.g. a typo)
                    # gets the placeholder number -1 instead of crashing.
                    numbers.append(-1)
            df["State"] = numbers   # replace the text State column with our new list of numbers

        return df[self.feature_columns]   # return ONLY the columns the model actually needs, in the right order

    def learn_and_prepare(self, df):
        """Shortcut: learn from this data, then immediately prepare it (used on training data only)."""
        self.learn(df)                       # step 1: study this data and remember the state numbers
        X = self.prepare(df)                 # step 2: convert this same data into the number format
        y = df[self.target_column]           # also grab the target column (the real answers) separately
        return X, y   # X = the clues (inputs), y = the answers (what we're trying to predict)

    def save(self, file_path):
        """Save this whole object to a file, so we can reuse it later without re-learning."""
        joblib.dump(self, file_path)   # joblib.dump() writes this whole object out to a file on disk


def load_data_preparer(file_path):
    """Load a previously-saved DataPreparer back from a file."""
    return joblib.load(file_path)   # joblib.load() reads the file back and rebuilds the exact same object
