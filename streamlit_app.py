"""
streamlit_app.py
==================
This is our INTERACTIVE WEBSITE. It lets anyone -- even someone who
can't code -- click through the whole project: load the data, clean it,
look at charts, train models, and get predictions.

Everything lives in this ONE file, in 5 sections you can scroll through
in order:

    0. PAGE STYLING     -- colors, fonts, and the reusable "road divider"
    1. HOME PAGE
    2. PAGE 1: LOAD DATA
    3. PAGE 2: CLEAN DATA + CHARTS
    4. PAGE 3: TRAIN MODELS
    5. PAGE 4: MAKE A PREDICTION

To start this app, run this in your terminal:
    streamlit run streamlit_app.py
"""

import streamlit as st                  # the tool that turns our Python code into a clickable website
import pandas as pd                     # for working with data as tables
import numpy as np                      # for math tools, like square root
import matplotlib.pyplot as plt         # for drawing charts
import seaborn as sns                   # makes charts look nicer, sits on top of matplotlib
from sklearn.model_selection import train_test_split                     # splits data into random pieces
from sklearn.linear_model import LinearRegression                         # Model 1: straight-line model
from sklearn.ensemble import RandomForestRegressor                         # Model 2: many small trees averaged together
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # tools that score a model's accuracy

from data_helper import DataPreparer, add_time_columns, COLUMNS_TO_AVOID, TARGET_COLUMN
# ^ our own shared helper file -- same one used by train.py and fastapi_app.py

sns.set_theme(style="whitegrid")   # sets a clean, light background style for every chart in this app

st.set_page_config(page_title="Nigerian Traffic Crashes", page_icon="🚦", layout="wide")
# ^ this MUST be the first Streamlit command in the file. It sets the
#   browser tab's title/icon, and "wide" makes the page use the full screen width.


# ======================================================================
# 0. PAGE STYLING
#    A Nigerian-flag-inspired color theme (green / white / amber road-
#    sign yellow), plus one signature visual: a "road divider" bar that
#    appears at the top of every page, since this project is all about
#    ROADS. Everything below is just colors and fonts -- no data logic.
# ======================================================================

GREEN = "#008751"        # Nigeria flag green -- our main color
DEEP_GREEN = "#045C3B"    # a darker green, used for the sidebar background
AMBER = "#FFB100"         # road-sign amber/yellow -- our accent color
ASPHALT = "#2E2E2E"       # dark road-gray -- used for regular text
PAGE_BG = "#F6F5F0"       # warm off-white -- the main page background color
CARD_BG = "#FFFFFF"       # plain white -- used for small boxes like metrics

# The block below injects raw CSS (website styling code) into the page.
# We build it as one big piece of text using an f-string, so we can drop
# our color variables (GREEN, AMBER, etc.) straight into the styling.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');
/* ^ this line downloads two nicer-looking fonts from Google Fonts: "Poppins" for headings, "Inter" for regular text */

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;   /* use our regular font everywhere by default */
}}

/* Overall page background */
.stApp {{
    background-color: {PAGE_BG};   /* paints the whole page background our warm off-white color */
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {DEEP_GREEN};   /* paints the left-hand menu bar dark green */
}}
section[data-testid="stSidebar"] * {{
    color: #FFFFFF !important;    /* makes ALL text inside the sidebar white, so it's readable on dark green */
}}
section[data-testid="stSidebar"] .stRadio > label {{
    font-weight: 600;    /* makes the menu options in the sidebar a little bolder */
}}

/* Headings use the display font */
h1, h2, h3 {{
    font-family: 'Poppins', sans-serif !important;   /* use our bold heading font for all titles */
    color: {ASPHALT};
}}

/* Buttons */
.stButton > button {{
    background-color: {GREEN};    /* every button starts out green */
    color: white;
    border-radius: 8px;            /* rounds the corners of the button */
    border: none;
    font-weight: 600;
    padding: 0.5em 1.2em;
}}
.stButton > button:hover {{
    background-color: {AMBER};    /* when the mouse hovers over a button, it turns amber */
    color: {ASPHALT};
}}

/* Metric boxes */
[data-testid="stMetric"] {{
    background-color: {CARD_BG};    /* gives Streamlit's built-in metric boxes a white card look */
    border: 1px solid #E5E2D8;
    border-radius: 12px;
    padding: 12px;
}}

/* Our custom "road divider" -- a dark asphalt strip with a dashed
   amber center line, like the painted line down the middle of a road. */
.road-divider {{
    height: 14px;
    background-color: {ASPHALT};    /* the dark "asphalt" strip color */
    border-radius: 7px;
    margin: 6px 0 22px 0;             /* adds space above and below the bar */
    background-image: repeating-linear-gradient(
        to right, {AMBER} 0, {AMBER} 22px, transparent 22px, transparent 40px
    );
    /* ^ this draws repeating amber dashes across the bar, like a road's painted center line */
    background-position: center;
    background-repeat: repeat-x;
    background-size: 40px 4px;
}}

/* Section header pill */
.section-header {{
    display: inline-block;
    background-color: {GREEN};       /* green rounded "pill" background for section titles */
    color: white !important;
    padding: 6px 18px;
    border-radius: 999px;              /* a very large radius makes the corners fully round */
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    margin: 18px 0 10px 0;
}}
.section-header.amber {{
    background-color: {AMBER};        /* an alternate amber-colored version of the same pill */
    color: {ASPHALT} !important;
}}

/* Hero title */
.hero-title {{
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    background: linear-gradient(90deg, {GREEN}, {AMBER});   /* a green-to-amber color gradient */
    -webkit-background-clip: text;                             /* this trick "cuts" the gradient into the shape of the text */
    -webkit-text-fill-color: transparent;                       /* makes the text itself invisible, so only the gradient shows through */
    margin-bottom: 0px;
}}
.hero-subtitle {{
    font-family: 'Inter', sans-serif;
    color: {ASPHALT};
    font-size: 1.05rem;
    margin-top: 4px;
}}
</style>
""", unsafe_allow_html=True)
# ^ unsafe_allow_html=True tells Streamlit "yes, I really do want to inject
#   raw HTML/CSS here" -- normally Streamlit blocks this for safety.


def road_divider():
    """Draws our signature road-line bar (used at the top of every page)."""
    st.markdown('<div class="road-divider"></div>', unsafe_allow_html=True)
    # ^ this just inserts one empty <div> styled by the .road-divider CSS rule above


def page_title(title, subtitle):
    """Draws the big colorful title block at the top of a page."""
    st.markdown(f'<div class="hero-title">{title}</div>', unsafe_allow_html=True)      # the big gradient title text
    st.markdown(f'<div class="hero-subtitle">{subtitle}</div>', unsafe_allow_html=True)  # the smaller subtitle text below it
    road_divider()   # add our signature road-line bar underneath


def section_header(text, color="green"):
    """Draws a small colored pill-shaped header to introduce a section."""
    css_class = "section-header" if color == "green" else "section-header amber"
    # ^ pick which CSS style to use, based on the "color" we were given
    st.markdown(f'<span class="{css_class}">{text}</span>', unsafe_allow_html=True)


def is_word_column(df, column_name):
    """
    Checks whether a column holds WORDS (like state names) instead of
    NUMBERS. We use this instead of checking the column's "dtype" label
    directly, because different versions of pandas sometimes name that
    label differently for text columns -- this check works no matter
    which pandas version is running the app.
    """
    return not pd.api.types.is_numeric_dtype(df[column_name])
    # ^ pandas already has a simple built-in check for "is this a number
    #   column?" -- we just flip the answer with "not" to get "is this NOT
    #   a number column?", i.e. "does it hold words instead?"


# ======================================================================
# SIDEBAR NAVIGATION
# ======================================================================
st.sidebar.markdown("## 🚦 Navigate")   # a small heading at the top of the sidebar menu

page = st.sidebar.radio(
    "Go to:",                            # the (hidden) label for this set of radio buttons
    [                                      # the list of pages someone can choose from
        "🏠 Home",
        "📥 1. Load Data",
        "🧹 2. Clean Data & Charts",
        "🤖 3. Train Models",
        "🎯 4. Make a Prediction",
    ],
    label_visibility="collapsed",         # hides the "Go to:" label text, since our heading above already explains it
)
# `page` now holds whichever option the person clicked, e.g. "🏠 Home"

DATA_FILE = "data/Nigerian_Road_Traffic_Crashes_2020_2024.csv"   # the default data file location


# ======================================================================
# 1. HOME PAGE
# ======================================================================
if page == "🏠 Home":   # this whole block only runs if the person picked "Home" in the sidebar
    page_title("Nigerian Traffic Crashes", "A simple, guided walk from raw data to a working prediction tool.")

    col1, col2 = st.columns([2, 1])   # split the page into two side-by-side columns (left is twice as wide as right)
    with col1:                          # everything indented under here appears in the LEFT column
        st.markdown("""
        This app walks through the whole project in **4 easy steps**, one page at a time:

        1. **📥 Load Data** — bring in the crash data and take a first look at it.
        2. **🧹 Clean Data & Charts** — fix messy values and explore the data with charts.
        3. **🤖 Train Models** — teach two different models to predict crash numbers, and
           see which one guesses best.
        4. **🎯 Make a Prediction** — use the winning model to predict crashes for a state
           and quarter of your choice.

        Use the green menu on the left to move between pages, in order, the first time.
        """)
    with col2:                          # everything indented under here appears in the RIGHT column
        st.markdown("### 🚗 Quick Facts")
        st.info("Covers **Q4 2020 to Q1 2024** across Nigerian states.")           # a light-blue info box
        st.info("Goal: predict **Total_Crashes** using recorded contributing factors.")
        st.info("Two models are compared automatically — no guessing needed.")


# ======================================================================
# 2. PAGE 1: LOAD DATA
# ======================================================================
elif page == "📥 1. Load Data":   # only runs if the person picked this page
    page_title("Step 1: Load the Data", "Bring the crash data into the app and take a first look.")

    section_header("📂 Choose a File")
    uploaded_file = st.file_uploader("Upload your own CSV (optional) — or skip this to use the default project data:", type="csv")
    # ^ shows a "browse files" button; uploaded_file will be None if nobody uploads anything

    if uploaded_file is not None:              # did the person actually upload a file?
        df = pd.read_csv(uploaded_file)          # yes -> read THEIR file into a table
        st.success("Your uploaded file was loaded.")   # a green success message
    else:
        df = pd.read_csv(DATA_FILE)               # no -> fall back to reading our default project file
        st.info(f"Using the default file: {DATA_FILE}")

    st.session_state["raw_data"] = df
    # ^ st.session_state is Streamlit's "memory" that survives between page switches and clicks.
    #   We save the data here so the OTHER pages (Clean Data, Train Models, etc.) can use it too.

    section_header("📏 Size of the Data")
    c1, c2 = st.columns(2)                       # two equal-width side-by-side columns
    c1.metric("Number of Rows", df.shape[0])      # df.shape[0] = how many rows the table has
    c2.metric("Number of Columns", df.shape[1])    # df.shape[1] = how many columns the table has

    section_header("👀 First Few Rows")
    st.dataframe(df.head(10), use_container_width=True)   # show the first 10 rows as a scrollable table

    section_header("🏷️ Column Names", color="amber")
    st.write(list(df.columns))                     # just print out the plain list of column names

    section_header("🔢 Column Types")
    st.dataframe(df.dtypes.astype(str).rename("Type"), use_container_width=True)
    # ^ df.dtypes tells us whether each column holds numbers, text, etc.

    section_header("❓ Any Missing Values?", color="amber")
    missing = df.isnull().sum()                     # count how many empty cells are in each column
    st.dataframe(missing.rename("Missing Count"), use_container_width=True)
    if missing.sum() == 0:                            # if the TOTAL across all columns is 0...
        st.success("Good news — no missing values anywhere in this data.")

    section_header("👯 Any Duplicate Rows?")
    st.write(f"Exact duplicate rows found: **{df.duplicated().sum()}**")
    # ^ df.duplicated() marks True for any row that's an exact copy of an earlier one; .sum() counts them

    section_header("📊 Summary Numbers", color="amber")
    st.dataframe(df.describe(), use_container_width=True)
    # ^ df.describe() gives min, max, average, etc. for every number column, all at once


# ======================================================================
# 3. PAGE 2: CLEAN DATA + CHARTS
# ======================================================================
elif page == "🧹 2. Clean Data & Charts":
    page_title("Step 2: Clean the Data & Explore It", "Fix messy values, then look for patterns with charts.")

    if "raw_data" not in st.session_state:          # did the person visit Page 1 yet? (that's where raw_data gets saved)
        st.warning("Please visit '📥 1. Load Data' first.")
        st.stop()                                      # st.stop() halts the rest of this page from running

    df = st.session_state["raw_data"].copy()          # grab the data saved from Page 1, and work on a copy of it
    df = add_time_columns(df)                          # turns "Q4 2020" into Quarter_Num=4, Year=2020

    section_header("🧽 Step A: Remove Columns You Don't Want (optional)")
    columns_to_drop = st.multiselect("Pick any columns to remove:", options=list(df.columns), default=[])
    # ^ multiselect shows a box where the person can pick zero, one, or many column names
    if columns_to_drop:                                 # if they picked at least one column...
        df = df.drop(columns=columns_to_drop)             # ...remove those columns from the table

    section_header("🩹 Step B: Fix Missing Values")
    total_missing = df.isnull().sum().sum()               # add up missing cells across ALL columns into one number
    st.write(f"Missing values currently in the data: **{total_missing}**")
    fix_method = st.selectbox(                              # a drop-down box with one choice at a time
        "How should we handle them?",
        ["Nothing to fix", "Remove rows with missing values", "Fill with the average", "Fill with the middle value", "Fill with 0"],
    )
    if total_missing > 0 and fix_method != "Nothing to fix":   # only bother fixing anything if there IS something to fix
        number_cols = df.select_dtypes(include="number").columns   # grab just the columns that hold numbers
        if fix_method == "Remove rows with missing values":
            df = df.dropna()                                          # delete any row that has an empty cell
        elif fix_method == "Fill with the average":
            df[number_cols] = df[number_cols].fillna(df[number_cols].mean())   # replace empty cells with that column's average
        elif fix_method == "Fill with the middle value":
            df[number_cols] = df[number_cols].fillna(df[number_cols].median())  # replace empty cells with that column's middle value
        elif fix_method == "Fill with 0":
            df[number_cols] = df[number_cols].fillna(0)                          # replace empty cells with plain 0
        st.success(f"Done. Missing values remaining: {df.isnull().sum().sum()}")

    section_header("🎯 Step C: Pick What to Predict (the Target)", color="amber")
    number_columns = df.select_dtypes(include="number").columns.tolist()   # a list of all number columns
    suggested = TARGET_COLUMN if TARGET_COLUMN in number_columns else number_columns[0]
    # ^ suggest Total_Crashes by default, unless it's somehow not available, then just suggest the first number column

    target = st.selectbox("What should the model try to predict?", options=number_columns, index=number_columns.index(suggested))
    # ^ index=... tells the drop-down which option to show as already-selected when the page first loads

    st.caption(f"We suggest **{suggested}** — the main number this whole project is about.")

    section_header("🧩 Step D: Pick the Clues (Features)", color="amber")
    possible_features = [c for c in df.columns if c != target]   # every column EXCEPT whichever one is the target
    leaky = [c for c in COLUMNS_TO_AVOID if c in possible_features]   # check if any "don't use these" columns are still present
    if leaky:
        st.warning(f"⚠️ Careful: {leaky} are only known AFTER a crash happens. Using them as clues would be 'cheating'.")
    good_defaults = [c for c in possible_features if c not in COLUMNS_TO_AVOID and c != "Quarter"]
    # ^ our suggested starting selection: everything except the "avoid" columns and the raw text Quarter column

    features = st.multiselect("Which columns should the model use as clues?", options=possible_features, default=good_defaults)

    if target in features:                              # safety check: the target can't ALSO be used as a clue
        st.error("The target can't also be a clue. Please remove it from the clues list.")
        st.stop()
    if not features:                                       # safety check: they need to pick at least one clue
        st.warning("Pick at least one clue (feature) to continue.")
        st.stop()

    st.session_state["clean_data"] = df                  # save the cleaned table for the other pages to use
    st.session_state["target"] = target                    # save which column was chosen as the target
    st.session_state["features"] = features                  # save which columns were chosen as clues

    section_header("👀 Cleaned Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    section_header("📈 Step E: Explore With Charts", color="amber")
    chart_choice = st.selectbox(
        "Pick a chart:",
        ["Distribution (Histogram)", "Box Plot", "Compare Two Numbers (Scatter)", "Correlation Grid", "Bar Chart by Group"],
    )

    fig = None   # we'll only fill this in if a chart actually gets drawn below (used later for the download button)

    if chart_choice == "Distribution (Histogram)":
        col = st.selectbox("Which column?", number_columns, index=number_columns.index(target))
        fig, ax = plt.subplots(figsize=(8, 4))                    # start a new blank chart canvas
        sns.histplot(df[col], kde=True, ax=ax, color=GREEN)         # draw a histogram (bars showing how values are spread out)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)                                                # actually display the chart on the page
        st.caption(f"**What you're seeing:** how {col} values are spread out. A long tail to the right means most rows are low, with a few very high ones pulling the average up.")

    elif chart_choice == "Box Plot":
        col = st.selectbox("Number column:", number_columns, index=number_columns.index(target))
        group_options = [c for c in df.columns if is_word_column(df, c) or df[c].nunique() < 20]
        # ^ only offer to group by columns that are text, or that don't have too many different values
        group = st.selectbox("Group by (optional):", ["None"] + group_options)
        fig, ax = plt.subplots(figsize=(10, 5))
        if group != "None":                                            # did they choose a column to group by?
            sns.boxplot(data=df, x=group, y=col, ax=ax, color=AMBER)      # one box per group
            plt.xticks(rotation=90 if df[group].nunique() > 8 else 0)     # tilt the labels if there are many groups
        else:
            sns.boxplot(y=df[col], ax=ax, color=AMBER)                    # just one single box, no grouping
        ax.set_title(f"Box Plot of {col}" + (f" by {group}" if group != "None" else ""))
        st.pyplot(fig)
        st.caption("**What you're seeing:** the box covers the middle half of the values; dots above/below are unusually high/low rows (outliers).")

    elif chart_choice == "Compare Two Numbers (Scatter)":
        x_col = st.selectbox("Compare this...", [c for c in number_columns if c != target])
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=x_col, y=target, alpha=0.6, ax=ax, color=GREEN)
        # ^ draws one dot per row, positioned by its x_col value and its target value; alpha makes dots see-through
        ax.set_title(f"{x_col} vs {target}")
        st.pyplot(fig)
        corr_value = df[[x_col, target]].corr().iloc[0, 1]    # calculate how strongly these two columns are related
        st.caption(f"**What you're seeing:** as {x_col} changes, {target} tends to move in a similar/opposite direction. Relationship strength (correlation): **{corr_value:.2f}**.")

    elif chart_choice == "Correlation Grid":
        chosen_cols = st.multiselect("Columns to include:", number_columns, default=number_columns)
        if len(chosen_cols) >= 2:                                # need at least 2 columns to compare anything
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[chosen_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            # ^ draws a color-coded grid; annot=True writes the actual number inside each square
            ax.set_title("How Strongly Are Columns Related?")
            st.pyplot(fig)
            st.caption("**What you're seeing:** numbers close to +1 or -1 mean a strong relationship. Close to 0 means little to no relationship.")

    elif chart_choice == "Bar Chart by Group":
        cat_options = [c for c in df.columns if is_word_column(df, c) or df[c].nunique() < 40]
        cat_col = st.selectbox("Group by:", cat_options)
        agg = st.radio("Show the:", ["average", "total"], horizontal=True)   # let them pick average or sum
        grouped = df.groupby(cat_col)[target].agg("mean" if agg == "average" else "sum").sort_values(ascending=False)
        # ^ groups the data by cat_col, then averages or sums the target within each group, sorted highest first
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=grouped.index, y=grouped.values, ax=ax, color=GREEN)
        plt.xticks(rotation=90)
        ax.set_title(f"{agg.title()} {target} by {cat_col}")
        st.pyplot(fig)
        st.caption(f"**What you're seeing:** which {cat_col} groups have the highest/lowest {agg} {target}.")

    if fig is not None:                                          # only show a download button if a chart was actually drawn
        import io
        buf = io.BytesIO()                                          # a temporary in-memory "file"
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")  # save the chart image into that temporary file
        st.download_button("📥 Save this chart as a picture", data=buf.getvalue(), file_name="chart.png", mime="image/png")


# ======================================================================
# 4. PAGE 3: TRAIN MODELS
# ======================================================================
elif page == "🤖 3. Train Models":
    page_title("Step 3: Teach Two Models & Pick the Best One", "The app automatically picks whichever model guesses best.")

    if "clean_data" not in st.session_state:                # make sure Page 2 was completed first
        st.warning("Please finish '🧹 2. Clean Data & Charts' first.")
        st.stop()

    df = st.session_state["clean_data"]                       # grab the cleaned data saved from Page 2
    target = st.session_state["target"]                         # grab the chosen target column
    features = st.session_state["features"]                       # grab the chosen clue columns

    section_header("🎯 What We're Predicting")
    c1, c2 = st.columns(2)
    c1.write("**Target (what to predict):**")
    c1.code(target)                                                # show it in a little code-style box
    c2.write("**Clues (what the model looks at):**")
    c2.code(", ".join(features))                                    # join the list into one comma-separated string

    if st.button("🚀 Teach the Models Now", type="primary"):     # a clickable button; the code below only runs when clicked
        with st.spinner("Splitting the data and teaching two models... this only takes a moment."):
            # ^ st.spinner shows a little "loading" animation while the code inside this block runs

            train_val, test_data = train_test_split(df, test_size=0.2, random_state=42)     # first cut off 20% as test data
            train_data, val_data = train_test_split(train_val, test_size=0.25, random_state=42)
            # ^ then split the remaining 80% into 60% training / 20% validation

            preparer = DataPreparer(feature_columns=features, target_column=target)
            X_train, y_train = preparer.learn_and_prepare(train_data)     # learn from training data, then convert it
            X_val, y_val = preparer.prepare(val_data), val_data[target]    # convert validation data (no new learning)

            candidates = {                                                    # our two models to compare
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            }

            scores = {}       # will hold each model's validation scores, by name
            trained = {}       # will hold each actual trained model object, by name
            for name, model in candidates.items():                            # go through both models, one at a time
                model.fit(X_train, y_train)                                     # teach this model using the training data
                guesses = model.predict(X_val)                                   # ask it to guess on the validation data
                scores[name] = {                                                   # calculate and store its 4 accuracy numbers
                    "MAE": mean_absolute_error(y_val, guesses),
                    "MSE": mean_squared_error(y_val, guesses),
                    "RMSE": np.sqrt(mean_squared_error(y_val, guesses)),
                    "R2": r2_score(y_val, guesses),
                }
                trained[name] = model                                              # remember the trained model itself too

            winner_name = min(scores, key=lambda n: scores[n]["RMSE"])
            # ^ pick whichever model name has the smallest RMSE value

            st.session_state["scores"] = scores                     # save results so they're still visible after this block ends
            st.session_state["winner_name"] = winner_name
            st.session_state["winner_model"] = trained[winner_name]
            st.session_state["preparer"] = preparer

        st.success("Done teaching both models!")

    if "scores" in st.session_state:                             # only show results if training has actually happened
        section_header("📋 How Each Model Did (on data it wasn't taught with)", color="amber")
        scores_table = pd.DataFrame(st.session_state["scores"]).T.round(3)   # turn the scores dictionary into a neat table
        st.dataframe(scores_table, use_container_width=True)

        winner = st.session_state["winner_name"]
        st.markdown(f"""
        <div style="background-color:{GREEN}; color:white; padding:16px; border-radius:12px; font-family:'Poppins',sans-serif; font-weight:700; font-size:1.2rem;">
        🏆 Automatically chosen winner: {winner}
        </div>
        """, unsafe_allow_html=True)
        # ^ a custom-styled green banner announcing the winning model
        st.caption("Smaller MAE/RMSE = smaller mistakes = better. Bigger R2 (closer to 1) = better. The app always picks whichever model has the smallest RMSE — nothing is chosen by hand.")
        st.info("Go to '🎯 4. Make a Prediction' to try this model out.")


# ======================================================================
# 5. PAGE 4: MAKE A PREDICTION
# ======================================================================
elif page == "🎯 4. Make a Prediction":
    page_title("Step 4: Try the Model", "Enter some numbers and see what the model predicts.")

    if "winner_model" not in st.session_state:                # make sure Page 3 (training) was completed first
        st.warning("Please train a model on '🤖 3. Train Models' first.")
        st.stop()

    model = st.session_state["winner_model"]                    # grab the winning trained model
    preparer = st.session_state["preparer"]                       # grab the DataPreparer that goes with it
    features = st.session_state["features"]                        # grab the list of clue columns
    df_ref = st.session_state["clean_data"]                          # grab the cleaned data, just to look up sensible default values

    st.info(f"Using the model that won earlier: **{st.session_state['winner_name']}**")

    tab1, tab2 = st.tabs(["🧍 One Prediction", "📄 Many at Once (Upload a File)"])
    # ^ creates two clickable tabs on the page; everything below is grouped under one tab or the other

    with tab1:                                                    # ---- everything here appears under Tab 1 ----
        section_header("✍️ Fill In the Details")
        input_values = {}                                            # we'll collect the person's answers here, one per clue
        cols = st.columns(2)                                          # two side-by-side columns, to fit more inputs on screen
        for i, feature in enumerate(features):                         # go through each clue column, one at a time
            col = cols[i % 2]                                            # alternate between the left and right column
            if feature in df_ref.columns and is_word_column(df_ref, feature):
                # ^ if this clue is a text column (like State)...
                choices = sorted(df_ref[feature].dropna().unique().tolist())   # get a sorted list of all its real values
                input_values[feature] = col.selectbox(feature, choices)          # ...show a drop-down list to pick from
            else:
                # ...otherwise it's a number column, so show a number-entry box
                default_value = float(df_ref[feature].median()) if feature in df_ref.columns else 0.0
                # ^ pre-fill the box with that column's typical (middle) value, as a sensible starting point
                input_values[feature] = col.number_input(feature, value=default_value)

        if st.button("🔮 Predict", type="primary"):                  # only runs the code below when this button is clicked
            one_row = pd.DataFrame([input_values])                     # wrap the person's answers into a one-row table
            prepared = preparer.prepare(one_row)                        # convert it into the number format the model needs
            prediction = model.predict(prepared)[0]                      # ask the model to guess; [0] grabs that one answer
            st.markdown(f"""
            <div style="background-color:{AMBER}; padding:20px; border-radius:14px; text-align:center;">
                <div style="font-family:'Inter',sans-serif; font-size:1rem; color:{ASPHALT};">Predicted Total Crashes</div>
                <div style="font-family:'Poppins',sans-serif; font-size:2.6rem; font-weight:800; color:{ASPHALT};">{prediction:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
            # ^ a big, bold amber card showing the final predicted number

    with tab2:                                                    # ---- everything here appears under Tab 2 ----
        section_header("📄 Upload a File With Many Rows", color="amber")
        st.write(f"Your file needs these columns: `{', '.join(features)}`")
        batch_file = st.file_uploader("Upload a CSV", type="csv", key="batch")
        # ^ key="batch" keeps this uploader separate from any other file_uploader on the page

        if batch_file is not None:                                  # did they actually upload something?
            batch_df = pd.read_csv(batch_file)                         # read their file into a table
            missing_cols = [f for f in features if f not in batch_df.columns]
            # ^ check whether any of the columns the model needs are missing from their file
            if missing_cols:
                st.error(f"Your file is missing these needed columns: {missing_cols}")
            else:
                prepared_batch = preparer.prepare(batch_df)              # convert their whole file into the number format
                predictions = model.predict(prepared_batch)                # get a prediction for every row at once
                result_df = batch_df.copy()                                 # start from a copy of their original file
                result_df["Predicted_Total_Crashes"] = predictions.round(1)   # add a new column with our predictions

                st.dataframe(result_df, use_container_width=True)           # show the results as a table on screen

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                # ^ turn the results table into plain CSV text, ready to be downloaded as a file
                st.download_button("📥 Download These Predictions", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
