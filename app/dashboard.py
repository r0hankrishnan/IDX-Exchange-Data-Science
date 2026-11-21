"""Home price explorer: look up a sold home, see what the model knows about it, and estimate its price.

    uv run streamlit run app/dashboard.py

Each estimate comes from a model trained on the 18 months before the home's close month,
the same setup as the validation backtests, using only what is known before listing.
"""

from __future__ import annotations

import html

import altair as alt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBRegressor

from idx_exchange_data_science import dataset_a as da
from idx_exchange_data_science import features as ft
from idx_exchange_data_science.modeling import build_model, random_forest
from idx_exchange_data_science.splits import split

X = 18
MIN_TRAIN_MONTHS = 3
TEST_MONTH = pd.Period("2026-04", freq="M")
MAX_OPTIONS = 200
NUMERIC = ft.DISTRICT_NUMERIC_COLS + ft.COMP_COLS
CATEGORICAL = ft.DISTRICT_COLS
BLUE, ORANGE = "#2a78d6", "#eb6834"

MODELS = {
    # Settings from the random search in 06_advanced_models.ipynb (tuning month 2025-09).
    "XGBoost (tuned)": lambda: XGBRegressor(
        tree_method="hist", n_estimators=600, learning_rate=0.092, max_depth=8, min_child_weight=1,
        subsample=1.0, colsample_bytree=0.5, n_jobs=-1, random_state=0),
    "Random forest": random_forest,
}
MODEL_NOTES = {
    "XGBoost (tuned)": "Best on the 2026-03 validation month (R² 0.914). Fits in a few seconds.",
    "Random forest": "Best on the full 6-month validation (05e). Slower: about 20 seconds per fit.",
}

LABELS = {
    "LivingArea": "Living area", "PropertyAge": "Age", "LotSizeSquareFeet": "Lot size",
    "DistrictSEDPct": "District disadvantaged share", "LogCompPricePerSqft": "Nearby $ per sq ft",
    "LogCompImpliedPrice": "Comparable-sales price", "CompDistanceKm": "Distance to comparables",
    "BedroomsTotal": "Bedrooms", "BathroomsTotalInteger": "Bathrooms", "GarageSpaces": "Garage spaces",
    "ParkingTotal": "Parking spaces", "AttachedGarageYN": "Attached garage", "FireplaceYN": "Fireplace",
    "NewConstructionYN": "New construction", "PoolPrivateYN": "Pool", "ViewYN": "View",
    "Latitude": "Latitude", "Longitude": "Longitude", "City": "City", "CountyOrParish": "County",
    "PostalCode": "ZIP code", "ElementaryDistrict": "Elementary district", "HighDistrict": "High school district",
}
FLAGS = {"PoolPrivateYN": "Pool", "ViewYN": "View", "FireplaceYN": "Fireplace",
         "AttachedGarageYN": "Attached garage", "NewConstructionYN": "New construction"}
FLAG_VALUES = {"Yes": 1.0, "No": 0.0, "Unknown": np.nan}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.home-title {font-size: 2rem; font-weight: 700; margin: 0;}
.home-subtitle {opacity: .65; margin: 0 0 1rem;}
.fact-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 16px; margin-bottom: 1rem;}
.fact-card h4 {margin: 0 0 6px; padding: 0; font-size: .75rem; text-transform: uppercase;
               letter-spacing: .06em; opacity: .6;}
.fact-row {display: flex; justify-content: space-between; gap: 12px; padding: 5px 0;
           border-bottom: 1px solid rgba(128,128,128,.12); font-size: .92rem;}
.fact-row:last-child {border-bottom: none;}
.fact-row span:first-child {opacity: .7;}
.fact-row span:last-child {text-align: right; font-weight: 500;}
.card-note {font-size: .78rem; opacity: .55; margin-top: 6px;}
</style>
"""


# --- formatting -------------------------------------------------------------------

def money(x) -> str:
    if pd.isna(x):
        return "—"
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}K"


def number(x, unit: str = "") -> str:
    return "—" if pd.isna(x) else f"{x:,.0f}{unit}"


def yes_no(x) -> str:
    return {1.0: "Yes", 0.0: "No"}.get(x, "Unknown")


def month_name(period: pd.Period) -> str:
    return period.strftime("%b %Y")


def fact_card(title: str, rows: dict[str, str], note: str = "") -> str:
    body = "".join(f"<div class='fact-row'><span>{html.escape(k)}</span><span>{html.escape(str(v))}</span></div>"
                   for k, v in rows.items())
    note = f"<div class='card-note'>{html.escape(note)}</div>" if note else ""
    return f"<div class='fact-card'><h4>{html.escape(title)}</h4>{body}{note}</div>"


def feature_label(name: str) -> str:
    if name.startswith("missingindicator_"):
        return LABELS[name.removeprefix("missingindicator_")] + " unknown"
    return LABELS.get(name, name)


# --- data and models --------------------------------------------------------------

@st.cache_resource(show_spinner="Building the feature table (about 15 seconds, once per session)...")
def load_homes() -> pd.DataFrame:
    """The modeling table plus address and listing details for display (not model inputs)."""
    df = ft.build_table()
    extra = pd.read_csv(ft.FILTERED_PATH, usecols=["ListingKey", "UnparsedAddress", "ListPrice", "DaysOnMarket"])
    df = df.merge(extra.drop_duplicates("ListingKey"), on="ListingKey", how="left", validate="many_to_one")
    df["Address"] = df["UnparsedAddress"].fillna("Address unknown").str.strip()
    df["SearchText"] = (df["Address"] + " " + df["City"].fillna("") + " " + df["PostalCode"].fillna("")).str.lower()
    df["CloseMonth"] = df[da.DATE_COL].dt.to_period("M")
    return df


@st.cache_resource(show_spinner=False, max_entries=8)
def fit_for_month(model_name: str, month: str, train_months: int, _df: pd.DataFrame):
    """The model trained on the `train_months` before `month`, and its predictions for that month's sales."""
    train, month_sales = split(_df, month, train_months)
    model = build_model(MODELS[model_name](), da.make_preprocessor(NUMERIC, CATEGORICAL))
    model.fit(train, train[da.TARGET])
    predicted = pd.Series(model.predict(month_sales), index=month_sales.index)
    return model, month_sales[da.TARGET], predicted, train[da.DATE_COL].agg(["min", "max", "size"])


@st.cache_resource(show_spinner=False, max_entries=8)
def explainer_for(model_name: str, month: str, train_months: int, _model) -> shap.TreeExplainer:
    return shap.TreeExplainer(_model[-1])


@st.cache_data(show_spinner=False, max_entries=50)
def comparables_for(index, _df: pd.DataFrame) -> pd.DataFrame:
    return ft.find_comparables(_df, index)


def search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Sales whose address, city or ZIP contain every word of the query, most recent first."""
    mask = pd.Series(True, index=df.index)
    for word in query.lower().split():
        mask &= df["SearchText"].str.contains(word, regex=False)
    return df[mask].sort_values(da.DATE_COL, ascending=False)


def option_label(home: pd.Series) -> str:
    return (f"{home['Address']}, {home['City']} {home['PostalCode']}  ·  "
            f"sold {month_name(home['CloseMonth'])} for {money(home[da.TARGET])}")


# --- page sections ----------------------------------------------------------------

def header(home: pd.Series) -> None:
    county = None if pd.isna(home["CountyOrParish"]) else f"{home['CountyOrParish']} County"
    place = ", ".join(str(p) for p in [home["City"], county, home["PostalCode"]] if pd.notna(p))
    st.markdown(f"<p class='home-title'>{html.escape(home['Address'])}</p>"
                f"<p class='home-subtitle'>{html.escape(place)}</p>", unsafe_allow_html=True)
    sold, listed = home[da.TARGET], home["ListPrice"]
    cols = st.columns(4)
    cols[0].metric("Sold for", money(sold))
    cols[1].metric("Closed", home[da.DATE_COL].strftime("%b %d, %Y"))
    cols[2].metric("List price", money(listed),
                   delta=None if pd.isna(listed) else f"{sold / listed - 1:+.1%} sold vs list", delta_color="off")
    cols[3].metric("Days on market", number(home["DaysOnMarket"]))


def details_tab(home: pd.Series) -> None:
    close_year = home[da.DATE_COL].year
    left, middle, right = st.columns(3)
    left.markdown(fact_card("Size and layout", {
        "Living area": number(home["LivingArea"], " sq ft"),
        "Lot size": number(home["LotSizeSquareFeet"], " sq ft"),
        "Bedrooms": number(home["BedroomsTotal"]),
        "Bathrooms": number(home["BathroomsTotalInteger"]),
        "Year built": number(close_year - home["PropertyAge"]),
    }), unsafe_allow_html=True)
    left.markdown(fact_card("Parking", {
        "Garage spaces": number(home["GarageSpaces"]),
        "Parking spaces": number(home["ParkingTotal"]),
        "Attached garage": yes_no(home["AttachedGarageYN"]),
    }), unsafe_allow_html=True)
    middle.markdown(fact_card("Features", {label: yes_no(home[col]) for col, label in FLAGS.items()
                                           if col != "AttachedGarageYN"}), unsafe_allow_html=True)
    middle.markdown(fact_card("Market at listing", {
        "Nearby $ per sq ft": "—" if pd.isna(home["LogCompPricePerSqft"])
        else f"${np.exp(home['LogCompPricePerSqft']):,.0f}",
        "Comparable-sales price": money(np.exp(home["LogCompImpliedPrice"])),
        "Avg. distance to comparables": "—" if pd.isna(home["CompDistanceKm"]) else f"{home['CompDistanceKm']:.1f} km",
    }, note="From the 10 nearest sales in the 12 months before listing."), unsafe_allow_html=True)
    right.markdown(fact_card("Location", {
        "City": home["City"] if pd.notna(home["City"]) else "—",
        "ZIP code": home["PostalCode"] if pd.notna(home["PostalCode"]) else "—",
        "County": home["CountyOrParish"] if pd.notna(home["CountyOrParish"]) else "—",
        "Elementary district": home["ElementaryDistrict"] if pd.notna(home["ElementaryDistrict"]) else "—",
        "High school district": home["HighDistrict"] if pd.notna(home["HighDistrict"]) else "—",
        "Disadvantaged students": "—" if pd.isna(home["DistrictSEDPct"]) else f"{home['DistrictSEDPct']:.0f}%",
    }), unsafe_allow_html=True)
    listed = home[ft.LISTING_DATE]
    right.markdown(fact_card("Listing", {
        "Listed": "Unknown" if pd.isna(listed) else listed.strftime("%b %d, %Y"),
        "List price": money(home["ListPrice"]),
        "Days on market": number(home["DaysOnMarket"]),
    }, note="Shown for context. The model doesn't use these: it predicts before listing."), unsafe_allow_html=True)


def comparables_tab(df: pd.DataFrame, index) -> None:
    home = df.loc[index]
    comps = comparables_for(index, df)
    if comps.empty:
        st.info("No comparable sales: the home has no coordinates, or there are too few earlier sales.")
        return

    median_ppsf = comps["PricePerSqft"].median()
    st.markdown(f"The 10 nearest homes that sold in the 12 months before this one was listed. "
                f"Their median price per square foot is **${median_ppsf:,.0f}**. Multiplied by this home's "
                f"{number(home['LivingArea'])} sq ft, that gives a comparable-sales price of "
                f"**{money(median_ppsf * home['LivingArea'])}** (sold for {money(home[da.TARGET])}).")

    st.dataframe(
        comps.assign(Sold=comps[da.DATE_COL].dt.date)[
            ["Address", "City", "Sold", da.TARGET, "LivingArea", "PricePerSqft", "DistanceKm"]],
        hide_index=True, width="stretch",
        column_config={
            da.TARGET: st.column_config.NumberColumn("Price", format="$%,.0f"),
            "LivingArea": st.column_config.NumberColumn("Sq ft", format="%,.0f"),
            "PricePerSqft": st.column_config.NumberColumn("$ per sq ft", format="$%,.0f"),
            "DistanceKm": st.column_config.NumberColumn("Distance", format="%.2f km"),
        },
    )


def edited_home(df: pd.DataFrame, index) -> pd.DataFrame:
    """The home's model inputs, with any changes from the form. Returns a one-row frame."""
    home = df.loc[index]
    row = df.loc[[index]].copy()
    close_year = home[da.DATE_COL].year

    def as_int(x):
        return None if pd.isna(x) else int(round(x))

    with st.form(f"edit_{index}"):
        st.markdown("**Adjust the home** to see how the estimate changes. Blank means unknown.")
        c1, c2, c3, c4 = st.columns(4)
        living = c1.number_input("Living area (sq ft)", min_value=1, step=50, value=as_int(home["LivingArea"]))
        lot = c2.number_input("Lot size (sq ft)", min_value=0, step=500, value=as_int(home["LotSizeSquareFeet"]))
        beds = c3.number_input("Bedrooms", min_value=1, max_value=20, step=1, value=as_int(home["BedroomsTotal"]))
        baths = c4.number_input("Bathrooms", min_value=1, max_value=20, step=1,
                                value=as_int(home["BathroomsTotalInteger"]))
        c1, c2, c3, c4 = st.columns(4)
        built = c1.number_input("Year built", step=1, value=as_int(close_year - home["PropertyAge"]))
        garage = c2.number_input("Garage spaces", min_value=0, max_value=20, step=1, value=as_int(home["GarageSpaces"]))
        parking = c3.number_input("Parking spaces", min_value=0, max_value=20, step=1,
                                  value=as_int(home["ParkingTotal"]))
        flag_cols = st.columns(len(FLAGS))
        flags = {col: flag_cols[i].selectbox(label, list(FLAG_VALUES), index=list(FLAG_VALUES).index(yes_no(home[col])))
                 for i, (col, label) in enumerate(FLAGS.items())}
        st.form_submit_button("Update estimate", type="primary")

    numbers = {"LivingArea": living, "LotSizeSquareFeet": lot, "BedroomsTotal": beds, "BathroomsTotalInteger": baths,
               "PropertyAge": None if built is None else close_year - built, "GarageSpaces": garage,
               "ParkingTotal": parking}
    # Only overwrite what changed, so the form's rounding doesn't alter an untouched home.
    for col, value in numbers.items():
        if value != as_int(home[col]):
            row[col] = np.nan if value is None else float(value)
    for col, choice in flags.items():
        row[col] = FLAG_VALUES[choice]
    # The comparable-sales price is the neighbours' $/sq ft times this home's living area.
    row["LogCompImpliedPrice"] = row["LogCompPricePerSqft"] + np.log(row["LivingArea"])
    return row


def estimate_tab(df: pd.DataFrame, index, model_name: str, allow_test: bool) -> None:
    home = df.loc[index]
    month = home["CloseMonth"]
    if month == TEST_MONTH and not allow_test:
        st.info(f"This home sold in {month_name(TEST_MONTH)}, the held-out test month. Estimates for it are off "
                "so the final test stays untouched. Turn them on in the sidebar if you need them.")
        return
    train_months = min(X, (month - df["CloseMonth"].min()).n)
    if train_months < MIN_TRAIN_MONTHS:
        st.info(f"Too little earlier data to train on: the data starts in {month_name(df['CloseMonth'].min())}.")
        return

    with st.spinner(f"Training {model_name} on the {train_months} months before {month_name(month)}..."):
        model, month_actual, month_predicted, window = fit_for_month(model_name, str(month), train_months, df)

    row = edited_home(df, index)
    original = float(model.predict(df.loc[[index]])[0])
    estimate = float(model.predict(row)[0])
    edited = not np.isclose(estimate, original)

    # Spread of actual / predicted for the other homes that sold the same month.
    others = month_predicted.index != index
    ratio = (month_actual / month_predicted)[others]
    low, high = estimate * ratio.quantile([0.1, 0.9])

    cols = st.columns(3)
    cols[0].metric("Model estimate", money(estimate),
                   delta=f"{'+' if estimate >= original else '−'}{money(abs(estimate - original))} from your edits"
                   if edited else None)
    if edited:
        cols[1].metric("Estimate for the home as listed", money(original))
    else:
        error = estimate / home[da.TARGET] - 1
        cols[1].metric("Actual sale", money(home[da.TARGET]), delta=f"estimate {error:+.1%} vs sale",
                       delta_color="off")
    cols[2].metric("Likely range (80%)", f"{money(low)} – {money(high)}")
    pct_error = (month_predicted / month_actual - 1).abs()[others]
    st.caption(
        f"{model_name}, trained on {window['size']:,} sales from {window['min']:%b %Y} to {window['max']:%b %Y}. "
        f"On the other {others.sum():,} homes that sold in {month_name(month)}, its median miss was "
        f"{pct_error.median():.1%} and {(pct_error < 0.1).mean():.0%} were within 10%. The range is where 80% "
        "of those homes' prices fell relative to their estimates. In production it would come from earlier months.")
    if train_months < X:
        st.caption(f"Only {train_months} months of earlier data exist, so this model trained on fewer than the usual "
                   f"{X}. Expect it to be less accurate.")

    drivers_chart(model_name, str(month), train_months, model, row)


def drivers_chart(model_name: str, month: str, train_months: int, model, row: pd.DataFrame) -> None:
    st.markdown("#### What drives this estimate")
    with st.spinner("Explaining the estimate..."):
        features = model[:-1].transform(row)
        explanation = explainer_for(model_name, month, train_months, model)(features)
    contributions = pd.Series(explanation.values[0], index=[feature_label(c) for c in features.columns])
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    shown = pd.concat([top.head(8), pd.Series({"All other features": top.iloc[8:].sum()})])
    data = pd.DataFrame({"feature": shown.index, "contribution": shown.values})
    data["effect"] = np.where(data["contribution"] >= 0, "Raises the estimate", "Lowers the estimate")
    data["label"] = data["contribution"].map(lambda v: ("+" if v >= 0 else "−") + money(abs(v)))

    base = alt.Chart(data).encode(
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("contribution:Q", title="Effect on the estimate ($)", axis=alt.Axis(format="$~s")),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.Color("effect:N", scale=alt.Scale(domain=["Raises the estimate", "Lowers the estimate"],
                                                    range=[BLUE, ORANGE]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("label:N", title="Effect")],
    )
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(opacity=0.4).encode(x="x:Q")
    st.altair_chart((rule + bars).properties(height=34 * len(data)), width="stretch")
    st.caption(f"SHAP values. The model starts from its average estimate, {money(explanation.base_values[0])}, and "
               "each feature moves it up or down. Location (ZIP, city, coordinates) and the comparable-sales price "
               "usually matter most.")


# --- app --------------------------------------------------------------------------

def pick_random_home(df: pd.DataFrame) -> None:
    home = df.sample(1).iloc[0]
    st.session_state["query"] = f"{home['Address']} {home['PostalCode']}"


def main() -> None:
    st.set_page_config(page_title="Home Price Explorer", page_icon="🏠", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    df = load_homes()

    with st.sidebar:
        st.header("Model")
        model_name = st.radio("Model", list(MODELS), label_visibility="collapsed")
        st.caption(MODEL_NOTES[model_name])
        allow_test = st.toggle(f"Allow estimates for {month_name(TEST_MONTH)} (test month)", value=False)
        st.divider()
        st.caption(f"{len(df):,} California single-family sales, {month_name(df['CloseMonth'].min())} to "
                   f"{month_name(df['CloseMonth'].max())} (CRMLS). Each estimate uses a model trained on the {X} "
                   "months before the home sold, with only information known before it was listed.")

    st.title("Home Price Explorer")
    if "query" not in st.session_state:
        st.session_state["query"] = st.query_params.get("q", "")  # links like ?q=ocean+ave+92651
    search_col, button_col = st.columns([5, 1], vertical_alignment="bottom")
    query = search_col.text_input("Search by address", key="query",
                                  placeholder="Street address, city or ZIP, e.g. “ocean ave 92651”")
    button_col.button("Random home", on_click=pick_random_home, args=(df,), width="stretch")
    st.query_params["q"] = query

    if not query.strip():
        st.info("Type part of an address, city or ZIP code to find a home. Every word you type must match.")
        return
    matches = search(df, query)
    if matches.empty:
        st.warning("No homes match. Try fewer words, or just the street name.")
        return

    shown = matches.head(MAX_OPTIONS)
    labels = {i: option_label(home) for i, home in shown.iterrows()}
    more = f" (showing the {MAX_OPTIONS} most recent; type more to narrow it down)" if len(matches) > MAX_OPTIONS else ""
    index = st.selectbox(f"{len(matches):,} matching sales{more}", list(labels), format_func=labels.get)

    home = df.loc[index]
    st.divider()
    header(home)
    estimate, details, comps = st.tabs(["Price estimate", "Home details", "Comparable sales"])
    with estimate:
        estimate_tab(df, index, model_name, allow_test)
    with details:
        details_tab(home)
    with comps:
        comparables_tab(df, index)


main()
