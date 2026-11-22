"""Home Price Explorer: look up a sold home or enter one by hand, and get the final model's estimate.

    uv run python -m idx_exchange_data_science.serving   # train and save the final model first
    uv run streamlit run app/dashboard.py

The app only serves the saved model. It uses only information known before a home is listed.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.compose import TransformedTargetRegressor

from idx_exchange_data_science import features as ft
from idx_exchange_data_science import serving
from idx_exchange_data_science.cleaning import DATE_COL, TARGET
from idx_exchange_data_science.paths import DISTRICTS_PATH, FILTERED_PATH

ICON = Path(__file__).parent / "assets" / "icon.svg"
MAX_RESULTS = 10
MAX_SUGGESTIONS = 6
TOP_DRIVERS = 8
P = dict(text="#18181B", muted="#71717A", border="#E4E4E0", surface="#FFFFFF", subtle="#F4F4F2",
         accent="#3B5BDB", warm="#C2410C", comp="#52525B", map_style="carto-positron")

LABELS = {
    "LivingArea": "Living area", "PropertyAge": "Age", "LotSizeSquareFeet": "Lot size",
    "DistrictSEDPct": "District disadvantaged share", "LogCompPricePerSqft": "Nearby $ per sq ft",
    "LogCompImpliedPrice": "Comparable-sales price", "CompDistanceKm": "Distance to comparables",
    "BedroomsTotal": "Bedrooms", "BathroomsTotalInteger": "Bathrooms", "GarageSpaces": "Garage spaces",
    "ParkingTotal": "Parking spaces", "AttachedGarageYN": "Attached garage", "FireplaceYN": "Fireplace",
    "NewConstructionYN": "New construction", "PoolPrivateYN": "Pool", "ViewYN": "View",
    "Latitude": "Latitude", "Longitude": "Longitude", "City": "City", "CountyOrParish": "County",
    "PostalCode": "ZIP code", "ElementaryDistrict": "Elementary district", "HighDistrict": "High school district",
    "MLSHighSchoolDistrict": "High school district (MLS)", "HOAFeeMonthly": "HOA fee", "HasHOA": "Has HOA",
    "HOAFeeMissing": "HOA fee unknown", "FlooringMissing": "Flooring unknown",
    **{f"Flooring{t}": f"{t} flooring" for t in ft.FLOORING_TYPES if t != "SeeRemarks"},
    "FlooringSeeRemarks": "Flooring: see remarks",
}
FLAGS = {"PoolPrivateYN": "Pool", "ViewYN": "View", "FireplaceYN": "Fireplace",
         "AttachedGarageYN": "Attached garage", "NewConstructionYN": "New construction"}
FLAG_VALUES = {"Yes": 1.0, "No": 0.0, "Unknown": np.nan}
COUNTS = ["LivingArea", "LotSizeSquareFeet", "BedroomsTotal", "BathroomsTotalInteger", "GarageSpaces", "ParkingTotal"]

CSS = f"""<style>
.block-container {{max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem;}}
.st-key-brand button {{padding: 0; min-height: 0;}}
.st-key-brand button p {{font-size: 1.02rem; letter-spacing: -0.01em;}}
.st-key-brand button:hover p {{color: {P['accent']};}}
.eyebrow {{font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: {P['muted']}; font-weight: 600;}}
.home-title {{font-size: 2.1rem; font-weight: 650; letter-spacing: -0.025em; line-height: 1.15; margin: 1.2rem 0 .2rem;}}
.home-place {{color: {P['muted']}; margin-bottom: 1rem;}}
.big-number {{font-size: 2.7rem; font-weight: 650; letter-spacing: -0.03em; line-height: 1.1;
              font-variant-numeric: tabular-nums; margin: .25rem 0 .1rem;}}
.mid-number {{font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; font-variant-numeric: tabular-nums;}}
.muted {{color: {P['muted']};}}
.small {{font-size: .86rem;}}
.note {{font-size: .82rem; color: {P['muted']}; line-height: 1.45;}}
.comparison {{margin-top: .9rem; padding-top: .8rem; border-top: 1px solid {P['border']}; font-size: .92rem;}}
.stat-strip {{display: flex; flex-wrap: wrap; border: 1px solid {P['border']}; border-radius: 14px;
              background: {P['surface']}; margin: 1rem 0 .4rem;}}
.stat {{flex: 1 1 120px; padding: 14px 20px; border-left: 1px solid {P['border']};}}
.stat:first-child {{border-left: none;}}
.stat .value {{font-size: 1.3rem; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 2px;}}
.section-title {{font-size: 1.15rem; font-weight: 600; letter-spacing: -0.01em; margin: 1.8rem 0 .2rem;}}
.section-sub {{color: {P['muted']}; font-size: .9rem; margin-bottom: .8rem;}}
.dl-group {{margin-bottom: 1rem;}}
.dl-row {{display: flex; justify-content: space-between; gap: 16px; padding: 7px 0;
          border-bottom: 1px solid {P['border']}; font-size: .9rem;}}
.dl-row:last-child {{border-bottom: none;}}
.dl-row span:first-child {{color: {P['muted']};}}
.dl-row span:last-child {{text-align: right; font-weight: 500;}}
.st-key-estimate_card, .st-key-sale_card, .st-key-details_card {{background: {P['surface']}; border-radius: 14px;}}
.st-key-results, .st-key-suggestions {{border: 1px solid {P['border']}; border-radius: 12px; background: {P['surface']}; padding: 6px; gap: 0;}}
.st-key-suggestions {{margin-top: -6px;}}
.st-key-results button, .st-key-suggestions button {{width: 100%; padding: 9px 12px; border-radius: 8px;}}
.st-key-results button:hover, .st-key-suggestions button:hover {{background: {P['subtle']};}}
.st-key-results button > div, .st-key-results [data-testid="stMarkdownContainer"],
.st-key-suggestions button > div, .st-key-suggestions [data-testid="stMarkdownContainer"] {{
    justify-content: flex-start; width: 100%; text-align: left;}}
.landing {{padding: 3rem 0 .5rem;}}
.landing h1 {{font-size: 2.4rem; font-weight: 650; letter-spacing: -0.03em; line-height: 1.15; margin-bottom: .6rem;}}
.landing p {{color: {P['muted']}; font-size: 1.05rem; max-width: 640px;}}
.steps {{display: flex; gap: 14px; margin-top: 2rem;}}
.step {{flex: 1; border: 1px solid {P['border']}; border-radius: 14px; background: {P['surface']}; padding: 18px 20px;}}
.step .number {{color: {P['accent']}; font-weight: 600; font-size: .9rem;}}
.step .title {{font-weight: 600; margin: 4px 0 4px;}}
.step .body {{color: {P['muted']}; font-size: .9rem; line-height: 1.45;}}
.footnote {{color: {P['muted']}; font-size: .85rem; margin-top: 1.4rem;}}
</style>"""


# --- formatting -----------------------------------------------------------------------

def money(x) -> str:
    if pd.isna(x):
        return "-"
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}K"


def number(x, unit: str = "") -> str:
    return "-" if pd.isna(x) else f"{x:,.0f}{unit}"


def yes_no(x) -> str:
    return {1.0: "Yes", 0.0: "No"}.get(x, "Unknown")


def text(x) -> str:
    return "-" if pd.isna(x) else str(x)


def as_int(x):
    return None if pd.isna(x) else int(round(x))


def feature_label(name: str) -> str:
    if name.startswith("missingindicator_"):
        return LABELS[name.removeprefix("missingindicator_")] + " unknown"
    return LABELS.get(name, name)


def stat_strip(items: dict[str, str]) -> str:
    cells = "".join(f"<div class='stat'><div class='eyebrow'>{html.escape(k)}</div>"
                    f"<div class='value'>{html.escape(v)}</div></div>" for k, v in items.items())
    return f"<div class='stat-strip'>{cells}</div>"


def definition_group(title: str, rows: dict[str, str]) -> str:
    heading = f"<div class='eyebrow'>{html.escape(title)}</div>" if title else ""
    body = "".join(f"<div class='dl-row'><span>{html.escape(k)}</span><span>{html.escape(str(v))}</span></div>"
                   for k, v in rows.items())
    return f"<div class='dl-group'>{heading}{body}</div>"


# --- data and model -------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading 309,000 sales (about 15 seconds, once)...")
def load_homes() -> pd.DataFrame:
    """The modeling table plus the address and listing details shown in the app (not model inputs)."""
    df = ft.build_table()
    extra = pd.read_csv(FILTERED_PATH, usecols=["ListingKey", "UnparsedAddress", "ListPrice", "DaysOnMarket"])
    df = df.merge(extra.drop_duplicates("ListingKey"), on="ListingKey", how="left", validate="many_to_one")
    df["Address"] = df["UnparsedAddress"].fillna("Address unknown").str.strip()
    place = df["Address"] + " " + df["City"].fillna("") + " " + df["PostalCode"].fillna("")
    df["SearchText"] = " " + place.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True)
    df["CloseMonth"] = df[DATE_COL].dt.to_period("M")
    return df


@st.cache_resource(show_spinner="Loading school districts...")
def load_districts():
    return ft.load_school_districts(DISTRICTS_PATH)


@st.cache_resource(show_spinner="Loading the model...", max_entries=1)
def load_model(modified: float) -> tuple[dict, shap.TreeExplainer]:
    """The saved final model and its SHAP explainer. `modified` reloads them after retraining."""
    bundle = serving.load()
    return bundle, shap.TreeExplainer(pipeline_of(bundle["model"])[-1])


def pipeline_of(model):
    return model.regressor_ if isinstance(model, TransformedTargetRegressor) else model


def is_log_target(model) -> bool:
    return isinstance(model, TransformedTargetRegressor)


@st.cache_data(show_spinner=False, max_entries=50)
def comparables_for(key, _df: pd.DataFrame, _home: pd.DataFrame) -> pd.DataFrame:
    return ft.find_comparables(_df, _home)


def search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Sales whose address, city or ZIP contain every word of the query, most recent first."""
    mask = pd.Series(True, index=df.index)
    for word in re.findall(r"[a-z0-9]+", query.lower()):  # punctuation is ignored: "Ave," matches "Ave"
        mask &= df["SearchText"].str.contains(word, regex=False)
    return df[mask].sort_values(DATE_COL, ascending=False)


# --- navigation -----------------------------------------------------------------------
# st.session_state["view"] is "landing", "results", "home" or "manual".

# "handled" is the search text that was last acted on; suggestions show only for new text.
# These run as widget callbacks, so they can also set the search box (key "query").

def home_label(home: pd.Series) -> str:
    return f"{home['Address']}, {text(home['City'])} {text(home['PostalCode'])}"


def open_home(index, df: pd.DataFrame) -> None:
    label = home_label(df.loc[index])
    st.session_state.update(view="home", home=index, loaded=df.loc[index, "Address"], query=label, handled=label)
    st.query_params["home"] = str(df.loc[index, "ListingKey"])


def show_results(query: str) -> None:
    st.session_state.update(view="results", results_for=query, handled=query)
    st.query_params.clear()


def run_search(df: pd.DataFrame) -> None:
    """The Search button: one match opens the home, otherwise the results page lists them."""
    query = st.session_state.get("query", "").strip()
    if not query:
        return
    matches = search(df, query)
    if len(matches) == 1:
        open_home(matches.index[0], df)
    else:
        show_results(query)


def go_home() -> None:
    st.session_state.update(view="landing", query="", handled="")
    st.query_params.clear()


def enter_details() -> None:
    query = st.session_state.get("query", "")
    st.session_state.update(previous=(st.session_state["view"], st.session_state.get("home")), view="manual",
                            manual_address=query, handled=query)
    st.query_params.clear()


def close_details() -> None:
    view, home = st.session_state.get("previous", ("landing", None))
    st.session_state.update(view=view if view != "manual" else "landing", home=home)


@st.fragment
def search_bar(df: pd.DataFrame) -> None:
    """One search box with suggestions that update as you type (Enter shows them at once).
    The Search button opens the full results. Only this part reruns while typing."""
    box, button = st.columns([6, 1], vertical_alignment="bottom")
    query = box.text_input("Search", key="query", live=True, icon=":material/search:", label_visibility="collapsed",
                           placeholder="Search a sold home by street address, city or ZIP code")
    if button.button("Search", type="primary", width="stretch", on_click=run_search, args=(df,)):
        st.rerun()
    if query.strip() and query != st.session_state.get("handled"):
        suggestions(df, query)
    elif st.session_state["view"] == "results" and query == st.session_state.get("results_for"):
        results_view(df, query)  # inside the fragment, so new typing replaces old results at once


def suggestions(df: pd.DataFrame, query: str) -> None:
    matches = search(df, query)
    with st.container(key="suggestions"):
        if matches.empty:
            st.caption("No sold homes match. Try fewer words, or only the street name.")
            if st.button("Enter this home's details", icon=":material/edit_note:", type="tertiary",
                         on_click=enter_details):
                st.rerun()
            return
        for index, match in matches.head(MAX_SUGGESTIONS).iterrows():
            if st.button(f"**{match['Address']}**  |  {text(match['City'])} {text(match['PostalCode'])}  |  "
                         f"Sold {match[DATE_COL]:%b %Y} for {money(match[TARGET])}", key=f"suggest_{index}",
                         type="tertiary", width="stretch", on_click=open_home, args=(index, df)):
                st.rerun()
        if len(matches) > MAX_SUGGESTIONS:
            if st.button(f"See all {len(matches):,} results", icon=":material/list:", key="see_all", type="tertiary",
                         on_click=show_results, args=(query,)):
                st.rerun()


# --- the editable home details ----------------------------------------------------------

def defaults_from(home: pd.Series, year_base: int) -> dict:
    types = [t for t in ft.FLOORING_TYPES if home.get(f"Flooring{t}") == 1]
    return {
        **{col: as_int(home[col]) for col in COUNTS},
        "YearBuilt": as_int(year_base - home["PropertyAge"]),
        **{col: yes_no(home[col]) for col in FLAGS},
        "HOAFeeMonthly": as_int(home.get("HOAFeeMonthly")),
        "Flooring": types,
    }


def details_form(key: str, defaults: dict, inputs: set[str], zip_options: list[str] | None = None) -> dict | None:
    """The home details the model uses. Blank means unknown. Returns the values (None until a manual form is sent)."""
    manual = zip_options is not None
    with st.form(key, border=False):
        values = {}
        if manual:
            c = st.columns([2, 1])
            values["Address"] = c[0].text_input("Address (optional)", value=st.session_state.get("manual_address", ""))
            values["PostalCode"] = c[1].selectbox("ZIP code", zip_options, index=None, placeholder="Select a ZIP code")
        c = st.columns(4)
        values["LivingArea"] = c[0].number_input("Living area (sq ft)", min_value=1, step=50, value=defaults["LivingArea"])
        values["LotSizeSquareFeet"] = c[1].number_input("Lot size (sq ft)", min_value=0, step=500,
                                                        value=defaults["LotSizeSquareFeet"])
        values["BedroomsTotal"] = c[2].number_input("Bedrooms", min_value=1, max_value=20, step=1,
                                                    value=defaults["BedroomsTotal"])
        values["BathroomsTotalInteger"] = c[3].number_input("Bathrooms", min_value=1, max_value=20, step=1,
                                                            value=defaults["BathroomsTotalInteger"])
        c = st.columns(4)
        values["YearBuilt"] = c[0].number_input("Year built", step=1, value=defaults["YearBuilt"])
        values["GarageSpaces"] = c[1].number_input("Garage spaces", min_value=0, max_value=20, step=1,
                                                   value=defaults["GarageSpaces"])
        values["ParkingTotal"] = c[2].number_input("Parking spaces", min_value=0, max_value=20, step=1,
                                                   value=defaults["ParkingTotal"])
        if "HOAFeeMonthly" in inputs:
            values["HOAFeeMonthly"] = c[3].number_input("HOA fee ($ per month)", min_value=0, step=25,
                                                        value=defaults["HOAFeeMonthly"])
        c = st.columns(len(FLAGS))
        for i, (col, label) in enumerate(FLAGS.items()):
            values[col] = c[i].selectbox(label, list(FLAG_VALUES), index=list(FLAG_VALUES).index(defaults[col]))
        if any(col.startswith("Flooring") for col in inputs):
            values["Flooring"] = st.multiselect("Flooring", ft.FLOORING_TYPES, default=defaults["Flooring"],
                                                format_func=lambda t: "See remarks" if t == "SeeRemarks" else t)
        sent = st.form_submit_button("Estimate" if manual else "Update estimate", type="primary")
    if sent:
        st.session_state.setdefault("sent", set()).add(key)
    if manual and key not in st.session_state.get("sent", set()):
        return None
    return values


def apply_details(row: pd.DataFrame, values: dict, defaults: dict, year_base: int) -> pd.DataFrame:
    """The model inputs with the form's changes. Only changed values are written, so rounding in the
    form doesn't alter a home that wasn't edited."""
    row = row.copy()
    for col in COUNTS:
        if values[col] != defaults[col]:
            row[col] = np.nan if values[col] is None else float(values[col])
    if values["YearBuilt"] != defaults["YearBuilt"]:
        row["PropertyAge"] = np.nan if values["YearBuilt"] is None else float(year_base - values["YearBuilt"])
    for col in FLAGS:
        row[col] = FLAG_VALUES[values[col]]
    if "HOAFeeMonthly" in values and values["HOAFeeMonthly"] != defaults["HOAFeeMonthly"]:
        fee = values["HOAFeeMonthly"]
        row["HOAFeeMonthly"] = np.nan if fee is None else float(fee)
        row["HasHOA"] = float(bool(fee))
        row["HOAFeeMissing"] = float(fee is None)
    if "Flooring" in values and values["Flooring"] != defaults["Flooring"]:
        for t in ft.FLOORING_TYPES:
            row[f"Flooring{t}"] = float(t in values["Flooring"])
        row["FlooringMissing"] = float(not values["Flooring"])
    # The comparable-sales price is the neighbours' $ per sq ft times this home's living area.
    row["LogCompImpliedPrice"] = row["LogCompPricePerSqft"] + np.log(row["LivingArea"].where(row["LivingArea"] > 0))
    return row


def new_home(df: pd.DataFrame, zip_code: str, address: str) -> pd.DataFrame:
    """A home listed next month in `zip_code`, placed at the ZIP's median coordinates, with its school
    districts. Its own details come from apply_details, and its comparable sales from comparable_features."""
    in_zip = df[df["PostalCode"] == zip_code]
    city = in_zip["City"].mode().iat[0] if in_zip["City"].notna().any() else np.nan
    mls_district = in_zip["MLSHighSchoolDistrict"].mode()
    listed = (df["CloseMonth"].max() + 1).to_timestamp()
    row = pd.DataFrame([{col: np.nan for col in df.columns}])
    row = row.assign(
        PostalCode=zip_code, City=city, CountyOrParish=in_zip["CountyOrParish"].mode().iat[0],
        Latitude=in_zip["Latitude"].median(), Longitude=in_zip["Longitude"].median(),
        MLSHighSchoolDistrict=mls_district.iat[0] if len(mls_district) else np.nan,
        HasHOA=0.0, HOAFeeMissing=1.0, FlooringMissing=1.0, **{f"Flooring{t}": 0.0 for t in ft.FLOORING_TYPES},
        **{ft.LISTING_DATE: listed, DATE_COL: listed + pd.DateOffset(months=ft.DEFAULT_MONTHS_TO_CLOSE)},
        Address=address.strip() or f"A home in {text(city)} {zip_code}",
    )
    return ft.add_school_districts(row, load_districts())


# --- page sections ----------------------------------------------------------------------

def range_band(estimate: float, ranges: dict, level: float, sale: float | None) -> str:
    """Nested 80 / 90 / 95% ranges around the estimate, the selected one emphasized, and the sale price."""
    outer_low, outer_high = (estimate * r for r in ranges[max(ranges)])
    low_end = min(outer_low, sale) if sale else outer_low
    high_end = max(outer_high, sale) if sale else outer_high
    pad = (high_end - low_end) * 0.05
    low_end, high_end = low_end - pad, high_end + pad

    def x(v):  # position along the track, in percent
        return (v - low_end) / (high_end - low_end) * 100

    bands = ""
    for lvl, opacity in zip(sorted(ranges, reverse=True), (0.14, 0.26, 0.42)):
        lo, hi = (estimate * r for r in ranges[lvl])
        selected = lvl == level
        outline = f"box-shadow: inset 0 0 0 2px {P['accent']};" if selected else ""
        bands += (f"<div style='position:absolute; left:{x(lo):.2f}%; width:{x(hi) - x(lo):.2f}%; top:24px; height:18px;"
                  f" border-radius:9px; background:{P['accent']}; opacity:{opacity + (0.14 if selected else 0):.2f};"
                  f" {outline}'></div>")
    marks = (f"<div style='position:absolute; left:{x(estimate):.2f}%; top:16px; height:34px; width:3px;"
             f" margin-left:-1.5px; border-radius:2px; background:{P['text']};'></div>"
             f"<div style='position:absolute; left:{x(estimate):.2f}%; top:-2px; transform:translateX(-50%);"
             f" font-size:.75rem; font-weight:600; white-space:nowrap;'>Estimate</div>")
    if sale:
        marks += (f"<div style='position:absolute; left:{x(sale):.2f}%; top:26px; width:14px; height:14px;"
                  f" margin-left:-7px; border-radius:50%; background:{P['warm']}; box-shadow:0 0 0 2.5px {P['surface']};'></div>"
                  f"<div style='position:absolute; left:{x(sale):.2f}%; top:50px; transform:translateX(-50%);"
                  f" font-size:.75rem; font-weight:600; color:{P['warm']}; white-space:nowrap;'>Sold {money(sale)}</div>")
    return (f"<div role='img' aria-label='Price ranges around the estimate' "
            f"style='position:relative; height:72px; margin:.6rem 8% .3rem;'>{bands}{marks}</div>")


def estimate_card(estimate: float, original: float, ranges: dict, sale: float | None, key: str) -> None:
    with st.container(border=True, key="estimate_card"):
        st.markdown("<div class='eyebrow'>Estimated value</div>"
                    f"<div class='big-number'>{money(estimate)}</div>", unsafe_allow_html=True)
        if not np.isclose(estimate, original):
            st.markdown(f"<div class='muted small'>With your changes. {money(original)} for the home as listed.</div>",
                        unsafe_allow_html=True)
        level = st.segmented_control("Range", list(ranges), default=min(ranges), key=f"level_{key}",
                                     format_func=lambda v: f"{v:.0%} range", label_visibility="collapsed")
        level = level or min(ranges)
        lo, hi = (estimate * r for r in ranges[level])
        st.markdown(f"<div class='mid-number'>{money(lo)} - {money(hi)}</div>"
                    + range_band(estimate, ranges, level, sale)
                    + f"<div class='note'>{level:.0%} of the homes that the model had not seen sold inside this "
                      "range of their estimate (validation months). A wider range is more certain.</div>",
                    unsafe_allow_html=True)


def sale_card(home: pd.Series, original: float, bundle: dict) -> None:
    sold, listed = home[TARGET], home["ListPrice"]
    gap = original / sold - 1
    side = "above" if gap > 0 else "below"
    with st.container(border=True, key="sale_card"):
        st.markdown(
            "<div class='eyebrow'>Sale record</div>"
            f"<div class='big-number'>{money(sold)}</div>"
            f"<div class='muted small'>Sold {home[DATE_COL]:%B %-d, %Y}</div>"
            + definition_group("", {"List price": money(listed),
                                    "Sold vs list": "-" if pd.isna(listed) else f"{sold / listed - 1:+.1%}",
                                    "Days on market": number(home["DaysOnMarket"])})
            + f"<div class='comparison'>The estimate is <b>{abs(gap):.1%} {side}</b> the sale price.</div>",
            unsafe_allow_html=True)
        if bundle["train_first"] <= home[DATE_COL] <= bundle["train_last"]:
            st.markdown("<div class='note'>This sale is in the model's training data, so this gap is not a fair "
                        "accuracy check. The Model panel shows the validation accuracy.</div>", unsafe_allow_html=True)


def comparables_section(comps: pd.DataFrame, home: pd.Series) -> None:
    st.markdown("<div class='section-title'>Comparable sales</div>", unsafe_allow_html=True)
    if comps.empty:
        st.markdown("<div class='section-sub'>No comparable sales: the home has no location, or there are too few "
                    "earlier sales.</div>", unsafe_allow_html=True)
        return
    ppsf = comps["PricePerSqft"].median()
    implied = f", or <b>{money(ppsf * home['LivingArea'])}</b> for {number(home['LivingArea'])} sq ft" \
        if pd.notna(home["LivingArea"]) else ""
    st.markdown(f"<div class='section-sub'>The 10 nearest homes that sold in the 12 months before listing. Their median "
                f"is <b>${ppsf:,.0f} per sq ft</b>{implied}.</div>", unsafe_allow_html=True)
    map_col, table_col = st.columns(2, gap="medium")
    with map_col:
        st.plotly_chart(comps_map(comps, home), theme=None, config={"displayModeBar": False, "scrollZoom": True})
    with table_col:
        table = pd.DataFrame({"Address": comps["Address"], "Sold": comps[DATE_COL].dt.strftime("%b %Y"),
                              "Price": comps[TARGET].map(money), "$/sq ft": comps["PricePerSqft"].map("${:,.0f}".format)})
        narrow = {col: st.column_config.Column(width="small") for col in ["Sold", "Price", "$/sq ft"]}
        st.dataframe(table, hide_index=True, height=388, column_config=narrow)


def comps_map(comps: pd.DataFrame, home: pd.Series) -> go.Figure:
    hover = (comps["Address"] + "<br>" + comps[TARGET].map(money) + " | " + comps[DATE_COL].dt.strftime("%b %Y")
             + "<br>$" + comps["PricePerSqft"].round().astype(int).map("{:,}".format) + " per sq ft | "
             + comps["DistanceKm"].map("{:.1f} km away".format))
    halo = dict(color=P["surface"], opacity=1)
    fig = go.Figure([
        go.Scattermap(lat=comps["Latitude"], lon=comps["Longitude"], mode="markers", hoverinfo="skip",
                      marker=dict(size=15, **halo)),
        go.Scattermap(lat=comps["Latitude"], lon=comps["Longitude"], mode="markers", text=hover,
                      hovertemplate="%{text}<extra></extra>", marker=dict(size=10, color=P["comp"])),
        go.Scattermap(lat=[home["Latitude"]], lon=[home["Longitude"]], mode="markers", hoverinfo="skip",
                      marker=dict(size=24, **halo)),
        go.Scattermap(lat=[home["Latitude"]], lon=[home["Longitude"]], mode="markers",
                      text=[f"<b>{html.escape(home['Address'])}</b><br>This home"],
                      hovertemplate="%{text}<extra></extra>", marker=dict(size=17, color=P["accent"])),
    ])
    zoom = float(np.clip(13.2 - np.log2(max(comps["DistanceKm"].max(), 0.3)), 8, 15))
    fig.update_layout(
        map=dict(style=P["map_style"], center=dict(lat=home["Latitude"], lon=home["Longitude"]), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0), height=388, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=P["surface"], bordercolor=P["border"], font=dict(color=P["text"], size=12)),
    )
    return fig


def drivers_section(model, explainer: shap.TreeExplainer, row: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>What drives this estimate</div>", unsafe_allow_html=True)
    features = pipeline_of(model)[:-1].transform(row)
    explanation = explainer(features)
    values = pd.Series(explanation.values[0], index=[feature_label(c) for c in features.columns])
    log = is_log_target(model)
    if log:  # contributions are in log price: show each one as a % change
        values, base = np.expm1(values), np.expm1(explanation.base_values[0])
    else:
        base = explanation.base_values[0]
    ordered = values.reindex(values.abs().sort_values(ascending=False).index)
    rest = np.prod(1 + ordered.iloc[TOP_DRIVERS:]) - 1 if log else ordered.iloc[TOP_DRIVERS:].sum()
    shown = pd.concat([ordered.head(TOP_DRIVERS), pd.Series({"All other features": rest})])
    data = pd.DataFrame({"feature": shown.index, "effect": shown.values})
    data["direction"] = np.where(data["effect"] >= 0, "Raises the estimate", "Lowers the estimate")
    data["label"] = data["effect"].map(lambda v: f"{v:+.1%}" if log else ("+" if v >= 0 else "-") + money(abs(v)))
    st.markdown(f"<div class='section-sub'>The model starts from a typical home ({money(base)}). Each feature then "
                f"moves the estimate {'by a percentage' if log else 'up or down'} (SHAP values).</div>",
                unsafe_allow_html=True)
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=16).encode(
        y=alt.Y("feature:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=200, labelPadding=8, ticks=False, domain=False)),
        x=alt.X("effect:Q", title=None, axis=alt.Axis(format="+.0%" if log else "$~s", domain=False, ticks=False)),
        color=alt.Color("direction:N", legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(domain=["Raises the estimate", "Lowers the estimate"],
                                        range=[P["accent"], P["warm"]])),
        tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("label:N", title="Effect")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=P["muted"], opacity=0.5).encode(x="x:Q")
    st.altair_chart((zero + bars).properties(height=32 * len(data)).configure_view(stroke=None)
                    .configure_axis(gridColor=P["border"], labelColor=P["muted"])
                    .configure_legend(labelColor=P["muted"]), width="stretch")


def details_section(home: pd.Series) -> None:
    flooring = [("See remarks" if t == "SeeRemarks" else t) for t in ft.FLOORING_TYPES if home.get(f"Flooring{t}") == 1]
    listed = home.get(ft.LISTING_DATE)
    with st.container(border=True, key="details_card"):
        st.markdown(
            definition_group("Features", {
                **{label: yes_no(home[col]) for col, label in FLAGS.items()},
                "Garage spaces": number(home["GarageSpaces"]), "Parking spaces": number(home["ParkingTotal"]),
                "Flooring": "Unknown" if home.get("FlooringMissing") == 1 else ", ".join(flooring) or "Other",
                "HOA fee": "Unknown" if pd.isna(home.get("HOAFeeMonthly")) else f"${home['HOAFeeMonthly']:,.0f} per month",
            })
            + definition_group("Location", {
                "City": text(home["City"]), "County": text(home["CountyOrParish"]), "ZIP code": text(home["PostalCode"]),
                "Elementary district": text(home["ElementaryDistrict"]),
                "High school district": text(home["HighDistrict"]),
                "Disadvantaged students": "-" if pd.isna(home["DistrictSEDPct"]) else f"{home['DistrictSEDPct']:.0f}%",
                "Listed": "Unknown" if pd.isna(listed) else f"{listed:%b %-d, %Y}",
            }), unsafe_allow_html=True)


def model_panel(bundle: dict, df: pd.DataFrame) -> None:
    model = bundle["model"]
    inputs = serving.input_columns(model)
    groups = ["Dataset A"] + [name for name, col in [
        ("school districts", "ElementaryDistrict"), ("MLS district", "MLSHighSchoolDistrict"),
        ("HOA", "HOAFeeMonthly"), ("flooring", "FlooringMissing"), ("comparable sales", "LogCompImpliedPrice"),
    ] if col in inputs]
    v = bundle["validation"]
    months = bundle["validation_months"]
    st.markdown(f"**{type(pipeline_of(model)[-1]).__name__}**{' on log price' if is_log_target(model) else ''}  \n"
                f"Features: {', '.join(groups)}.  \n"
                f"Trained on {bundle['n_train']:,} sales, {bundle['train_first']:%b %Y} to {bundle['train_last']:%b %Y}.")
    st.markdown(f"<div class='eyebrow' style='margin-top:.8rem'>Validation, {months[0]} to {months[-1]}</div>",
                unsafe_allow_html=True)
    c = st.columns(2)
    c[0].metric("Median error", f"{v['median_ape']:.1%}")
    c[1].metric("Within 10%", f"{v['within_10pct']:.0%}")
    c[0].metric("R²", f"{v['r2']:.3f}")
    c[1].metric("Mean error", f"{v['mape']:.1%}")
    st.markdown("<div class='note'>Each validation month is predicted by a model trained only on the months before "
                "it. The price ranges come from these out-of-sample errors: the 80% range holds the middle 80% of the "
                f"actual / estimated price ratios. Data: {len(df):,} CRMLS single-family sales.</div>",
                unsafe_allow_html=True)


# --- views ------------------------------------------------------------------------------

def enter_details_link(label: str = "Enter a home's details instead") -> None:
    if st.button(label, icon=":material/edit_note:", type="tertiary", on_click=enter_details):
        st.rerun()  # leave the search fragment and show the form


def landing(df: pd.DataFrame) -> None:
    first, last = df["CloseMonth"].min().strftime("%B %Y"), df["CloseMonth"].max().strftime("%B %Y")
    enter_details_link("Not a sold home? Enter its details")
    st.markdown(
        "<div class='landing'><h1>Estimate what a California home will sell for</h1>"
        "<p>The model prices a single-family home from what is known before it goes on the market: the home itself, "
        "its location and recent sales nearby.</p></div>"
        "<div class='steps'>"
        "<div class='step'><div class='number'>1</div><div class='title'>Find a home</div>"
        "<div class='body'>Type a street address, city or ZIP code, then pick a home from the suggestions.</div></div>"
        "<div class='step'><div class='number'>2</div><div class='title'>Check its details</div>"
        "<div class='body'>The listing details load for you. Open them to review or change any value.</div></div>"
        "<div class='step'><div class='number'>3</div><div class='title'>Read the estimate</div>"
        "<div class='body'>See the estimated price, how certain it is, the nearby sales and what drives it.</div></div>"
        "</div>"
        f"<div class='footnote'>{len(df):,} CRMLS single-family sales | {first} to {last}</div>",
        unsafe_allow_html=True)


def results_view(df: pd.DataFrame, query: str) -> None:
    matches = search(df, query)
    if matches.empty:
        st.markdown(f"<div class='section-title'>No sold homes match \"{html.escape(query)}\"</div>"
                    "<div class='section-sub'>Try fewer words, or only the street name. If the home is not in the "
                    "data, you can enter its details.</div>", unsafe_allow_html=True)
        enter_details_link()
        return
    shown = matches.head(MAX_RESULTS)
    more = f", showing the {MAX_RESULTS} most recent" if len(matches) > MAX_RESULTS else ""
    st.markdown(f"<div class='section-title'>{len(matches):,} homes match \"{html.escape(query)}\"</div>"
                f"<div class='section-sub'>Select a home{more}.</div>", unsafe_allow_html=True)
    with st.container(key="results"):
        for index, match in shown.iterrows():
            if st.button(f"**{match['Address']}**  |  {text(match['City'])} {text(match['PostalCode'])}  |  "
                         f"Sold {match[DATE_COL]:%b %Y} for {money(match[TARGET])}",
                         key=f"pick_{index}", type="tertiary", width="stretch", on_click=open_home, args=(index, df)):
                st.rerun()
    enter_details_link("Not listed here? Enter the home's details")


def home_view(df: pd.DataFrame, index, bundle: dict, explainer) -> None:
    home = df.loc[index]
    model, inputs = bundle["model"], serving.input_columns(bundle["model"])
    year_base = home[DATE_COL].year
    if st.session_state.pop("loaded", None):
        st.toast(f"Listing details loaded for {home['Address']}", icon=":material/check_circle:")
    place = ", ".join([text(home["City"]), f"{text(home['CountyOrParish'])} County", text(home["PostalCode"])])
    st.markdown(f"<div class='home-title'>{html.escape(home['Address'])}</div>"
                f"<div class='home-place'>{html.escape(place)}</div>", unsafe_allow_html=True)
    defaults = defaults_from(home, year_base)
    with st.expander("Listing details loaded. Review or adjust them.", icon=":material/check_circle:"):
        st.caption("These are the values the model uses. Change any of them to see a new estimate. Blank means unknown.")
        values = details_form(f"edit_{index}", defaults, inputs)
    base = df.loc[[index]]
    row = apply_details(base, values, defaults, year_base)
    original, estimate = float(model.predict(base)[0]), float(model.predict(row)[0])
    left, right = st.columns([3, 2], gap="medium")
    with left:
        estimate_card(estimate, original, bundle["ranges"], home[TARGET], str(index))
    with right:
        sale_card(home, original, bundle)
    st.markdown(stat_strip({
        "Living area": number(home["LivingArea"], " sq ft"), "Bedrooms": number(home["BedroomsTotal"]),
        "Bathrooms": number(home["BathroomsTotalInteger"]), "Lot": number(home["LotSizeSquareFeet"], " sq ft"),
        "Year built": text(as_int(year_base - home["PropertyAge"])),
    }), unsafe_allow_html=True)
    comparables_section(comparables_for(index, df, base), home)
    drivers_col, details_col = st.columns([3, 2], gap="large")
    with drivers_col:
        drivers_section(model, explainer, row)
    with details_col:
        st.markdown("<div class='section-title'>Property details</div>", unsafe_allow_html=True)
        details_section(home)


def manual_view(df: pd.DataFrame, bundle: dict, explainer) -> None:
    model, inputs = bundle["model"], serving.input_columns(bundle["model"])
    title, close = st.columns([6, 1], vertical_alignment="bottom")
    title.markdown("<div class='home-title'>Enter a home's details</div>", unsafe_allow_html=True)
    close.button("Close", icon=":material/close:", key="close_details", width="stretch", on_click=close_details)
    st.markdown("<div class='home-place'>For a home that is not in the data. The model places the home at the center "
                "of its ZIP code to find its school districts and nearby sales.</div>", unsafe_allow_html=True)
    zips = df.groupby("PostalCode")["City"].agg(lambda c: c.mode().iat[0] if c.notna().any() else "")
    blank = {**dict.fromkeys(COUNTS + ["YearBuilt", "HOAFeeMonthly"]), **dict.fromkeys(FLAGS, "Unknown"), "Flooring": []}
    with st.container(border=True, key="details_card"):
        values = details_form("manual", blank, inputs, [f"{z} | {zips[z]}" for z in sorted(zips.index)])
    if values is None:
        return
    if not values["PostalCode"] or not values["LivingArea"]:
        st.warning("Select a ZIP code and enter the living area.")
        return
    zip_code = values["PostalCode"].split(" | ")[0]
    base = new_home(df, zip_code, values["Address"])
    comps = comparables_for(("zip", zip_code), df, base)
    base = base.assign(**ft.comparable_features(comps, np.nan))
    row = apply_details(base, values, blank, base[ft.LISTING_DATE].iat[0].year)
    home = row.iloc[0]
    estimate = float(model.predict(row)[0])
    st.markdown(f"<div class='home-title' style='font-size:1.5rem'>{html.escape(home['Address'])}</div>"
                f"<div class='home-place'>Estimated as if listed in {home[ft.LISTING_DATE]:%B %Y}</div>",
                unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        estimate_card(estimate, estimate, bundle["ranges"], None, f"manual_{zip_code}")
    with right:
        drivers_section(model, explainer, row)
    comparables_section(comps, home)


def main() -> None:
    st.set_page_config(page_title="Home Price Explorer", page_icon=str(ICON), layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if not serving.MODEL_PATH.exists():
        st.error("No trained model yet. Run `uv run python -m idx_exchange_data_science.serving`, then reload.")
        st.stop()
    bundle, explainer = load_model(serving.MODEL_PATH.stat().st_mtime)
    if "ranges" not in bundle:
        st.error("The saved model is from an older version. Train it again with "
                 "`uv run python -m idx_exchange_data_science.serving`, then reload.")
        st.stop()
    df = load_homes()

    if "view" not in st.session_state:  # first load: a shared link like ?home=<ListingKey> opens that home
        st.session_state["view"] = "landing"
        shared = df.index[df["ListingKey"].astype(str) == st.query_params.get("home", "")]
        if len(shared):
            label = home_label(df.loc[shared[0]])
            st.session_state.update(view="home", home=shared[0], query=label, handled=label)

    brand, model_col = st.columns([5, 1], vertical_alignment="center")
    brand.button("**Home Price Explorer** :gray[| IDX Exchange]", key="brand", type="tertiary", on_click=go_home,
                 help="Back to the start page")
    with model_col.popover("Model", icon=":material/insights:", width="stretch"):
        model_panel(bundle, df)
    search_bar(df)

    view = st.session_state["view"]
    if view == "home":  # "results" is drawn by search_bar
        home_view(df, st.session_state["home"], bundle, explainer)
    elif view == "manual":
        manual_view(df, bundle, explainer)
    elif view == "landing":
        landing(df)


main()
