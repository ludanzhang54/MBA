"""
Oriental Horizon Review Intelligence — ISOM5240 Group Project
==============================================================

A Streamlit app that turns guest reviews into pricing signals for the hotel
revenue management system (RMS) of Oriental Horizon Hotel Business.

Two fine-tuned Hugging Face pipelines run on every review:

    1. Review sentiment  (negative / neutral / positive)  -> reputation
    2. Guest segment     (business / couple / family)     -> demand mix

The results become a Net Reputation Score, a suggested room-rate adjustment
and segment-specific pricing actions. A second tab measures the accuracy of
both models on labelled test reviews directly on Streamlit Cloud.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st
from transformers import pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_USERNAME = "YOUR_HF_USERNAME"

# Fine-tuned models uploaded by Notebooks 01 and 02.
SENTIMENT_MODEL = f"{HF_USERNAME}/oh-review-sentiment"
SEGMENT_MODEL = f"{HF_USERNAME}/oh-guest-segment"

# Labelled test data uploaded by Notebook 00.
DATA_DIR = f"hf://datasets/{HF_USERNAME}/oriental-horizon-hotel-reviews"

MAX_LENGTH = 256        # same truncation as during fine-tuning
BATCH_SIZE = 8
SEED = 42               # same sampling seed as Notebook 03
MIN_RELIABLE_REVIEWS = 20

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SEGMENT_LABELS = ["business", "couple", "family"]

# Chart colours. Sentiment is a polarity (red = bad, grey = neutral, blue = good);
# segments are categories with three distinct, colour-blind-safe hues.
SENTIMENT_COLORS = ["#e34948", "#9b9a95", "#2a78d6"]
SEGMENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

# Pricing levers for each guest segment, shown when the segment is common.
SEGMENT_ACTIONS = {
    "business": (
        "💼 **Business travellers** book mostly Monday–Thursday and are less "
        "price-sensitive. Raise weekday rates on high-demand dates, protect rooms "
        "for negotiated corporate rates, and sell late check-out and fast Wi-Fi."
    ),
    "couple": (
        "💑 **Couples** travel at weekends and on holidays and respond to experiences. "
        "Raise weekend rates on peak dates and bundle breakfast, room upgrades or "
        "late check-out instead of discounting."
    ),
    "family": (
        "👨‍👩‍👧 **Families** travel in school holidays, compare prices and stay longer. "
        "Offer family rooms and connecting rooms, length-of-stay discounts and "
        "early-booking offers, and avoid steep last-minute price rises."
    ),
}

SAMPLE_REVIEWS = """\
The room was spotless and the staff upgraded us for our anniversary. Breakfast was amazing.
Good location for the conference centre, fast wifi and a quiet desk to work in the evening.
The kids loved the pool and the staff found us an extra bed for our son. Great family stay.
Room smaller than in the photos and the air conditioning was noisy, but the staff were helpful.
Dirty bathroom, rude reception and we waited 40 minutes to check in. Very disappointing.
Excellent value, close to the metro, but breakfast was the same every day for my business trip."""


# ---------------------------------------------------------------------------
# Model and data loading (cached: loaded once, shared by all users)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the AI models (first time only, about 1 minute)...")
def load_models():
    """Load both fine-tuned text-classification pipelines on the CPU."""
    sentiment = pipeline("text-classification", model=SENTIMENT_MODEL, device=-1)
    segment = pipeline("text-classification", model=SEGMENT_MODEL, device=-1)
    return sentiment, segment


@st.cache_data(show_spinner="Loading labelled test reviews...")
def load_test_set(task):
    """Load the held-out test reviews of one task ('sentiment' or 'segment')."""
    return pd.read_csv(f"{DATA_DIR}/{task}_test.csv")


# ---------------------------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------------------------

def classify(classifier, texts):
    """Run one pipeline on a list of texts. Returns (labels, confidence scores)."""
    outputs = classifier(texts, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LENGTH)
    return [o["label"] for o in outputs], [round(o["score"], 3) for o in outputs]


def analyse_reviews(texts):
    """Predict sentiment and guest segment for every review."""
    sentiment_model, segment_model = load_models()
    sentiments, sentiment_scores = classify(sentiment_model, texts)
    segments, segment_scores = classify(segment_model, texts)
    return pd.DataFrame({
        "review": texts,
        "sentiment": sentiments,
        "sentiment_confidence": sentiment_scores,
        "segment": segments,
        "segment_confidence": segment_scores,
    })


def label_shares(series, labels):
    """Share of each label in a column, in a fixed label order (missing labels = 0)."""
    return series.value_counts(normalize=True).reindex(labels, fill_value=0.0)


def net_reputation_score(results):
    """Positive share minus negative share, in points from -100 to +100."""
    shares = label_shares(results["sentiment"], SENTIMENT_LABELS)
    return round((shares["positive"] - shares["negative"]) * 100, 1)


def suggested_adjustment(nrs, benchmark, max_adjustment):
    """Rate adjustment in %: +1% per 10 NRS points above the benchmark, capped at ±max."""
    return max(-max_adjustment, min(max_adjustment, (nrs - benchmark) / 10))


def read_uploaded_reviews(uploaded_file):
    """Read reviews from an uploaded CSV ('review' or 'text' column) or TXT file (one per line)."""
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        column = next((c for c in df.columns if c.lower() in ("review", "text", "comment")), df.columns[0])
        return df[column].dropna().astype(str).tolist()
    return uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()


def clean_texts(texts):
    """Strip whitespace and drop empty lines."""
    return [t.strip() for t in texts if t and t.strip()]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def share_bar_chart(shares, labels, colors):
    """Horizontal bar chart of label shares with a percentage label on each bar."""
    data = pd.DataFrame({"label": labels, "share": [shares[label] for label in labels]})
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=labels, title=None),
        x=alt.X("share:Q", axis=alt.Axis(format="%", grid=False), scale=alt.Scale(domain=[0, 1]), title=None),
        tooltip=[alt.Tooltip("label:N", title="Label"), alt.Tooltip("share:Q", title="Share", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color("label:N", scale=alt.Scale(domain=labels, range=colors), legend=None)
    )
    text = base.mark_text(align="left", dx=4).encode(text=alt.Text("share:Q", format=".0%"))
    return (bars + text).properties(height=130)


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------

def sidebar_settings():
    """Pricing inputs from the revenue manager."""
    with st.sidebar:
        st.header("⚙️ Pricing settings")
        base_rate = st.number_input("Current base rate (CNY / night)", min_value=100, value=800, step=50)
        benchmark = st.slider(
            "Competitor Net Reputation Score", -100, 100, 40,
            help="Average NRS of your competitive set. Analyse competitor reviews in this app to measure it.",
        )
        max_adjustment = st.slider("Maximum rate adjustment (%)", 1, 15, 5)
        st.divider()
        st.caption(
            "**Net Reputation Score (NRS)** = % positive reviews − % negative reviews. "
            "Each 10 points above the competitor NRS suggests +1% on the base rate (and below it, −1%), "
            "up to the maximum."
        )
    return base_rate, benchmark, max_adjustment


def review_input():
    """Let the user paste, upload or sample reviews. Returns a list of texts."""
    source = st.radio(
        "Reviews to analyse",
        ["Paste reviews", "Upload a file", "Random test reviews"],
        horizontal=True,
    )
    if source == "Paste reviews":
        text = st.text_area("One review per line", value=SAMPLE_REVIEWS, height=180)
        return clean_texts(text.splitlines())
    if source == "Upload a file":
        uploaded = st.file_uploader("CSV with a 'review' column, or a TXT file with one review per line",
                                    type=["csv", "txt"])
        return clean_texts(read_uploaded_reviews(uploaded)) if uploaded else []
    count = st.select_slider("Number of reviews", options=[25, 50, 100, 200], value=50)
    return load_test_set("sentiment").sample(n=count, random_state=SEED)["text"].tolist()


def show_pricing_insight(results, base_rate, benchmark, max_adjustment):
    """Headline numbers, charts, pricing suggestion and the per-review table."""
    nrs = net_reputation_score(results)
    adjustment = suggested_adjustment(nrs, benchmark, max_adjustment)
    suggested_rate = base_rate * (1 + adjustment / 100)
    sentiment_shares = label_shares(results["sentiment"], SENTIMENT_LABELS)
    segment_shares = label_shares(results["segment"], SEGMENT_LABELS)

    if len(results) < MIN_RELIABLE_REVIEWS:
        st.warning(f"Only {len(results)} reviews analysed. Use at least {MIN_RELIABLE_REVIEWS} "
                   "for a reliable pricing signal.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reviews analysed", len(results))
    col2.metric("Net Reputation Score", f"{nrs:+.0f}", f"{nrs - benchmark:+.0f} vs competitors")
    col3.metric("Suggested rate change", f"{adjustment:+.1f}%")
    rate_change = suggested_rate - base_rate
    col4.metric("Suggested base rate", f"¥{suggested_rate:,.0f}",
                f"{'-' if rate_change < 0 else '+'}¥{abs(rate_change):,.0f}")  # sign first -> red/green arrow

    left, right = st.columns(2)
    with left:
        st.markdown("**Review sentiment**")
        st.altair_chart(share_bar_chart(sentiment_shares, SENTIMENT_LABELS, SENTIMENT_COLORS), width="stretch")
    with right:
        st.markdown("**Guest segment mix**")
        st.altair_chart(share_bar_chart(segment_shares, SEGMENT_LABELS, SEGMENT_COLORS), width="stretch")

    st.subheader("💡 Pricing actions")
    if adjustment > 0:
        st.success(f"Reputation is **{nrs - benchmark:+.0f} points above** competitors: there is room for "
                   f"a **{adjustment:.1f}% premium** (¥{base_rate:,.0f} → ¥{suggested_rate:,.0f}).")
    elif adjustment < 0:
        st.error(f"Reputation is **{benchmark - nrs:.0f} points below** competitors: a premium is hard to "
                 f"justify. Consider **{adjustment:.1f}%** (¥{base_rate:,.0f} → ¥{suggested_rate:,.0f}) "
                 "and fix the issues in the negative reviews below.")
    else:
        st.info("Reputation is in line with competitors: keep the current base rate.")

    for segment in segment_shares.sort_values(ascending=False).index:
        if segment_shares[segment] >= 0.25:
            st.markdown(f"{SEGMENT_ACTIONS[segment]} *({segment_shares[segment]:.0%} of reviews)*")

    st.subheader("Negative share by segment")
    st.caption("Which guest group is least satisfied, and where to fix the product first.")
    by_segment = (
        results.assign(negative=results["sentiment"] == "negative")
        .groupby("segment")
        .agg(reviews=("review", "size"), negative_share=("negative", "mean"))
        .reindex(SEGMENT_LABELS)
        .dropna()
    )
    st.dataframe(by_segment.style.format({"negative_share": "{:.0%}", "reviews": "{:.0f}"}), width="stretch")

    st.subheader("Review details")
    st.dataframe(results, width="stretch", hide_index=True)
    st.download_button("⬇️ Download results (CSV)", results.to_csv(index=False), "review_insights.csv", "text/csv")
    st.caption("Suggestions support the revenue manager and the RMS. They do not replace final pricing decisions.")


def analysis_tab(base_rate, benchmark, max_adjustment):
    """Tab 1: analyse reviews and produce pricing signals."""
    texts = review_input()
    if st.button("🔍 Analyse reviews", type="primary", disabled=not texts):
        with st.spinner(f"Analysing {len(texts)} reviews..."):
            st.session_state.results = analyse_reviews(texts)
    if "results" in st.session_state:
        show_pricing_insight(st.session_state.results, base_rate, benchmark, max_adjustment)


def accuracy_tab():
    """Tab 2: measure model accuracy on labelled test reviews, live on Streamlit Cloud."""
    st.write(
        "Runs both fine-tuned models on **labelled test reviews they never saw during training** "
        "and compares the predictions with the true labels. With the same seed, the 300-review "
        "test reproduces Experiment 2 in Notebook 03."
    )
    sample_size = st.select_slider("Test reviews per model", options=[100, 300, 500], value=300)
    if not st.button("▶️ Run accuracy check", type="primary"):
        return

    sentiment_model, segment_model = load_models()
    for task, model, labels in [("sentiment", sentiment_model, SENTIMENT_LABELS),
                                ("segment", segment_model, SEGMENT_LABELS)]:
        test = load_test_set(task)
        sample = test.sample(n=min(sample_size, len(test)), random_state=SEED)
        with st.spinner(f"Testing the {task} model on {len(sample)} reviews..."):
            predicted, _ = classify(model, sample["text"].tolist())
        sample = sample.assign(predicted=predicted)
        accuracy = (sample["label"] == sample["predicted"]).mean()

        st.subheader(f"{task.capitalize()} model")
        st.metric(f"Accuracy on {len(sample)} test reviews", f"{accuracy:.1%}")
        matrix = pd.crosstab(sample["label"], sample["predicted"]).reindex(
            index=labels, columns=labels, fill_value=0
        )
        matrix.index.name, matrix.columns.name = "true label", "predicted"
        st.caption("Confusion matrix (rows = true label, columns = prediction)")
        st.dataframe(matrix, width="stretch")


def about_tab():
    """Tab 3: how the app works."""
    st.markdown(f"""
**Business problem.** Oriental Horizon prices its rooms with a revenue management system
that uses numbers such as occupancy and booking pace. Guest reviews also affect what
guests will pay, but they are free text, so the RMS cannot use them. This app turns
reviews into two signals the RMS can use.

**Pipeline**

* **Pipeline 1:** guest review → sentiment model (`{SENTIMENT_MODEL}`) → Net Reputation Score → rate adjustment
* **Pipeline 2:** guest review → guest segment model (`{SEGMENT_MODEL}`) → segment mix → segment pricing actions

Both models are `distilbert`/`distilroberta` models from Hugging Face. We fine-tuned them on
class-balanced Booking.com hotel reviews (Kaggle *515K Hotel Reviews Data in Europe*).
""")


def main():
    st.set_page_config(page_title="Oriental Horizon Review Intelligence", page_icon="🏨", layout="wide")
    st.title("🏨 Oriental Horizon · Review Intelligence for Revenue Management")
    st.caption("Turns guest reviews into reputation and guest-segment signals for room pricing.")

    base_rate, benchmark, max_adjustment = sidebar_settings()
    tab_analyse, tab_accuracy, tab_about = st.tabs(["📊 Analyse reviews", "✅ Accuracy check", "ℹ️ About"])
    with tab_analyse:
        analysis_tab(base_rate, benchmark, max_adjustment)
    with tab_accuracy:
        accuracy_tab()
    with tab_about:
        about_tab()


if __name__ == "__main__":
    main()
