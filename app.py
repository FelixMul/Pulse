import asyncio
import ast
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_card import card


# Make backend modules importable
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.config import settings  # type: ignore
from app.llm_processor import llm_processor  # type: ignore
from app.simple_storage import storage  # type: ignore

# Data and model configuration
DATASET_PATH = Path("ModelFinetuning/data_creation/emails_data/email_dataset_final.csv")
PREDICTIONS_DIR = Path("data")
MODEL_MAP = {
    "small": "llama3.1:8b-instruct",
    "big": "gpt-oss:20b",
}


def run_coro(coro):
    """Run an async coroutine safely in Streamlit."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    # Ensure required columns exist
    required = ["recipe_id", "subject_line", "email_body", "topics", "sentiment"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df


@st.cache_data(show_spinner=False, ttl=60)
def load_predictions(model_key: str) -> Optional[pd.DataFrame]:
    path = PREDICTIONS_DIR / f"predictions_{model_key}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def parse_topics(topics_str: str) -> List[str]:
    try:
        value = ast.literal_eval(topics_str)
        if isinstance(value, list):
            return [str(x).strip() for x in value]
    except Exception:
        pass
    # Fallbacks: comma separated or single label
    if isinstance(topics_str, str) and "," in topics_str:
        return [t.strip() for t in topics_str.split(",") if t.strip()]
    return [str(topics_str).strip()] if topics_str else []


def normalize_sentiment_dataset(label: str) -> str:
    m = label.strip().lower() if isinstance(label, str) else ""
    mapping = {
        "very negative": "very_negative",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "very positive": "very_positive",
    }
    return mapping.get(m, "neutral")


def topics_match(pred_topic: str, gt_topics: List[str]) -> bool:
    if not pred_topic:
        return False
    pred = pred_topic.strip().lower()
    gt_norm = [t.strip().lower() for t in gt_topics if isinstance(t, str)]
    return pred in gt_norm


@st.cache_data(show_spinner=False, ttl=10)
def _cached_llm_ok(model_tag: str) -> bool:
    # Cache the connectivity check briefly to avoid pinging every rerun
    try:
        return bool(run_coro(llm_processor.test_connection(model_tag=model_tag)))
    except Exception:
        return False


def render_health_sidebar(selected_model_key: str) -> None:
    st.sidebar.header("System")
    try:
        # Check selected model availability
        model_tag = MODEL_MAP.get(selected_model_key, settings.LLM_MODEL)
        llm_ok = _cached_llm_ok(model_tag)
        if llm_ok:
            st.sidebar.success("Ollama connected")
        else:
            st.sidebar.warning("LLM not connected")
    except Exception as e:
        st.sidebar.error(f"Health check failed: {e}")

    st.sidebar.caption(f"Model: {settings.LLM_MODEL}")
    st.sidebar.caption(f"Ollama: {settings.OLLAMA_BASE_URL}")
    st.sidebar.caption(f"Data dir: {settings.DATA_DIR}")


def tab_browser(base_df: pd.DataFrame, preds_df: Optional[pd.DataFrame]) -> None:
    st.subheader("Browse emails")

    # Merge predictions for active model; also compute a union for Aggregates elsewhere
    merged = base_df.copy()
    # Always prefer BIG model predictions for browsing to ensure availability
    _preds_big = load_predictions("big")
    effective_preds = _preds_big if _preds_big is not None else preds_df
    if effective_preds is not None:
        merged = merged.merge(effective_preds, how="left", on="recipe_id")

    # Two-column layout (Outlook style)
    left_col, right_col = st.columns([1, 2], gap="small")

    # Independent scrolling for columns within tabs
    st.markdown(
        """
        <style>
        /* Target the columns specifically within the Streamlit Tabs component's content area */
        div[data-testid="stTabs"] div[data-testid="stVerticalBlock"] {
            height: calc(100vh - 240px);
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Helper to render lean cards: minimal padding around text only
    def render_email_card(rid: str, preview: str, selected: bool,
                         *,
                         id_font_size: str = "0.7rem",
                         preview_font_size: str = "0.74rem") -> bool:
        return card(
            title=f"ID: {rid}",
            text=preview,
            styles={
                "card": {
                    "width": "100%",
                    "padding": "0px !important",
                    "text-align": "left",
                    "margin": "0 !important",
                    "border": "none !important",
                    "background": ("#2E3138" if selected else "transparent !important"),
                },
                "title": {
                    "font-size": id_font_size + " !important",
                    "font-weight": "600 !important",
                    "letter-spacing": "0",
                    "margin": "0 !important",
                    "text-align": "left",
                    "padding": "0px",
                },
                "text": {
                    "font-size": preview_font_size,
                    "overflow": "hidden",
                    "text-align": "left",
                    "margin": "0px",
                    "padding": "0px",
                }
            },
            key=f"card_{rid}",
        )

    with left_col:
        st.markdown("##### Emails")
        # Build a concise list with ID and first words (no subject)
        list_df = merged[["recipe_id", "email_body"]].copy()
        def preview_text(s: str, n: int = 6) -> str:
            if not isinstance(s, str):
                return ""
            words = s.strip().split()
            return " ".join(words[:n]) + ("…" if len(words) > n else "")
        list_df["preview"] = list_df["email_body"].apply(preview_text)
        
        # Maintain selection and incremental loading in session state
        rid_list = list_df["recipe_id"].astype(str).tolist()
        if "selected_rid" not in st.session_state and rid_list:
            st.session_state.selected_rid = rid_list[0]
        if "list_limit" not in st.session_state:
            st.session_state.list_limit = 10

        # Determine slice to display
        total_rows = len(list_df)
        display_count = min(st.session_state.list_limit, total_rows)
        display_df = list_df.head(display_count)

        # Render stacked lean cards; clicking selects the email
        for rid, pv in zip(display_df["recipe_id"].astype(str), display_df["preview"]):
            clicked = render_email_card(rid, pv, selected=(rid == st.session_state.get("selected_rid")))
            if clicked:
                st.session_state.selected_rid = rid

        # Load more control
        if display_count < total_rows:
            if st.button("Load more", key="load_more_emails"):
                st.session_state.list_limit = min(st.session_state.list_limit + 10, total_rows)

    with right_col:
        selected_rid = st.session_state.get("selected_rid")
        if not selected_rid:
            st.info("No email selected.")
            return
        row = merged[merged["recipe_id"].astype(str) == str(selected_rid)].head(1)
        if row.empty:
            st.info("No data for selection.")
            return

        rec = row.iloc[0].to_dict()
        gt_topics = parse_topics(rec.get("topics", ""))
        gt_sent = normalize_sentiment_dataset(rec.get("sentiment", ""))

        # Show predictions at the top, then the email body (no subject)
        st.markdown("#### Prediction")
        pred_topic = rec.get("pred_topic", "")
        pred_sent = rec.get("pred_sentiment", "")
        conf = rec.get("confidence", 0.0)
        summary = rec.get("summary", "")
        
        if pred_topic == "" or pred_sent == "" or pd.isna(pred_topic) or pd.isna(pred_sent):
            st.warning("No precomputed prediction for this model.")
        else:
            # Hoverable badges; always show true labels on hover
            t_match = topics_match(pred_topic, gt_topics)
            s_match = (pred_sent == gt_sent)

            topic_title = f"True: {', '.join(gt_topics) if gt_topics else '-'} — {'Match' if t_match else 'Mismatch'}"
            sent_title = f"True: {gt_sent or '-'} — {'Match' if s_match else 'Mismatch'}"

            topic_badge = (
                f"<span title=\"{topic_title}\" "
                f"style=\"padding:4px 8px;border-radius:6px;background:{'#dcfce7' if t_match else '#fef2f2'};"
                f"color:{'#166534' if t_match else '#991b1b'};\">{pred_topic or '-'} {'✅' if t_match else '❌'}" 
                f"</span>"
            )
            sent_badge = (
                f"<span title=\"{sent_title}\" "
                f"style=\"padding:4px 8px;border-radius:6px;background:{'#dcfce7' if s_match else '#fef2f2'};"
                f"color:{'#166534' if s_match else '#991b1b'};\">{pred_sent or '-'} {'✅' if s_match else '❌'}" 
                f"</span>"
            )

            st.markdown(f"Topic: {topic_badge}", unsafe_allow_html=True)
            st.markdown(f"Sentiment: {sent_badge}", unsafe_allow_html=True)
            try:
                st.caption(f"Confidence: {float(conf)*100:.1f}%")
            except Exception:
                st.caption("Confidence: ?")
            if summary:
                st.write("Summary:", summary)

        st.markdown("#### Email")
        st.write(rec.get("email_body", ""))


def tab_aggregates(base_df: pd.DataFrame, preds_df: Optional[pd.DataFrame]) -> None:
    st.subheader("Aggregate distributions")
    # Sticky toolbar toggle inside tab so it stays visible while scrolling
    st.markdown(
        """
        <style>
        #agg-toolbar { position: sticky; top: 0; z-index: 10; padding: 4px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div id="agg-toolbar">', unsafe_allow_html=True)
    show_true = st.toggle(
        "Show true labels",
        value=bool(st.session_state.get("agg_show_true", False)),
        key="agg_show_true",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # If no predictions for active model, try to load both to show some charts
    merged = None
    if preds_df is not None:
        merged = base_df.merge(preds_df, how="inner", on="recipe_id")
    else:
        # fallback to any available predictions
        small = load_predictions("small")
        big = load_predictions("big")
        if small is not None:
            merged = base_df.merge(small, how="inner", on="recipe_id")
        elif big is not None:
            merged = base_df.merge(big, how="inner", on="recipe_id")

    if merged is None or merged.empty:
        st.info("No predictions available to chart.")
        return

    # ---------- Topic pie(s) ----------
    pred_topic_counts = merged["pred_topic"].value_counts().reset_index()
    pred_topic_counts.columns = ["topic", "count"]

    if not pred_topic_counts.empty:
        # Build true topic counts from ground truth topics on overlapping rows (precomputed)
        true_topics_series = merged["topics"].apply(parse_topics)
        true_topics_exploded = true_topics_series.explode().dropna()
        true_topic_counts = true_topics_exploded.value_counts().reset_index()
        true_topic_counts.columns = ["topic", "count"]

        # Color map anchored to predicted pie order, to keep colors identical across modes
        palette = px.colors.qualitative.Plotly
        pred_order = list(pred_topic_counts["topic"])  # fixed order from predicted
        all_topics = list(dict.fromkeys(pred_order + list(true_topic_counts["topic"])) )
        color_map = {t: palette[i % len(palette)] for i, t in enumerate(pred_order)}
        # Assign fallback colors for topics seen only in true
        extra_start = len(pred_order)
        for j, t in enumerate(all_topics[len(pred_order):], start=0):
            color_map[t] = palette[(extra_start + j) % len(palette)]

        # Dynamic height so legend never scrolls
        est_height = max(500, int(24 + 22 * len(all_topics)))

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "domain"}, {"type": "domain"}]],
            subplot_titles=("Predicted", "True"),
            horizontal_spacing=0.2,
        )
        left_pie = go.Pie(
            labels=pred_topic_counts["topic"],
            values=pred_topic_counts["count"],
            textinfo="none",
            hovertemplate="%{label}: %{percent}<extra></extra>",
            marker=dict(colors=[color_map[t] for t in pred_topic_counts["topic"]]),
            name="Predicted",
            showlegend=True,
            legendgroup="topics",
            hole=0.55,
            domain=dict(x=[0.05, 0.45], y=[0.15, 0.85]),
        )
        right_pie = go.Pie(
            labels=true_topic_counts["topic"],
            values=true_topic_counts["count"],
            textinfo="none",
            hovertemplate="%{label}: %{percent}<extra></extra>",
            marker=dict(colors=[color_map[t] for t in true_topic_counts["topic"]]),
            name="True",
            showlegend=False,
            legendgroup="topics",
            hole=0.55,
            domain=dict(x=[0.55, 0.95], y=[0.15, 0.85]),
        )
        fig.add_trace(left_pie, 1, 1)
        fig.add_trace(right_pie, 1, 2)

        # Place a vertical legend centered between pies
        fig.update_layout(
            height=est_height,
            legend=dict(
                orientation="v",
                x=0.5,
                xanchor="center",
                y=0.5,
                yanchor="middle",
                itemsizing="constant",
            ),
            margin=dict(l=10, r=10, t=30, b=10),
        )

        if not show_true:
            # Hide the true pie and center the predicted pie (leave space for legend)
            fig.data[1].visible = False
            # Use full width but with a reduced outer radius via domain margins
            fig.data[0].domain = dict(x=[0.15, 0.85], y=[0.15, 0.85])
            # Remove the 'True' subplot title
            if getattr(fig.layout, "annotations", None) and len(fig.layout.annotations) > 1:
                fig.layout.annotations[1].text = ""
            # Hide legend to maximize plot width
            fig.update_layout(showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    # ---------- Sentiment bars ----------
    desired_order = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    pred_sent_counts = merged["pred_sentiment"].value_counts().reset_index()
    pred_sent_counts.columns = ["sentiment", "count"]
    pred_sent_counts = pd.DataFrame({"sentiment": desired_order}).merge(pred_sent_counts, on="sentiment", how="left").fillna({"count": 0})

    true_sent_series = merged["sentiment"].apply(normalize_sentiment_dataset)
    true_sent_counts = true_sent_series.value_counts().reset_index()
    true_sent_counts.columns = ["sentiment", "count"]
    true_sent_counts = pd.DataFrame({"sentiment": desired_order}).merge(true_sent_counts, on="sentiment", how="left").fillna({"count": 0})

    # Precompute both charts and toggle visibility only
    fig_bar = px.bar(pred_sent_counts, x="sentiment", y="count", category_orders={"sentiment": desired_order}, title="Predicted sentiment")
    fig_bar.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")
    fig_true = px.bar(true_sent_counts, x="sentiment", y="count", category_orders={"sentiment": desired_order}, title="True sentiment", color_discrete_sequence=["#10B981"])  # green for true
    fig_true.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")

    if show_true:
        col_pred, col_true = st.columns(2)
        with col_pred:
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_true:
            st.plotly_chart(fig_true, use_container_width=True)
    else:
        # Center the single bar chart
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.plotly_chart(fig_bar, use_container_width=True)


    # Removed unused helpers for clarity


def tab_stats(base_df: pd.DataFrame, preds_df_small: Optional[pd.DataFrame], preds_df_big: Optional[pd.DataFrame]) -> None:
    st.subheader("Predictions availability")
    total = len(base_df)
    s_count = len(preds_df_small) if preds_df_small is not None else 0
    b_count = len(preds_df_big) if preds_df_big is not None else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset rows", total)
    c2.metric("Small model preds", s_count)
    c3.metric("Big model preds", b_count)


def main() -> None:
    st.set_page_config(page_title="Parliament Pulse", layout="wide")
    st.title("Parliament Pulse")

    # Model selector
    model_key = st.sidebar.selectbox("Model", options=["small", "big"], format_func=lambda k: f"{k} ({MODEL_MAP[k]})")
    # Aggregates toggle pinned in sidebar (sticky)
    st.sidebar.write("")
    st.sidebar.subheader("Aggregates")
    st.session_state["agg_show_true"] = st.sidebar.toggle("Show true labels", value=bool(st.session_state.get("agg_show_true", False)))
    render_health_sidebar(model_key)

    # Load data
    try:
        base_df = load_dataset()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    preds_small = load_predictions("small")
    preds_big = load_predictions("big")

    # Active preds for browser
    active_preds = preds_small if model_key == "small" else preds_big

    tab1, tab2, tab3 = st.tabs(["Browse", "Aggregates", "Stats"])
    with tab1:
        tab_browser(base_df, active_preds)
    with tab2:
        tab_aggregates(base_df, active_preds)
    with tab3:
        tab_stats(base_df, preds_small, preds_big)


if __name__ == "__main__":
    main()


