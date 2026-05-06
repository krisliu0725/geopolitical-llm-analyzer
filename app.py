"""Geopolitical LLM Response Analyzer — V2.0

Tabs: Analyze | Experiment | Statistics | Settings
Dual scoring system: TRS (T1-T5) + GBS (D1-D5) — 10 sub-dimensions
Internationalized: 中文 / English
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload

from config import (
    SCORER_PROVIDERS, PROMPT_CATEGORIES, MODEL_ALIGNMENTS,
    TRS_DIMENSION_LABELS, GBS_DIMENSION_LABELS,
    TRS_DIMENSION_SHORT, BIAS_DIMENSION_SHORT,
)
from database.connection import init_db, get_session
from database.models import Prompt, ModelConfig, Analysis
from i18n import t as tx
from llm_clients import get_client
from llm_clients.rate_limiter import RateLimiter
from scoring.ai_scorer import score_response
from scoring.statistics import (
    analyses_to_dataframe, descriptive_stats_trs, descriptive_stats_bias,
    per_model_bias_profile,
)
from visualization.bar_charts import mean_trs_by_model, mean_bias_by_model_grouped
from visualization.radar_charts import bias_radar_per_model, bias_radar_by_alignment
from visualization.heatmaps import bias_heatmap, model_comparison_heatmap
from visualization.line_charts import dimension_profile_lines, trs_dimension_profile_lines
from visualization.distribution_charts import (
    trs_box_plot, dimension_violin_plot, density_histogram,
    alignment_pie_chart, category_pie_chart,
)
from export.excel_exporter import generate_workbook, generate_raw_csv

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Geopolitical LLM Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — warm cream Claude-style ──────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; color: hsl(22,10%,15%); }
.stApp { background: hsl(38,25%,97%); }
section.main > div { padding-top: 0.25rem; }

h2 { font-size: 1.3rem !important; font-weight: 650 !important; color: hsl(22,10%,15%) !important; letter-spacing: -0.02em !important; margin-top: 1.5rem !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: hsl(22,10%,15%) !important; }
h4 { font-size: 0.9rem !important; font-weight: 600 !important; color: hsl(22,8%,45%) !important; text-transform: uppercase; letter-spacing: 0.04em !important; }

/* ── Cards ── */
.card {
    background: hsl(38,30%,99%); border: 1px solid hsl(38,16%,92%); border-radius: 16px;
    padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.card-header {
    font-size: 0.78rem; font-weight: 600; color: hsl(22,8%,45%);
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem;
}

/* ── Score badges ── */
.score-badge { display:inline-block; padding:0.18rem 0.7rem; border-radius:999px; font-weight:600; font-size:0.9rem; }
.pro-west   { background:hsl(218,60%,93%); color:hsl(218,55%,38%); }
.mild-west  { background:hsl(218,40%,95%); color:hsl(218,45%,50%); }
.pro-china  { background:hsl(6,60%,93%);  color:hsl(6,55%,38%); }
.mild-china { background:hsl(6,40%,95%);  color:hsl(6,45%,50%); }
.neutral    { background:hsl(155,45%,92%); color:hsl(155,45%,32%); }
.trs-high   { background:hsl(155,40%,92%); color:hsl(155,40%,30%); }
.trs-mid    { background:hsl(38,45%,90%);  color:hsl(30,55%,32%); }
.trs-low    { background:hsl(6,45%,93%);   color:hsl(6,45%,35%); }

/* ── Metric cards ── */
.metric-card { background:hsl(38,30%,99%); border:1px solid hsl(38,16%,92%); border-radius:10px; padding:1.1rem 1.25rem; text-align:center; box-shadow:0 1px 2px rgba(0,0,0,0.03); }
.metric-value { font-size:1.6rem; font-weight:700; color:hsl(22,10%,15%); }
.metric-label { font-size:0.75rem; color:hsl(22,6%,60%); margin-top:0.2rem; font-weight:500; }

/* ── Buttons ── */
.stButton > button {
    border-radius:10px !important; font-weight:500 !important; font-size:0.88rem !important;
    padding:0.45rem 1rem !important; transition:all 0.15s ease !important;
    border:1px solid hsl(38,15%,88%) !important; background:hsl(38,30%,99%) !important; color:hsl(22,10%,15%) !important;
    box-shadow:0 1px 2px rgba(0,0,0,0.03) !important;
}
.stButton > button:hover { background:hsl(38,20%,95%) !important; border-color:hsl(38,15%,80%) !important; }
.stButton > button[kind="primary"] { background:hsl(22,68%,48%) !important; color:#fff !important; border-color:hsl(22,68%,48%) !important; font-weight:550 !important; }
.stButton > button[kind="primary"]:hover { background:hsl(22,68%,42%) !important; }

/* ── Inputs ── */
.stSelectbox > div > div, .stTextInput > div > div, .stTextArea > div > div {
    border-radius:10px !important; border-color:hsl(38,15%,88%) !important; background:hsl(38,30%,99%) !important;
}
.stSelectbox > div > div:focus-within, .stTextInput > div > div:focus-within, .stTextArea > div > div:focus-within {
    border-color:hsl(22,68%,48%) !important; box-shadow:0 0 0 2px hsla(22,68%,48%,0.12) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap:2px; background:hsl(38,18%,92%); border-radius:10px; padding:3px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:0.45rem 1.25rem; font-weight:500; font-size:0.88rem; color:hsl(22,8%,45%); background:transparent; border:none; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background:hsl(38,30%,99%); color:hsl(22,68%,48%); box-shadow:0 1px 2px rgba(0,0,0,0.03); }

/* ── Expanders ── */
.streamlit-expanderHeader { border-radius:10px !important; border:1px solid hsl(38,16%,92%) !important; background:hsl(38,30%,99%) !important; font-weight:500 !important; box-shadow:0 1px 2px rgba(0,0,0,0.03) !important; }

/* ── Dataframes / Tables ── */
div[data-testid="stDataFrame"] { border:1px solid hsl(38,16%,92%); border-radius:10px; overflow:hidden; }
div[data-testid="stDataFrame"] th { background:hsl(38,18%,93%) !important; color:hsl(22,8%,45%) !important; font-weight:600 !important; font-size:0.76rem !important; text-transform:uppercase; letter-spacing:0.04em; }
div[data-testid="stDataFrame"] td { font-size:0.88rem; border-bottom:1px solid hsl(38,16%,92%) !important; }

div[data-testid="stTable"] table { border:1px solid hsl(38,16%,92%); border-radius:10px; overflow:hidden; border-collapse:separate; border-spacing:0; }
div[data-testid="stTable"] th { background:hsl(38,18%,93%); color:hsl(22,8%,45%); font-weight:600; font-size:0.76rem; text-transform:uppercase; letter-spacing:0.04em; border-bottom:1px solid hsl(38,15%,88%); padding:0.55rem 0.75rem; }
div[data-testid="stTable"] td { border-bottom:1px solid hsl(38,16%,92%); padding:0.45rem 0.75rem; font-size:0.88rem; }

div[data-testid="stAlert"] { border-radius:10px !important; border:1px solid hsl(38,16%,92%) !important; }

/* ── Language toggle ── */
.lang-toggle { display:flex; justify-content:flex-end; align-items:center; gap:0.5rem; margin-bottom:0.5rem; }
.lang-toggle span { font-size:0.8rem; color:hsl(22,6%,60%); }

/* ── Misc ── */
#MainMenu { visibility:hidden; } footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent !important; }
hr { border-color:hsl(38,16%,92%) !important; }
.stCaption { color:hsl(22,6%,60%) !important; }
</style>
""", unsafe_allow_html=True)

# ── DB init ────────────────────────────────────────────────────────────────

@st.cache_resource
def _init_db_once():
    init_db()

_init_db_once()

# ── Session state ──────────────────────────────────────────────────────────

_defaults = {
    "lang": "zh",
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    "custom_api_key": os.getenv("CUSTOM_API_KEY", ""),
    "custom_base_url": os.getenv("CUSTOM_BASE_URL", ""),
    "scoring_result": None,
    "scoring_error": None,
    "scoring_latency": None,
    "exp_filter_model": "All",
    "exp_filter_alignment": "All",
    "exp_filter_category": "All",
    "exp_filter_prompt": "All",
    "exp_detail_id": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

L = lambda k: tx(k, st.session_state.lang)

# ── Language switcher ──────────────────────────────────────────────────────

lc1, lc2 = st.columns([6, 1])
with lc2:
    lang_labels = {"zh": "中文", "en": "English"}
    current_lang = st.session_state.lang
    new_lang = "en" if current_lang == "zh" else "zh"
    if st.button(lang_labels[new_lang], key="lang_switcher", help=f"Switch to {lang_labels[new_lang]}"):
        st.session_state.lang = new_lang
        st.rerun()

# ── Cached data loaders ────────────────────────────────────────────────────

@st.cache_data(ttl=3)
def _load_prompts():
    s = get_session()
    try:
        return [{"id": p.id, "short_name": p.short_name, "full_text": p.full_text, "category": p.category}
                for p in s.query(Prompt).order_by(Prompt.created_at.desc()).all()]
    finally:
        s.close()


@st.cache_data(ttl=3)
def _load_models():
    s = get_session()
    try:
        return [{"id": m.id, "provider": m.provider, "model_name": m.model_name,
                 "display_name": m.display_name, "alignment": m.alignment}
                for m in s.query(ModelConfig).filter(ModelConfig.is_active.is_(True))
                .order_by(ModelConfig.display_name).all()]
    finally:
        s.close()


def _load_analyses_with_eager():
    """Query analyses with eager-loaded prompt + model_config to avoid DetachedInstanceError."""
    s = get_session()
    try:
        return s.query(Analysis).options(
            joinedload(Analysis.prompt), joinedload(Analysis.model_config)
        ).order_by(Analysis.analyzed_at.desc()).all(), s
    except Exception:
        s.close()
        raise


@st.cache_data(ttl=3)
def _load_analyses_df():
    analyses, s = _load_analyses_with_eager()
    try:
        prompts = s.query(Prompt).all()
        models = s.query(ModelConfig).all()
        df = analyses_to_dataframe(analyses)
        prompts_df = pd.DataFrame([{"id": p.id, "short_name": p.short_name, "full_text": p.full_text,
                                    "category": p.category, "created_at": p.created_at} for p in prompts])
        models_df = pd.DataFrame([{"id": m.id, "display_name": m.display_name, "provider": m.provider,
                                   "model_name": m.model_name, "alignment": m.alignment, "is_active": m.is_active}
                                  for m in models])
        return df, prompts_df, models_df
    finally:
        s.close()


def _get_analysis_detail(aid: int):
    """Get a single analysis with eager-loaded relationships."""
    s = get_session()
    try:
        return s.query(Analysis).options(
            joinedload(Analysis.prompt), joinedload(Analysis.model_config)
        ).get(aid)
    finally:
        s.close()


def _clear_caches():
    _load_prompts.clear()
    _load_models.clear()
    _load_analyses_df.clear()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _score_class(score: float, dim: str = "bias") -> str:
    if dim == "trs":
        if score >= 4.0: return "trs-high"
        if score >= 3.0: return "trs-mid"
        return "trs-low"
    if score <= 2.49:
        return "pro-west" if score <= 1.99 else "mild-west"
    if score >= 3.51:
        return "pro-china" if score >= 4.01 else "mild-china"
    return "neutral"


def _save_analysis(prompt_mode, prompt_short, prompt_text, prompt_category,
                   selected_prompt, selected_model, response_text, notes,
                   scorer_provider, scorer_model, result) -> bool:
    s = get_session()
    try:
        if prompt_mode == "new" and prompt_short:
            existing = s.query(Prompt).filter_by(short_name=prompt_short).first()
            if existing:
                saved_prompt = existing
            else:
                saved_prompt = Prompt(short_name=prompt_short, full_text=prompt_text, category=prompt_category)
                s.add(saved_prompt)
                s.flush()
        elif selected_prompt:
            saved_prompt = s.query(Prompt).get(selected_prompt["id"])
        else:
            return False

        s.add(Analysis(
            prompt_id=saved_prompt.id, model_config_id=selected_model["id"],
            response_text=response_text,
            # TRS sub-dimensions
            t1_score=result["t1_score"], t2_score=result["t2_score"],
            t3_score=result["t3_score"], t4_score=result["t4_score"],
            t5_score=result["t5_score"],
            # GBS sub-dimensions
            d1_score=result["d1_score"], d2_score=result["d2_score"],
            d3_score=result["d3_score"], d4_score=result["d4_score"],
            d5_score=result["d5_score"],
            key_phrases=json.dumps(result.get("key_phrases", []), ensure_ascii=False),
            notes=notes, scorer_model=f"{scorer_provider}/{scorer_model}",
        ))
        s.commit()
        _clear_caches()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ── Tabs ────────────────────────────────────────────────────────────────────

t1, t2, t3, t4 = st.tabs([
    f"🔬 {L('tab_analyze')}",
    f"🧪 {L('tab_experiment')}",
    f"📈 {L('tab_statistics')}",
    f"⚙️ {L('tab_settings')}",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE
# ═════════════════════════════════════════════════════════════════════════════

with t1:
    st.markdown(f"## 🔬 {L('analyze_title')}")
    st.caption(L("analyze_desc"))

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f'<div class="card"><div class="card-header">📋 {L("question_section")}</div>', unsafe_allow_html=True)

        prompts_data = _load_prompts()
        prompt_mode = st.radio(
            L("question_source"),
            options=["existing", "new"],
            format_func=lambda v: L("q_select_existing") if v == "existing" else L("q_create_new"),
            horizontal=True, label_visibility="collapsed", key="pmode",
        )

        selected_prompt = None
        prompt_short = ""
        prompt_text = ""
        prompt_category = "general"

        if prompt_mode == "existing" and prompts_data:
            prompt_opts = {f"{p['short_name']} [{p['category']}]": p for p in prompts_data}
            sel = st.selectbox(L("q_choose"), list(prompt_opts.keys()), label_visibility="collapsed")
            selected_prompt = prompt_opts.get(sel)
            if selected_prompt:
                prompt_text = selected_prompt["full_text"]
                prompt_category = selected_prompt["category"]
            st.text_area(L("q_text"), value=prompt_text, height=100, disabled=True, key="ept")
            st.caption(f"{L('q_category_label')}: **{prompt_category}**")
        else:
            prompt_short = st.text_input(L("q_short_name"), placeholder="e.g., 台海问题 — 主权", key="npn")
            prompt_category = st.selectbox(L("q_category"), PROMPT_CATEGORIES, key="npcat")
            prompt_text = st.text_area(L("q_full_text"), height=120, placeholder="输入完整问题...", key="npt")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="card"><div class="card-header">💬 {L("response_section")}</div>', unsafe_allow_html=True)

        models_data = _load_models()
        model_opts = {m["display_name"]: m for m in models_data} if models_data else {}
        sel_model_label = st.selectbox(
            L("resp_model_label"),
            list(model_opts.keys()) if model_opts else ["— 未注册模型 —"],
            key="srm",
        )
        selected_model = model_opts.get(sel_model_label)

        response_text = st.text_area(L("resp_text_label"), height=200, placeholder=L("resp_text_placeholder"), key="rti")
        notes = st.text_area(L("resp_notes_label"), height=60, placeholder=L("resp_notes_placeholder"), key="ni")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(f'<div class="card"><div class="card-header">🤖 {L("scoring_section")}</div>', unsafe_allow_html=True)

        scorer_provider = st.selectbox(L("scoring_provider"), SCORER_PROVIDERS, key="sprov")
        model_defaults = {"deepseek": "deepseek-chat", "gemini": "gemini-2.0-flash", "custom": "qwen3.6-plus"}
        scorer_model = st.text_input(L("scoring_model_name"), value=model_defaults.get(scorer_provider, ""), key="smn")
        scorer_base_url = None
        if scorer_provider == "custom":
            scorer_base_url = st.text_input(L("scoring_base_url"), value=st.session_state.custom_base_url,
                                            placeholder="https://api.example.com/v1", key="sbu")

        api_key = st.session_state.get(f"{scorer_provider}_api_key", "")
        if not api_key:
            st.warning(L("scoring_no_key").replace("{provider}", scorer_provider))

        st.markdown("</div>", unsafe_allow_html=True)

        can_run = bool(prompt_text and response_text and api_key and selected_model
                       and (scorer_provider != "custom" or scorer_base_url))

        if st.button(L("run_analysis"), type="primary", use_container_width=True, disabled=not can_run):
            status = st.status(L("scoring_progress"), expanded=True)
            t_start = time.monotonic()
            try:
                if scorer_provider == "custom":
                    client = get_client("custom", api_key, scorer_model, base_url=scorer_base_url)
                else:
                    client = get_client(scorer_provider, api_key, scorer_model)

                status.write(L("scoring_sending"))
                limiter = RateLimiter()
                result = limiter.call_with_retry(score_response, client, prompt_text, response_text)
                elapsed = time.monotonic() - t_start

                if result:
                    st.session_state.scoring_result = result
                    st.session_state.scoring_error = None
                    st.session_state.scoring_latency = elapsed
                    completed_msg = L("scoring_completed").format(elapsed=elapsed)
                    status.update(label=completed_msg, state="complete", expanded=False)
                else:
                    st.session_state.scoring_error = L("scoring_failed")
                    st.session_state.scoring_result = None
                    status.update(label="Failed", state="error", expanded=False)
            except Exception as exc:
                elapsed = time.monotonic() - t_start
                st.session_state.scoring_error = L("scoring_error").format(elapsed=elapsed, error=exc)
                st.session_state.scoring_result = None
                status.update(label="Error", state="error", expanded=False)

        # ── Results ──
        if st.session_state.scoring_error:
            st.error(st.session_state.scoring_error)

        result = st.session_state.scoring_result
        if result:
            # Compute composite scores
            trs_val = round(sum(result[f"t{i}_score"] for i in range(1, 6)) / 5, 2)
            gbs_val = round(sum(result[f"d{i}_score"] for i in range(1, 6)) / 5, 2)

            st.markdown("---")
            st.markdown(f"### 📊 {L('results_title')}")
            if st.session_state.scoring_latency:
                st.caption(L("scored_in").format(elapsed=st.session_state.scoring_latency))

            # ── TRS Card ──
            trs_class = _score_class(trs_val, "trs")
            st.markdown(f"""
            <div class="card" style="border-left:3px solid hsl(22,68%,48%);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                    <span style="font-weight:650;font-size:1rem;">{L('trs_card_title')}</span>
                    <span class="score-badge {trs_class}" style="font-size:1.1rem;">{trs_val:.2f} / 5.00</span>
                </div>
                <p style="color:hsl(22,6%,60%);font-size:0.78rem;margin:0;">{L('trs_formula')}</p>
            </div>
            """, unsafe_allow_html=True)

            # T1-T5 mini cards
            st.markdown(f"**{L('trs_sub_dims')}**")
            tcols = st.columns(5)
            for i in range(1, 6):
                val = result[f"t{i}_score"]
                with tcols[i - 1]:
                    st.markdown(f"""
                    <div class="card" style="text-align:center;padding:0.8rem 0.4rem;">
                        <div style="font-size:0.72rem;color:hsl(22,6%,60%);margin-bottom:0.3rem;font-weight:600;">{L(f't{i}_label')}</div>
                        <span style="font-weight:700;font-size:1.0rem;">{val:.2f}</span>
                        <div style="font-size:0.65rem;color:hsl(22,6%,60%);margin-top:0.2rem;">{L(f't{i}_short')}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── GBS Card ──
            gbs_class = _score_class(gbs_val)
            st.markdown(f"""
            <div class="card" style="border-left:3px solid hsl(218,45%,48%);margin-top:0.8rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                    <span style="font-weight:650;font-size:1rem;">{L('gbs_card_title')}</span>
                    <span class="score-badge {gbs_class}" style="font-size:1.1rem;">{gbs_val:.2f} / 5.00</span>
                </div>
                <p style="color:hsl(22,6%,60%);font-size:0.78rem;margin:0;">{L('gbs_formula')} · <3=Pro-West &nbsp; 3=Neutral &nbsp; >3=Pro-China</p>
            </div>
            """, unsafe_allow_html=True)

            # D1-D5 mini cards
            st.markdown(f"**{L('gbs_sub_dims')}**")
            dcols = st.columns(5)
            for i in range(1, 6):
                val = result[f"d{i}_score"]
                bc = _score_class(val)
                with dcols[i - 1]:
                    st.markdown(f"""
                    <div class="card" style="text-align:center;padding:0.8rem 0.4rem;">
                        <div style="font-size:0.72rem;color:hsl(22,6%,60%);margin-bottom:0.3rem;font-weight:600;">{L(f'd{i}_label')}</div>
                        <span class="score-badge {bc}">{val:.2f}</span>
                        <div style="font-size:0.65rem;color:hsl(22,6%,60%);margin-top:0.2rem;">{L(f'd{i}_short')}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Reasoning expander
            with st.expander(L("dim_reasoning_title")):
                st.markdown("**TRS**")
                for i in range(1, 6):
                    rkey = f"t{i}_reasoning"
                    st.markdown(f"**T{i}** — {result.get(rkey, '—')}")
                st.markdown("**GBS**")
                for i in range(1, 6):
                    rkey = f"d{i}_reasoning"
                    st.markdown(f"**D{i}** — {result.get(rkey, '—')}")

            # Key phrases
            if result.get("key_phrases"):
                st.markdown(f"### 🔑 {L('key_phrases_title')}")
                kp_rows = [{L("kp_dim"): kp.get("dimension", ""), L("kp_quote"): kp.get("phrase", ""),
                            L("kp_annotation"): kp.get("annotation", "")} for kp in result["key_phrases"]]
                st.table(pd.DataFrame(kp_rows))

            st.markdown("---")
            sc1, sc2 = st.columns([1, 1])
            with sc1:
                if st.button(L("save_to_db"), type="primary", use_container_width=True):
                    try:
                        ok = _save_analysis(prompt_mode, prompt_short, prompt_text, prompt_category,
                                            selected_prompt, selected_model, response_text, notes,
                                            scorer_provider, scorer_model, result)
                        if ok:
                            st.success(L("saved_ok"))
                            st.session_state.scoring_result = None
                            st.rerun()
                        else:
                            st.error(L("save_no_question"))
                    except Exception as exc:
                        st.error(L("save_failed").format(error=exc))
            with sc2:
                if st.button(L("clear_results"), use_container_width=True):
                    st.session_state.scoring_result = None
                    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════════

with t2:
    st.markdown(f"## 🧪 {L('exp_title')}")
    st.caption(L("exp_desc"))

    df_all, _, _ = _load_analyses_df()

    if df_all.empty:
        st.info(L("no_analyses"))
    else:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.selectbox(L("filter_model"), [L("filter_all")] + sorted(df_all["model_name"].unique().tolist()), key="exp_filter_model")
            with fc2:
                st.selectbox(L("filter_alignment"), [L("filter_all")] + sorted(df_all["alignment"].unique().tolist()), key="exp_filter_alignment")
            with fc3:
                st.selectbox(L("filter_category"), [L("filter_all")] + sorted(df_all["prompt_category"].unique().tolist()), key="exp_filter_category")
            with fc4:
                st.selectbox(L("filter_question"), [L("filter_all")] + sorted(df_all["prompt_name"].unique().tolist()), key="exp_filter_prompt")
            st.markdown("</div>", unsafe_allow_html=True)

        mask = pd.Series(True, index=df_all.index)
        for col, key in [("model_name", "exp_filter_model"), ("alignment", "exp_filter_alignment"),
                         ("prompt_category", "exp_filter_category"), ("prompt_name", "exp_filter_prompt")]:
            val = st.session_state[key]
            if val != L("filter_all"):
                mask &= df_all[col] == val
        df_filt = df_all[mask]
        st.caption(L("showing_n").format(n=len(df_filt), total=len(df_all)))

        with st.expander(L("comparison_title"), expanded=False):
            metric_choice = st.radio(L("comparison_metric"),
                                     [L("metric_trs"), L("metric_gbs"),
                                      L("metric_d1"), L("metric_d2"), L("metric_d3"),
                                      L("metric_d4"), L("metric_d5")],
                                     horizontal=True, key="comp_metric")
            metric_map = {
                L("metric_trs"): "trs", L("metric_gbs"): "gbs",
                L("metric_d1"): "d1_score", L("metric_d2"): "d2_score",
                L("metric_d3"): "d3_score", L("metric_d4"): "d4_score",
                L("metric_d5"): "d5_score",
            }
            metric_col = metric_map[metric_choice]
            pivot = df_filt.pivot_table(index="prompt_name", columns="model_name", values=metric_col, aggfunc="mean")
            if not pivot.empty:
                st.table(pivot.map(lambda x: f"{x:.2f}" if pd.notna(x) else "—"))

        if st.session_state.exp_detail_id:
            detail = _get_analysis_detail(st.session_state.exp_detail_id)
            if detail:
                st.markdown("---")
                st.markdown(f"### {L('detail_title')}")
                dc1, dc2, dc3 = st.columns(3)
                with dc1: st.metric(L("detail_trs"), f"{detail.trs:.2f}")
                with dc2: st.metric(L("detail_gbs"), f"{detail.gbs:.2f}")
                with dc3: st.metric(L("detail_scorer"), detail.scorer_model or "N/A")

                st.markdown(f"**TRS Sub-Dimensions**")
                tdc = st.columns(5)
                for i in range(1, 6):
                    with tdc[i - 1]: st.metric(f"T{i}", f"{getattr(detail, f't{i}_score'):.2f}")

                st.markdown(f"**GBS Sub-Dimensions**")
                ddc = st.columns(5)
                for i in range(1, 6):
                    with ddc[i - 1]: st.metric(f"D{i}", f"{getattr(detail, f'd{i}_score'):.2f}")

                if detail.prompt:
                    st.markdown(f"**{L('detail_prompt')}:** {detail.prompt.full_text}")
                st.markdown(f"**{L('detail_response')}:**")
                st.text_area(L("detail_response"), detail.response_text, height=200, disabled=True, key="dresp")
                if detail.key_phrases:
                    st.markdown(f"**{L('detail_key_phrases')}:**")
                    try:
                        kps = json.loads(detail.key_phrases)
                    except Exception:
                        kps = []
                    for kp in kps:
                        st.markdown(f"- **[{kp.get('dimension', '')}]** 「{kp.get('phrase', '')}」 — *{kp.get('annotation', '')}*")
                if detail.notes:
                    st.caption(f"{L('detail_notes')}: {detail.notes}")
                if st.button(L("detail_close")):
                    st.session_state.exp_detail_id = None
                    st.rerun()

        st.markdown(f"### {L('all_analyses_title')}")
        display = df_filt[["analysis_id", "prompt_name", "prompt_category", "model_name",
                           "alignment", "trs", "d1_score", "d2_score", "d3_score", "d4_score",
                           "d5_score", "gbs", "scorer_model", "analyzed_at"]].copy()
        display.columns = [L("col_id"), L("col_question"), L("col_cat"), L("col_model"),
                           L("col_align"), L("col_trs"), L("col_d1"), L("col_d2"), L("col_d3"),
                           L("col_d4"), L("col_d5"), L("col_gbs"), L("col_scorer"), L("col_date")]
        for c in [L("col_trs"), L("col_d1"), L("col_d2"), L("col_d3"), L("col_d4"), L("col_d5"), L("col_gbs")]:
            display[c] = display[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        display[L("col_date")] = display[L("col_date")].apply(lambda x: str(x)[:10] if pd.notna(x) else "")

        event = st.dataframe(display, use_container_width=True, hide_index=True,
                             height=min(400, 35 * len(display) + 38),
                             on_select="rerun", selection_mode="single-row", key="exp_df")
        sel_rows = event.get("selection", {}).get("rows", []) if event else []
        if sel_rows:
            sel_aid = int(display.iloc[sel_rows[0]][L("col_id")])
            ab1, ab2, _ = st.columns([1, 1, 2])
            with ab1:
                if st.button(f"🔍 {L('btn_view')}", use_container_width=True):
                    st.session_state.exp_detail_id = sel_aid
                    st.rerun()
            with ab2:
                if st.button(f"🗑 {L('btn_delete')}", use_container_width=True):
                    s = get_session()
                    s.query(Analysis).filter_by(id=sel_aid).delete()
                    s.commit()
                    s.close()
                    _clear_caches()
                    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — STATISTICS
# ═════════════════════════════════════════════════════════════════════════════

with t3:
    st.markdown(f"## 📈 {L('stats_title')}")
    st.caption(L("stats_desc"))

    df, prompts_df, models_df = _load_analyses_df()

    if df.empty:
        st.info(L("no_analyses_stats"))
    else:
        # ── Data Filters ──
        with st.expander(L("viz_filter_title"), expanded=True):
            fc1, fc2 = st.columns(2)
            with fc1:
                all_prompts = sorted(df["prompt_name"].unique().tolist())
                selected_prompts = st.multiselect(
                    L("viz_filter_prompts"), options=all_prompts,
                    default=all_prompts, key="stat_filter_prompts",
                )
            with fc2:
                all_alignments = sorted(df["alignment"].unique().tolist())
                selected_alignments = st.multiselect(
                    L("viz_filter_alignments"), options=all_alignments,
                    default=all_alignments, key="stat_filter_alignments",
                )
        mask = pd.Series(True, index=df.index)
        if selected_prompts:
            mask &= df["prompt_name"].isin(selected_prompts)
        if selected_alignments:
            mask &= df["alignment"].isin(selected_alignments)
        df_filt = df[mask]
        st.caption(L("showing_n").format(n=len(df_filt), total=len(df)))

        # ── Overview ──
        st.markdown(f"#### {L('overview_title')}")
        us_n = int((df_filt["alignment"] == "US/Western").sum())
        cn_n = int((df_filt["alignment"] == "China/Non-Western").sum())
        cards = [
            (str(len(df_filt)), L("card_total")),
            (str(df_filt["prompt_name"].nunique()), L("card_questions")),
            (str(df_filt["model_name"].nunique()), L("card_models")),
            (f"{df_filt['trs'].mean():.2f}", L("card_avg_trs")),
            (f"{df_filt['gbs'].mean():.2f}", L("card_avg_gbs")),
            (f"🔵{us_n}  🔴{cn_n}", L("card_us_cn")),
        ]
        for mc, (val, label) in zip(st.columns(6), cards):
            with mc:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown(f"#### {L('trs_stats_title')}")
        st.table(descriptive_stats_trs(df_filt))

        st.markdown(f"#### {L('bias_stats_title')}")
        st.table(descriptive_stats_bias(df_filt))

        st.markdown(f"#### {L('profile_title')}")
        st.table(per_model_bias_profile(df_filt))

        # ── Basic Charts ──
        st.markdown("---")
        st.markdown(f"### 📊 {L('viz_title')}")

        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(mean_trs_by_model(descriptive_stats_trs(df_filt)), use_container_width=True)
        with c2: st.plotly_chart(mean_bias_by_model_grouped(descriptive_stats_bias(df_filt)), use_container_width=True)
        st.plotly_chart(bias_radar_per_model(per_model_bias_profile(df_filt)), use_container_width=True)

        hc1, hc2 = st.columns(2)
        with hc1: st.plotly_chart(bias_heatmap(per_model_bias_profile(df_filt)), use_container_width=True)
        with hc2:
            sd_options = ["trs", "gbs", "d1_score", "d2_score", "d3_score", "d4_score", "d5_score"]
            sd_labels = {"trs": "TRS", "gbs": "GBS", "d1_score": "D1", "d2_score": "D2",
                        "d3_score": "D3", "d4_score": "D4", "d5_score": "D5"}
            dim_pick = st.selectbox(L("viz_dim_choice"), sd_options,
                                    format_func=lambda x: sd_labels[x], key="stat_dim")
            st.plotly_chart(model_comparison_heatmap(df_filt, dim_pick), use_container_width=True)

        profile = per_model_bias_profile(df_filt)
        if "Alignment" in profile.columns and profile["Alignment"].nunique() > 1:
            st.plotly_chart(bias_radar_by_alignment(profile), use_container_width=True)

        # ── Advanced Charts ──
        st.markdown("---")
        st.markdown(f"### {L('viz_advanced_title')}")

        # Row 1: Line charts
        st.markdown(f"#### {L('viz_line_title')}")
        lc1, lc2 = st.columns(2)
        with lc1:
            st.plotly_chart(dimension_profile_lines(df_filt), use_container_width=True)
        with lc2:
            st.plotly_chart(trs_dimension_profile_lines(df_filt), use_container_width=True)

        # Row 2: Box + Violin
        st.markdown(f"#### {L('viz_distribution_title')}")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.plotly_chart(trs_box_plot(df_filt), use_container_width=True)
        with dc2:
            violin_dim = st.selectbox(
                L("viz_violin_dim"),
                ["gbs", "d1_score", "d2_score", "d3_score", "d4_score", "d5_score"],
                format_func=lambda x: {
                    "gbs": "GBS", "d1_score": "D1", "d2_score": "D2",
                    "d3_score": "D3", "d4_score": "D4", "d5_score": "D5",
                }[x],
                key="stat_violin_dim",
            )
            st.plotly_chart(dimension_violin_plot(df_filt, violin_dim), use_container_width=True)

        # Row 3: Density + Pie
        ec1, ec2 = st.columns(2)
        with ec1:
            density_dim = st.selectbox(
                L("viz_density_dim"),
                ["trs", "gbs"],
                format_func=lambda x: {"trs": "TRS", "gbs": "GBS"}[x],
                key="stat_density_dim",
            )
            st.plotly_chart(density_histogram(df_filt, density_dim), use_container_width=True)
        with ec2:
            pie_type = st.radio(
                L("viz_pie_type"),
                ["alignment", "category"],
                format_func=lambda x: L(f"viz_pie_{x}"),
                horizontal=True,
                key="stat_pie_type",
            )
            if pie_type == "alignment":
                st.plotly_chart(alignment_pie_chart(df_filt), use_container_width=True)
            else:
                st.plotly_chart(category_pie_chart(df_filt), use_container_width=True)

        # ── Export ──
        st.markdown("---")
        st.markdown(f"### 📥 {L('export_title')}")
        ec1, ec2 = st.columns(2)
        with ec1:
            if st.button(L("export_btn"), type="primary", use_container_width=True):
                buf = generate_workbook(analyses_df=df_filt, prompts_df=prompts_df, model_configs_df=models_df)
                st.download_button(L("export_download"), data=buf,
                                   file_name=f"geopolitical_llm_analysis_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="dl_excel")
        with ec2:
            csv_data = generate_raw_csv(df_filt)
            st.download_button(L("export_csv_download"), data=csv_data,
                               file_name=f"geopolitical_llm_raw_data_{datetime.now():%Y%m%d_%H%M%S}.csv",
                               mime="text/csv", key="dl_csv")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

with t4:
    st.markdown(f"## ⚙️ {L('settings_title')}")

    set_col1, set_col2 = st.columns([1, 1], gap="large")

    with set_col1:
        st.markdown(f'<div class="card"><div class="card-header">🔑 {L("api_keys_title")}</div>', unsafe_allow_html=True)

        st.session_state.deepseek_api_key = st.text_input(
            L("deepseek_key"), type="password", value=st.session_state.deepseek_api_key, placeholder="sk-...", key="set_dsk")
        st.caption(L("deepseek_hint"))

        st.session_state.gemini_api_key = st.text_input(
            L("gemini_key"), type="password", value=st.session_state.gemini_api_key, placeholder="AIza...", key="set_gk")
        st.caption(L("gemini_hint"))

        st.session_state.custom_api_key = st.text_input(
            L("custom_key"), type="password", value=st.session_state.custom_api_key, placeholder="sk-...", key="set_ck")
        st.session_state.custom_base_url = st.text_input(
            L("custom_url"), value=st.session_state.custom_base_url, placeholder="https://api.example.com/v1", key="set_cu")
        st.caption(L("custom_hint"))

        st.markdown("</div>", unsafe_allow_html=True)

    with set_col2:
        st.markdown(f'<div class="card"><div class="card-header">🤖 {L("model_registry_title")}</div>', unsafe_allow_html=True)

        with st.form("settings_add_model"):
            ap1, ap2 = st.columns(2)
            with ap1:
                add_provider = st.selectbox(L("add_provider"), ["deepseek", "gemini", "custom"])
                add_display = st.text_input(L("add_display_name"), placeholder="e.g., GPT-4o (US)")
            with ap2:
                add_api_model = st.text_input(L("add_api_model"), placeholder="e.g., gpt-4o")
                add_align = st.selectbox(L("add_alignment"), MODEL_ALIGNMENTS)
            if st.form_submit_button(L("add_model_btn"), type="primary", use_container_width=True):
                if add_display and add_api_model:
                    s = get_session()
                    s.add(ModelConfig(provider=add_provider, model_name=add_api_model,
                                      display_name=add_display, alignment=add_align))
                    s.commit()
                    s.close()
                    _clear_caches()
                    st.success(L("add_model_success").format(name=add_display))
                    st.rerun()
                else:
                    st.error(L("add_model_error"))

        st.markdown("</div>", unsafe_allow_html=True)

        all_models = _load_models()
        if all_models:
            st.markdown(f"**{L('registered_models')}**")
            for m in all_models:
                icon = {"US/Western": "🔵", "China/Non-Western": "🔴", "Other/Unknown": "⚪"}.get(m["alignment"], "⚪")
                mcol1, mcol2 = st.columns([4, 1])
                with mcol1:
                    st.caption(f"{icon} **{m['display_name']}** · `{m['provider']}/{m['model_name']}` · {m['alignment']}")
                with mcol2:
                    if st.button("🗑", key=f"delmod_{m['id']}", help=f"Delete {m['display_name']}"):
                        s = get_session()
                        s.query(ModelConfig).filter_by(id=m["id"]).delete()
                        s.commit()
                        s.close()
                        _clear_caches()
                        st.rerun()
        else:
            st.caption(L("no_models_registered"))

# ── Hide sidebar ──
with st.sidebar:
    st.empty()
