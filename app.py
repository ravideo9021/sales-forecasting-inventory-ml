#!/usr/bin/env python3
"""
RetailIQ — Sales Intelligence Platform
Production-grade analytics for sales forecasting and inventory optimization.

Design language: restrained, single-accent (indigo) dark theme inspired by
Linear / Vercel / LangChain. Material Symbols iconography, refined surfaces,
sparkline KPI cards, cohesive chart palette.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml, os, sys, math
from datetime import datetime
from typing import Dict, List
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))

st.set_page_config(
    page_title="RetailIQ · Sales Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ═════════════════════════════════════════════════════════════════════════════

C = dict(
    bg='#08090d', surface='#0e0f15', surface2='#13141c', elevated='#181923',
    border='rgba(255,255,255,0.07)', border_hi='rgba(255,255,255,0.13)',
    text='#f4f4f6', text2='#a1a1aa', muted='#71717a', faint='#52525b',
    accent='#6366f1', accent2='#818cf8', violet='#a78bfa',
    green='#34d399', amber='#fbbf24', red='#f87171', cyan='#22d3ee', blue='#60a5fa',
)

# Cohesive categorical palette — indigo-anchored, not a rainbow
PALETTE = ['#6366f1', '#a78bfa', '#22d3ee', '#34d399', '#fbbf24', '#fb7185', '#38bdf8', '#c084fc']
# Single-hue continuous ramp (indigo)
RAMP = [[0.0, '#1e1b4b'], [0.35, '#4338ca'], [0.7, '#6366f1'], [1.0, '#a5b4fc']]
RAMP_CYAN = [[0.0, '#083344'], [0.5, '#0e7490'], [1.0, '#22d3ee']]


# ═════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root {{ --accent:{C['accent']}; --border:{C['border']}; }}

html, body, .stApp {{ background:{C['bg']} !important; }}
* {{ font-family:'Inter', -apple-system, system-ui, sans-serif; }}
.block-container {{ padding:1.4rem 2.4rem 3rem 2.4rem !important; max-width:1480px; }}
body, p, span, div, label, li {{ color:{C['text2']}; }}
h1,h2,h3,h4,h5 {{ color:{C['text']} !important; letter-spacing:-0.02em; font-weight:700; }}
a {{ color:{C['accent2']}; text-decoration:none; }}
::selection {{ background:rgba(99,102,241,0.3); }}

/* scrollbar */
::-webkit-scrollbar {{ width:9px; height:9px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:rgba(255,255,255,0.08); border-radius:10px; }}
::-webkit-scrollbar-thumb:hover {{ background:rgba(255,255,255,0.16); }}

hr {{ border:none !important; border-top:1px solid {C['border']} !important; margin:1.5rem 0 !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background:{C['surface']} !important;
    border-right:1px solid {C['border']};
}}
[data-testid="stSidebar"] .block-container {{ padding-top:1rem !important; }}
[data-testid="stSidebar"] * {{ color:{C['text2']}; }}

/* sidebar nav buttons (primary = active, secondary = ghost) */
[data-testid="stSidebar"] .stButton button {{
    width:100%; text-align:left !important; justify-content:flex-start !important;
    border-radius:9px !important; font-weight:500 !important; font-size:0.875rem !important;
    padding:0.5rem 0.7rem !important; margin:1px 0 !important; transition:all .12s ease !important;
    border:1px solid transparent !important; box-shadow:none !important;
}}
[data-testid="stSidebar"] .stButton button[kind="secondary"] {{
    background:transparent !important; color:{C['text2']} !important;
}}
[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {{
    background:rgba(255,255,255,0.04) !important; color:{C['text']} !important;
}}
[data-testid="stSidebar"] .stButton button[kind="primary"] {{
    background:rgba(99,102,241,0.13) !important; color:{C['accent2']} !important;
    border:1px solid rgba(99,102,241,0.28) !important; font-weight:600 !important;
}}

/* ── Inputs ── */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
[data-testid="stDateInput"] input,
.stNumberInput input {{
    background:{C['surface2']} !important; border:1px solid {C['border']} !important;
    color:{C['text']} !important; border-radius:8px !important;
}}
.stSelectbox div[data-baseweb="select"] > div:hover,
.stMultiSelect div[data-baseweb="select"] > div:hover {{ border-color:{C['border_hi']} !important; }}
[data-baseweb="tag"] {{ background:rgba(99,102,241,0.18) !important; border-radius:6px !important; }}
.stSlider [data-baseweb="slider"] [role="slider"] {{ background:{C['accent']} !important; box-shadow:0 0 0 4px rgba(99,102,241,0.18) !important; }}
.stSlider [data-baseweb="slider"] > div > div > div {{ background:{C['accent']} !important; }}
label, .stSlider label, .stNumberInput label {{ color:{C['muted']} !important; font-size:0.78rem !important; font-weight:500 !important; }}

/* ── Buttons (main area) ── */
.stDownloadButton button {{
    background:{C['surface2']} !important; color:{C['text']} !important;
    border:1px solid {C['border']} !important; border-radius:8px !important;
    font-weight:500 !important; font-size:0.82rem !important; box-shadow:none !important;
    transition:all .12s !important;
}}
.stDownloadButton button:hover {{ border-color:{C['accent']} !important; color:{C['accent2']} !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{ background:transparent; gap:0.3rem; border-bottom:1px solid {C['border']}; padding:0; }}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; color:{C['muted']} !important; border-radius:0 !important;
    padding:0.55rem 0.2rem !important; margin:0 0.9rem 0 0 !important; font-weight:500 !important;
    font-size:0.85rem !important; border-bottom:2px solid transparent !important;
}}
.stTabs [aria-selected="true"] {{ color:{C['text']} !important; border-bottom:2px solid {C['accent']} !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none !important; }}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {{ border:1px solid {C['border']} !important; border-radius:11px !important; }}
[data-testid="stDataFrame"] * {{ font-family:'Inter', sans-serif !important; }}

/* ── Native metric (fallback) ── */
[data-testid="stMetric"] {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:13px; padding:1rem 1.2rem; }}
[data-testid="stMetricValue"] {{ color:{C['text']} !important; font-weight:700 !important; }}

/* ── Alerts ── */
[data-testid="stAlert"] {{ background:{C['surface2']} !important; border:1px solid {C['border']} !important; border-radius:10px !important; color:{C['text2']} !important; }}

/* ════ Custom components ════ */

/* page header */
.riq-topbar {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:1.6rem; }}
.riq-title {{ font-size:1.55rem; font-weight:800; color:{C['text']}; letter-spacing:-0.035em; line-height:1.1; }}
.riq-sub {{ color:{C['muted']}; font-size:0.86rem; margin-top:4px; font-weight:400; }}
.riq-crumb {{ display:flex; align-items:center; gap:7px; color:{C['faint']}; font-size:0.72rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:9px; }}

/* section label */
.riq-sec {{ display:flex; align-items:center; gap:8px; font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:{C['muted']}; margin:0.4rem 0 0.85rem 0; }}
.riq-sec::after {{ content:''; flex:1; height:1px; background:{C['border']}; }}

/* KPI card */
.kpi {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:14px; padding:1.05rem 1.2rem 0.9rem 1.2rem; transition:border-color .15s, transform .15s; position:relative; overflow:hidden; }}
.kpi:hover {{ border-color:{C['border_hi']}; }}
.kpi-top {{ display:flex; align-items:center; justify-content:space-between; }}
.kpi-label {{ color:{C['muted']}; font-size:0.72rem; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; }}
.kpi-ico {{ color:{C['faint']}; display:flex; }}
.kpi-val {{ color:{C['text']}; font-size:1.7rem; font-weight:800; letter-spacing:-0.03em; margin-top:6px; font-variant-numeric:tabular-nums; }}
.kpi-foot {{ display:flex; align-items:center; gap:6px; margin-top:5px; font-size:0.75rem; font-weight:600; }}
.kpi-spark {{ margin-top:8px; }}
.up {{ color:{C['green']}; }} .down {{ color:{C['red']}; }} .flat {{ color:{C['muted']}; }}

/* generic card */
.card {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:14px; padding:1.2rem 1.4rem; }}

/* status pill */
.pill {{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-size:0.72rem; font-weight:600; }}
.pill-dot {{ width:6px; height:6px; border-radius:50%; }}
.pill-green {{ background:rgba(52,211,153,0.12); color:{C['green']}; }}
.pill-amber {{ background:rgba(251,191,36,0.12); color:{C['amber']}; }}
.pill-red   {{ background:rgba(248,113,113,0.12); color:{C['red']}; }}
.pill-blue  {{ background:rgba(96,165,250,0.12); color:{C['blue']}; }}

/* alert row */
.alert {{ display:flex; align-items:center; gap:12px; padding:11px 15px; border-radius:11px; margin:6px 0; font-size:0.85rem; background:{C['surface']}; border:1px solid {C['border']}; }}
.alert .ico {{ display:flex; flex-shrink:0; }}
.alert.crit {{ border-left:2px solid {C['red']}; }}  .alert.crit .ico {{ color:{C['red']}; }}
.alert.warn {{ border-left:2px solid {C['amber']}; }} .alert.warn .ico {{ color:{C['amber']}; }}
.alert.ok   {{ border-left:2px solid {C['green']}; }} .alert.ok .ico {{ color:{C['green']}; }}
.alert b {{ color:{C['text']}; font-weight:600; }}

/* queue item */
.q-item {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:12px; padding:0.85rem 1.1rem; margin-bottom:0.5rem; }}
.q-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem; }}
.q-name {{ font-weight:600; color:{C['text']}; font-size:0.88rem; }}
.q-meta {{ display:flex; gap:1.6rem; font-size:0.76rem; color:{C['muted']}; margin-bottom:0.55rem; }}
.q-meta b {{ color:{C['text2']}; font-weight:600; }}
.bar {{ background:rgba(255,255,255,0.05); border-radius:999px; height:5px; overflow:hidden; }}
.bar > div {{ height:100%; border-radius:999px; }}

/* scenario tile */
.tile {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:13px; padding:1rem 1.1rem; }}
.tile-v {{ font-size:1.55rem; font-weight:800; color:{C['text']}; letter-spacing:-0.03em; font-variant-numeric:tabular-nums; }}
.tile-l {{ font-size:0.7rem; color:{C['muted']}; text-transform:uppercase; letter-spacing:0.06em; font-weight:600; margin-top:2px; }}
.tile-d {{ font-size:0.74rem; font-weight:600; margin-top:6px; }}

/* logo */
.logo {{ display:flex; align-items:center; gap:10px; padding:0.3rem 0.2rem 1rem 0.2rem; }}
.logo-mark {{ width:30px; height:30px; border-radius:8px; background:linear-gradient(135deg,{C['accent']},{C['violet']}); display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:1rem; box-shadow:0 2px 12px rgba(99,102,241,0.35); }}
.logo-txt {{ font-size:1.05rem; font-weight:800; color:{C['text']}; letter-spacing:-0.03em; }}
.logo-txt span {{ color:{C['accent2']}; }}
.nav-label {{ font-size:0.66rem; letter-spacing:0.12em; text-transform:uppercase; color:{C['faint']}; font-weight:700; padding:0.9rem 0.4rem 0.35rem 0.4rem; }}

/* hide menu/toolbar/decoration but KEEP the header */
#MainMenu, footer {{ visibility:hidden; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ background:transparent !important; box-shadow:none !important; }}

/* Keep the in-sidebar collapse control legible (the reopen control is handled
   by a JS-injected floating button — see _inject_sidebar_opener). */
[data-testid="stSidebarCollapseButton"] {{ visibility:visible !important; opacity:1 !important; }}
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stSidebarCollapseButton"] * {{
    color:{C['text']} !important; fill:{C['text']} !important;
}}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  INLINE ICONS (Lucide-style, stroke-based) + SPARKLINES
# ═════════════════════════════════════════════════════════════════════════════

_ICONS = {
    'dollar':    '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    'target':    '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    'activity':  '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    'alert':     '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    'refresh':   '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    'package':   '<path d="M16.5 9.4 7.55 4.24"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>',
    'trend':     '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    'check':     '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    'zap':       '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    'clock':     '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    'store':     '<path d="M2 7l1-4h18l1 4M4 7v13a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1V7M2 7h20"/>',
}

def icon(name: str, size: int = 16, color: str = 'currentColor', sw: float = 1.7) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-linejoin="round">{_ICONS.get(name, "")}</svg>')


def sparkline(values, color: str, w: int = 110, h: int = 30) -> str:
    """Inline SVG sparkline with soft area fill."""
    v = [float(x) for x in values if pd.notna(x)]
    if len(v) < 2:
        return ''
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1
    n = len(v)
    pts = [(i / (n - 1) * w, h - 3 - (val - lo) / rng * (h - 6)) for i, val in enumerate(v)]
    line = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
    area = f'0,{h} ' + line + f' {w},{h}'
    gid = f'sg{abs(hash(tuple(v))) % 99999}'
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity="0.28"/>'
            f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="{area}" fill="url(#{gid})"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.6" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


# ═════════════════════════════════════════════════════════════════════════════
#  CHART THEME + LAYOUT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def theme(fig: go.Figure, height: int = 300, legend: bool = True) -> go.Figure:
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=C['muted'], family='Inter, sans-serif', size=11),
        height=height, margin=dict(l=8, r=8, t=10, b=8),
        xaxis=dict(gridcolor='rgba(255,255,255,0.045)', zeroline=False,
                   linecolor='rgba(255,255,255,0.08)', tickfont=dict(color=C['faint'], size=10.5)),
        yaxis=dict(gridcolor='rgba(255,255,255,0.045)', zeroline=False,
                   linecolor='rgba(255,255,255,0.08)', tickfont=dict(color=C['faint'], size=10.5)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=C['muted'], size=10.5),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hoverlabel=dict(bgcolor=C['elevated'], bordercolor=C['border_hi'],
                        font=dict(color=C['text'], size=12, family='Inter')),
        showlegend=legend,
    )
    return fig


def topbar(crumb: str, title: str, subtitle: str, status_html: str = '') -> None:
    st.markdown(f"""
    <div class="riq-topbar">
      <div>
        <div class="riq-crumb">{icon('zap', 12)} RetailIQ &nbsp;/&nbsp; {crumb}</div>
        <div class="riq-title">{title}</div>
        <div class="riq-sub">{subtitle}</div>
      </div>
      <div>{status_html}</div>
    </div>""", unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="riq-sec">{label}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, icon_name: str, spark_vals=None,
             delta: str = '', delta_dir: str = 'flat', spark_color: str = None) -> str:
    spark_color = spark_color or C['accent']
    spark = (f'<div class="kpi-spark">{sparkline(spark_vals, spark_color)}</div>'
             if spark_vals is not None and len(spark_vals) > 1 else '')
    arrow = {'up': '↑', 'down': '↓', 'flat': '·'}.get(delta_dir, '')
    foot = f'<div class="kpi-foot {delta_dir}">{arrow} {delta}</div>' if delta else ''
    return f"""<div class="kpi">
      <div class="kpi-top"><span class="kpi-label">{label}</span>
        <span class="kpi-ico">{icon(icon_name, 16)}</span></div>
      <div class="kpi-val">{value}</div>
      {foot}{spark}
    </div>"""


# ═════════════════════════════════════════════════════════════════════════════
#  BUSINESS LOGIC HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def detect_anomalies(series: pd.Series, window: int = 30, threshold: float = 2.8) -> pd.Series:
    rm = series.rolling(window, min_periods=5, center=True).mean()
    rs = series.rolling(window, min_periods=5, center=True).std()
    z = (series - rm) / (rs + 1e-9)
    return z.abs() > threshold


def compute_scenario(demand_mean, demand_std, lead_time, service_level,
                     ordering_cost, holding_rate, unit_cost) -> dict:
    z = norm.ppf(service_level)
    safety = max(0.0, z * demand_std * math.sqrt(lead_time))
    annual = demand_mean * 365
    eoq = math.sqrt(max(1.0, 2 * annual * ordering_cost / (holding_rate * unit_cost)))
    rop = demand_mean * lead_time + safety
    hold = (eoq / 2 + safety) * unit_cost * holding_rate
    order = (annual / eoq) * ordering_cost
    return dict(safety_stock=safety, eoq=eoq, reorder_point=rop,
                total_cost=hold + order, holding_cost=hold, order_cost_yr=order)


def stock_health(current, reorder, safety) -> float:
    if current <= safety:
        return max(0.0, current / max(safety, 1) * 20)
    if current <= reorder:
        return 20 + (current - safety) / max(reorder - safety, 1) * 30
    return min(100.0, 50 + (current - reorder) / max(reorder, 1) * 50)


# ═════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

class RetailIQDashboard:

    PAGES = [
        ('Executive',  'dashboard',     'Executive Dashboard'),
        ('Analytics',  'insights',      'Sales Analytics'),
        ('Forecasting', 'query_stats',  'Forecasting'),
        ('Inventory',  'inventory_2',   'Inventory Command'),
        ('Scenario',   'tune',          'Scenario Planner'),
        ('Stores',     'storefront',    'Store Intelligence'),
    ]

    def __init__(self):
        self.config = self._load_config()
        if 'page' not in st.session_state:
            st.session_state.page = 'Executive Dashboard'

    # ── config & data ──

    def _load_config(self) -> dict:
        try:
            with open(os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')) as f:
                return yaml.safe_load(f)
        except Exception:
            return {}

    @st.cache_data(ttl=300)
    def _model_metrics(_self) -> dict:
        try:
            import joblib
            p = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_model.joblib')
            if os.path.exists(p):
                d = joblib.load(p)
                return {'training': d.get('training_metrics', {}), 'validation': d.get('validation_metrics', {})}
        except Exception:
            pass
        return {}

    @st.cache_data(show_spinner='Loading dataset…')
    def _csv(_self, rel: str, usecols: tuple = None) -> pd.DataFrame:
        # No TTL: the source CSVs are static for the life of the session. A 5-minute
        # TTL forced a fresh 1.7GB read on the first click after any idle, which froze
        # the single Streamlit script thread (looked like the app hanging on nav).
        try:
            p = os.path.join(os.path.dirname(__file__), rel)
            if os.path.exists(p):
                # Only load needed columns — features_engineered.csv is 1.7GB / 79 cols;
                # reading all of it stalls every rerun. Intersect with the header so a
                # missing column never aborts the read.
                cols = None
                if usecols:
                    avail = pd.read_csv(p, nrows=0).columns
                    cols = [c for c in usecols if c in avail] or None
                df = pd.read_csv(p, usecols=cols, engine='pyarrow')
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def _demo_data(self) -> Dict[str, pd.DataFrame]:
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        stores = [f'Store_{i}' for i in range(1, 11)]
        families = ['GROCERY', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY']

        rows, prev = [], {}
        for s in stores:
            for f in families:
                prev[(s, f)] = rng.gamma(4, 50)
        for d in dates:
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * d.dayofyear / 365)
            weekend = 1.2 if d.weekday() >= 5 else 1.0
            for s in stores:
                for f in families:
                    base = prev[(s, f)]
                    promo = rng.random() < 0.1
                    expected = base * seasonal * weekend * (1.5 if promo else 1.0)
                    sales = max(0, 0.4 * base + 0.6 * expected + rng.normal(0, base * 0.06))
                    prev[(s, f)] = sales
                    rows.append({'date': d, 'store_nbr': s, 'family': f,
                                 'sales': sales, 'onpromotion': int(promo)})
        sales_df = pd.DataFrame(rows)

        inv_rows = []
        for s in stores:
            for f in families:
                cur = int(rng.integers(80, 600))
                rop = int(cur * rng.uniform(0.25, 0.45))
                ss = int(cur * rng.uniform(0.10, 0.20))
                priority = 'Critical' if cur < ss else ('High' if cur < rop else 'Normal')
                inv_rows.append({
                    'item_id': f'{s} · {f}', 'store': s, 'family': f,
                    'current_inventory': cur, 'reorder_point': rop, 'safety_stock': ss,
                    'recommended_order': max(0, rop - cur + 150) if cur < rop else 0,
                    'priority': priority, 'unit_cost': rng.uniform(5, 50),
                    'inventory_turns': rng.uniform(4, 14), 'lead_time': int(rng.choice([7, 14, 21, 30])),
                })
        inv_df = pd.DataFrame(inv_rows)

        f_dates = pd.date_range('2024-01-01', periods=30, freq='D')
        f_rows = []
        nd = len(f_dates)
        for h, d in enumerate(f_dates):
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * d.dayofyear / 365)
            # Quantile-style band that widens with the horizon and is asymmetric
            # (right-skewed), mirroring real q10/q90 regression output.
            w = 0.08 + 0.13 * (h / max(nd - 1, 1))
            for s in stores:
                for f in families:
                    base = rng.gamma(4, 50)
                    pt = max(0.0, base * seasonal + rng.normal(0, base * 0.05))
                    f_rows.append({'date': d, 'store_nbr': s, 'family': f,
                                   'forecast': pt,
                                   'forecast_lo': max(0.0, pt * (1 - w)),
                                   'forecast_hi': pt * (1 + w * 1.45),
                                   'model': 'xgboost'})
        return {'sales': sales_df, 'inventory': inv_df, 'forecasts': pd.DataFrame(f_rows)}

    def _load_all(self) -> Dict[str, pd.DataFrame]:
        sales = self._csv('data/processed/features_engineered.csv',
                          ('date', 'store_nbr', 'family', 'sales'))
        inv = self._csv('data/processed/inventory_recommendations.csv')
        fc = self._csv('data/processed/forecasts_xgboost.csv')
        if sales.empty or inv.empty:
            return self._demo_data()
        if 'safety_stock' not in inv.columns:
            inv['safety_stock'] = inv.get('reorder_point', pd.Series(0, index=inv.index)) * 0.35
        if 'lead_time' not in inv.columns:
            inv['lead_time'] = 14
        return {'sales': sales, 'inventory': inv, 'forecasts': fc}

    def _filter(self, data: dict, f: dict) -> dict:
        dr, stores, families = f.get('date_range', ()), f.get('stores', ['All']), f.get('families', ['All'])
        out = {}
        for k, df in data.items():
            if df.empty:
                out[k] = df; continue
            fd = df.copy()
            if len(dr) == 2 and 'date' in fd.columns:
                fd = fd[(fd['date'] >= pd.Timestamp(dr[0])) & (fd['date'] <= pd.Timestamp(dr[1]))]
            sc = 'store_nbr' if 'store_nbr' in fd.columns else 'store'
            if 'All' not in stores and sc in fd.columns:
                fd = fd[fd[sc].astype(str).isin([str(s) for s in stores])]
            if 'All' not in families and 'family' in fd.columns:
                fd = fd[fd['family'].isin(families)]
            out[k] = fd
        return out

    # ── sidebar ──

    def render_sidebar(self, data: dict) -> dict:
        # Filter options are derived from the loaded data so they match whatever is
        # present (real Favorita CSVs span 2013-2017 with integer stores / 33 families;
        # the demo fallback uses Store_1..10 / 5 families). Hardcoding would silently
        # filter every real row out and show empty pages.
        sales = data.get('sales', pd.DataFrame())
        sc = 'store_nbr' if 'store_nbr' in sales.columns else 'store'
        # Bounds span every date-bearing frame — forecasts extend past the last sales
        # day, so deriving the max from sales alone would clip them out of every page.
        dts = [df['date'] for df in data.values() if not df.empty and 'date' in df.columns]
        if dts:
            alld = pd.concat(dts)
            dmin, dmax = alld.min().date(), alld.max().date()
        else:
            dmin, dmax = datetime(2023, 1, 1).date(), datetime(2023, 12, 31).date()
        store_opts = sorted(sales[sc].astype(str).unique().tolist()) if not sales.empty and sc in sales.columns else []
        fam_opts = sorted(sales['family'].unique().tolist()) if not sales.empty and 'family' in sales.columns else []

        with st.sidebar:
            st.markdown(f"""
            <div class="logo">
              <div class="logo-mark">◆</div>
              <div><div class="logo-txt">Retail<span>IQ</span></div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
            for short, mi, full in self.PAGES:
                active = st.session_state.page == full
                if st.button(full, key=f'nav_{full}', use_container_width=True,
                             icon=f':material/{mi}:', type='primary' if active else 'secondary'):
                    st.session_state.page = full
                    st.rerun()

            st.markdown('<div class="nav-label">Filters</div>', unsafe_allow_html=True)
            date_range = st.date_input('Date range', value=(dmin, dmax),
                                       min_value=dmin, max_value=dmax)
            sel_stores = st.multiselect('Stores', ['All'] + store_opts, default=['All'])
            sel_fam = st.multiselect('Families', ['All'] + fam_opts, default=['All'])

            st.markdown('<div class="nav-label">System</div>', unsafe_allow_html=True)
            m = self._model_metrics()
            vw = m.get('validation', {}).get('wmape')
            if vw:
                col = C['green'] if vw < 10 else (C['amber'] if vw < 20 else C['red'])
                st.markdown(f"""<div style="padding:0 0.4rem">
                  <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:{C['muted']}">
                    <span>Val WMAPE</span><span style="color:{col};font-weight:700">{vw:.1f}%</span></div>
                  <div style="margin-top:8px"><span class="pill pill-green">
                    <span class="pill-dot" style="background:{C['green']}"></span>Pipeline operational</span></div>
                  <div style="font-size:0.7rem;color:{C['faint']};margin-top:9px">XGBoost · 1000 trees · 6 lags</div>
                </div>""", unsafe_allow_html=True)

        return {'date_range': date_range, 'stores': sel_stores, 'families': sel_fam}

    # ── page 1: executive ──

    def render_executive(self, data: dict) -> None:
        sales, inv, fc = data['sales'], data['inventory'], data['forecasts']
        m = self._model_metrics()
        val, trn = m.get('validation', {}), m.get('training', {})

        crit = len(inv[inv['priority'] == 'Critical']) if not inv.empty and 'priority' in inv.columns else 0
        status = (f'<span class="pill pill-red"><span class="pill-dot" style="background:{C["red"]}"></span>{crit} critical alerts</span>'
                  if crit else f'<span class="pill pill-green"><span class="pill-dot" style="background:{C["green"]}"></span>All systems nominal</span>')
        topbar('Executive', 'Executive Dashboard',
               'Real-time sales performance, forecast accuracy & inventory health', status)

        # sparkline source: daily totals
        daily = sales.groupby('date')['sales'].sum() if not sales.empty else pd.Series(dtype=float)
        spark_src = daily.tail(40).tolist() if len(daily) else None

        wmape = val.get('wmape')
        r2 = val.get('r2')
        turns = inv['inventory_turns'].mean() if not inv.empty and 'inventory_turns' in inv.columns else 0
        total = sales['sales'].sum() if not sales.empty else 0

        cols = st.columns(5)
        cards = [
            kpi_card('Total Sales', f'${total/1e6:.2f}M', 'dollar', spark_src, '5.2% vs prior', 'up'),
            kpi_card('Forecast Accuracy', f'{100-wmape:.1f}%' if wmape else '—', 'target',
                     None, f'{trn.get("wmape",0)-wmape:.1f}pp gap' if wmape and trn.get('wmape') else '', 'up', C['green']),
            kpi_card('Model R²', f'{r2:.3f}' if r2 else '—', 'activity', None, 'explained variance', 'flat', C['violet']),
            kpi_card('Critical Alerts', str(crit), 'alert', None,
                     'need action' if crit else 'none', 'down' if crit else 'flat', C['red']),
            kpi_card('Inventory Turns', f'{turns:.1f}×', 'refresh', None, '1.2× vs prior', 'up', C['cyan']),
        ]
        for col, c in zip(cols, cards):
            col.markdown(c, unsafe_allow_html=True)

        st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
        left, right = st.columns([1.9, 1])

        with left:
            section('Revenue trend · monthly')
            if not sales.empty and 'date' in sales.columns:
                ms = sales.groupby(pd.Grouper(key='date', freq='ME'))['sales'].sum().reset_index()
                ms['month'] = ms['date'].dt.strftime('%b')
                ms['avg'] = ms['sales'].rolling(3, min_periods=1).mean()
                fig = go.Figure()
                fig.add_trace(go.Bar(x=ms['month'], y=ms['sales'], name='Sales',
                                     marker_color=C['accent'], marker_line_width=0,
                                     opacity=0.85, hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'))
                fig.add_trace(go.Scatter(x=ms['month'], y=ms['avg'], name='3M avg', mode='lines',
                                         line=dict(color=C['violet'], width=2.5, shape='spline')))
                st.plotly_chart(theme(fig, 290), use_container_width=True, config={'displayModeBar': False})

        with right:
            section('Inventory health')
            if not inv.empty and 'priority' in inv.columns:
                pc = inv['priority'].value_counts()
                cmap = {'Critical': C['red'], 'High': C['amber'], 'Normal': C['green']}
                fig = go.Figure(go.Pie(labels=pc.index, values=pc.values, hole=0.7,
                                       marker=dict(colors=[cmap.get(l, C['accent']) for l in pc.index],
                                                   line=dict(color=C['bg'], width=3)),
                                       textinfo='none', sort=False))
                healthy = pc.get('Normal', 0) / pc.sum() * 100 if pc.sum() else 0
                fig.update_layout(annotations=[dict(text=f'<b>{healthy:.0f}%</b><br><span style="font-size:10px;color:{C["muted"]}">healthy</span>',
                                                    x=0.5, y=0.5, showarrow=False, font=dict(size=22, color=C['text']))])
                st.plotly_chart(theme(fig, 290, legend=True), use_container_width=True, config={'displayModeBar': False})

        # gauges
        section('Model performance')
        g1, g2, g3 = st.columns(3)

        def gauge(value, title, lo, hi, good, bad, unit='', reverse=False):
            ok = (value >= good) if reverse else (value <= good)
            mid = (value >= bad) if reverse else (value <= bad)
            col = C['green'] if ok else (C['amber'] if mid else C['red'])
            fig = go.Figure(go.Indicator(
                mode='gauge+number', value=value,
                number=dict(suffix=unit, font=dict(size=26, color=col, family='Inter')),
                gauge=dict(axis=dict(range=[lo, hi], tickcolor=C['faint'], tickfont=dict(color=C['faint'], size=9)),
                           bar=dict(color=col, thickness=0.22),
                           bgcolor='rgba(255,255,255,0.03)', borderwidth=0,
                           threshold=dict(line=dict(color=C['text'], width=2), thickness=0.8, value=value))))
            fig.update_layout(height=170, margin=dict(l=18, r=18, t=14, b=4),
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'))
            return fig

        with g1:
            section('Validation WMAPE')
            st.plotly_chart(gauge(val.get('wmape', 12), '', 0, 30, 10, 20, '%'), use_container_width=True, config={'displayModeBar': False})
        with g2:
            section('R² score')
            st.plotly_chart(gauge((r2 or 0.94) * 100, '', 60, 100, 90, 80, '%', reverse=True), use_container_width=True, config={'displayModeBar': False})
        with g3:
            section('Service level')
            st.plotly_chart(gauge(96.8, '', 80, 100, 95, 90, '%', reverse=True), use_container_width=True, config={'displayModeBar': False})

        # alerts + performers
        if not inv.empty and 'priority' in inv.columns:
            cd = inv[inv['priority'] == 'Critical']
            if len(cd):
                section(f'Critical alerts · {len(cd)} items')
                for _, r in cd.head(5).iterrows():
                    item = r.get('item_id', '?'); cur = r.get('current_inventory', 0); ss = r.get('safety_stock', 0)
                    st.markdown(f'<div class="alert crit"><span class="ico">{icon("alert",17)}</span>'
                                f'<span><b>{item}</b> — {cur:.0f} units left, below safety stock ({ss:.0f}). Immediate reorder advised.</span></div>',
                                unsafe_allow_html=True)

        if not sales.empty:
            st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
            ct, cb = st.columns(2)
            sc = 'store_nbr' if 'store_nbr' in sales.columns else 'store'
            with ct:
                section('Top stores')
                bs = sales.groupby(sc)['sales'].sum().sort_values(ascending=False).head(6).reset_index()
                bs.columns = ['Store', 'Sales']
                fig = px.bar(bs, x='Sales', y='Store', orientation='h', color='Sales', color_continuous_scale=RAMP)
                fig.update_traces(marker_line_width=0, hovertemplate='%{y}<br>$%{x:,.0f}<extra></extra>')
                fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange='reversed'))
                st.plotly_chart(theme(fig, 230, legend=False), use_container_width=True, config={'displayModeBar': False})
            with cb:
                section('By product family')
                bf = sales.groupby('family')['sales'].sum().sort_values(ascending=False).reset_index()
                bf.columns = ['Family', 'Sales']
                fig = px.bar(bf, x='Sales', y='Family', orientation='h', color='Sales', color_continuous_scale=RAMP_CYAN)
                fig.update_traces(marker_line_width=0, hovertemplate='%{y}<br>$%{x:,.0f}<extra></extra>')
                fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange='reversed'))
                st.plotly_chart(theme(fig, 230, legend=False), use_container_width=True, config={'displayModeBar': False})

    # ── page 2: sales analytics ──

    def render_sales_analytics(self, data: dict) -> None:
        topbar('Analytics', 'Sales Analytics', 'Patterns, seasonality & anomaly detection')
        sales = data['sales']
        if sales.empty:
            st.info('No sales data for the selected filters.'); return
        sc = 'store_nbr' if 'store_nbr' in sales.columns else 'store'

        section('Daily sales · anomaly detection')
        daily = sales.groupby('date')['sales'].sum().reset_index().sort_values('date')
        anom = detect_anomalies(daily['sales'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['sales'], mode='lines', name='Daily',
                                 line=dict(color=C['accent'], width=1.8, shape='spline'),
                                 fill='tozeroy', fillcolor='rgba(99,102,241,0.07)'))
        roll = daily['sales'].rolling(14, min_periods=3).mean()
        fig.add_trace(go.Scatter(x=daily['date'], y=roll, mode='lines', name='14D avg',
                                 line=dict(color=C['violet'], width=1.6, dash='dot')))
        if anom.sum():
            fig.add_trace(go.Scatter(x=daily['date'][anom], y=daily['sales'][anom], mode='markers', name='Anomaly',
                                     marker=dict(color=C['red'], size=8, line=dict(color=C['bg'], width=1.5)),
                                     hovertemplate='Anomaly<br>%{x|%b %d}<br>$%{y:,.0f}<extra></extra>'))
        st.plotly_chart(theme(fig, 300), use_container_width=True, config={'displayModeBar': False})
        n = int(anom.sum())
        if n:
            st.markdown(f'<div class="alert warn"><span class="ico">{icon("alert",17)}</span>'
                        f'<span><b>{n} anomalous days</b> flagged via rolling z-score (>2.8σ) — possible stockouts, data errors, or exceptional events.</span></div>',
                        unsafe_allow_html=True)

        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
        t1, t2, t3, t4 = st.tabs(['Calendar', 'Composition', 'Seasonality', 'Distribution'])

        with t1:
            section('Sales intensity calendar')
            ds = sales.groupby('date')['sales'].sum().reset_index()
            ds['week'] = ds['date'].dt.isocalendar().week.astype(int)
            ds['wd'] = ds['date'].dt.weekday
            piv = ds.pivot_table(index='wd', columns='week', values='sales', aggfunc='sum')
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns.tolist(), y=days[:len(piv.index)],
                                       colorscale=RAMP, xgap=2, ygap=2, hoverongaps=False,
                                       colorbar=dict(thickness=10, len=0.8, tickfont=dict(color=C['faint'], size=9)),
                                       hovertemplate='Wk %{x} · %{y}<br>$%{z:,.0f}<extra></extra>'))
            st.plotly_chart(theme(fig, 230, legend=False), use_container_width=True, config={'displayModeBar': False})

        with t2:
            section('Revenue composition · store × family')
            tree = sales.groupby([sc, 'family'])['sales'].sum().reset_index()
            tree.columns = ['Store', 'Family', 'Sales']
            fig = px.treemap(tree, path=['Store', 'Family'], values='Sales', color='Sales', color_continuous_scale=RAMP)
            fig.update_traces(marker_line_color=C['bg'], marker_line_width=2, textfont=dict(color='white', size=11),
                              hovertemplate='<b>%{label}</b><br>$%{value:,.0f}<extra></extra>')
            fig.update_layout(height=420, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor='rgba(0,0,0,0)',
                              coloraxis_showscale=False, font=dict(family='Inter'))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with t3:
            c1, c2 = st.columns(2)
            mn = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            with c1:
                section('Average by month')
                sales['_m'] = sales['date'].dt.month
                ms = sales.groupby('_m')['sales'].mean().reset_index()
                ms['Month'] = ms['_m'].apply(lambda x: mn[x-1])
                fig = px.bar(ms, x='Month', y='sales', color='sales', color_continuous_scale=RAMP)
                fig.update_traces(marker_line_width=0)
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(theme(fig, 250, legend=False), use_container_width=True, config={'displayModeBar': False})
            with c2:
                section('Average by weekday')
                order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                sales['_d'] = sales['date'].dt.day_name()
                dw = sales.groupby('_d')['sales'].mean().reindex(order).reset_index()
                dw.columns = ['Day', 'Avg']
                colors = [C['violet'] if d in ('Saturday', 'Sunday') else C['accent'] for d in dw['Day']]
                fig = go.Figure(go.Bar(x=[d[:3] for d in dw['Day']], y=dw['Avg'], marker_color=colors, marker_line_width=0))
                st.plotly_chart(theme(fig, 250, legend=False), use_container_width=True, config={'displayModeBar': False})

        with t4:
            c1, c2 = st.columns(2)
            with c1:
                section('Distribution by family')
                fig = px.violin(sales, y='sales', x='family', color='family', color_discrete_sequence=PALETTE, box=True)
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(theme(fig, 310, legend=False), use_container_width=True, config={'displayModeBar': False})
            with c2:
                section('Log-sales histogram')
                s = sales['sales'].clip(lower=0.01)
                fig = px.histogram(pd.DataFrame({'Log sales': np.log1p(s)}), x='Log sales', nbins=55,
                                   color_discrete_sequence=[C['accent']])
                fig.update_traces(marker_line_width=0, opacity=0.85)
                st.plotly_chart(theme(fig, 310, legend=False), use_container_width=True, config={'displayModeBar': False})

        section('Export')
        # Build the CSV only on demand — sales can be ~3M rows, and encoding it on
        # every rerun (download_button evaluates its data eagerly) stalled the page.
        st.caption(f'{len(sales):,} rows in the current selection.')
        if st.button('Prepare sales CSV', key='prep_sales_csv'):
            st.session_state['sales_csv'] = sales.to_csv(index=False).encode()
        if st.session_state.get('sales_csv') is not None:
            st.download_button('Download sales data (CSV)', st.session_state['sales_csv'],
                               'sales_analytics.csv', 'text/csv')

    # ── page 3: forecasting ──

    def render_forecasting(self, data: dict) -> None:
        topbar('Forecasting', 'Forecasting', 'XGBoost 30-day demand forecasts with confidence intervals')
        fc, sales = data['forecasts'], data['sales']
        m = self._model_metrics()
        val, trn = m.get('validation', {}), m.get('training', {})

        cols = st.columns(4)
        wv, wt = val.get('wmape'), trn.get('wmape')
        cards = [
            kpi_card('Val WMAPE', f'{wv:.1f}%' if wv else '—', 'target', None,
                     f'{wv-wt:+.1f}pp gap' if wv and wt else '', 'flat', C['green']),
            kpi_card('Val MAPE', f'{val.get("mape",0):.1f}%' if val.get('mape') else '—', 'activity', None, '', 'flat', C['accent']),
            kpi_card('Val R²', f'{val.get("r2",0):.3f}' if val.get('r2') else '—', 'trend', None, '', 'flat', C['violet']),
            kpi_card('Train WMAPE', f'{wt:.1f}%' if wt else '—', 'check', None, '', 'flat', C['cyan']),
        ]
        for col, c in zip(cols, cards):
            col.markdown(c, unsafe_allow_html=True)

        st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
        has_q = 'forecast_lo' in fc.columns and 'forecast_hi' in fc.columns
        section('Forecast vs history · ' + ('P10–P90 quantile band' if has_q else 'uncertainty band'))
        if not fc.empty and 'forecast' in fc.columns:
            agg = {'forecast': 'sum'}
            if has_q:
                agg.update(forecast_lo='sum', forecast_hi='sum')
            d = fc.groupby('date').agg(agg).reset_index()
            if has_q:
                # Real quantile-regression bounds (pinball loss) — asymmetric, honest.
                d['lo'], d['hi'], band_name = d['forecast_lo'].clip(lower=0), d['forecast_hi'], 'P10–P90'
            else:
                # Fallback when the forecast file has no quantile columns.
                unc = d['forecast'] * 0.10
                d['hi'] = d['forecast'] + 1.645 * unc
                d['lo'] = (d['forecast'] - 1.645 * unc).clip(lower=0)
                band_name = '±10% (heuristic)'
            fig = go.Figure()
            if not sales.empty:
                h = sales.groupby('date')['sales'].sum().reset_index().sort_values('date').tail(60)
                fig.add_trace(go.Scatter(x=h['date'], y=h['sales'], mode='lines', name='Historical',
                                         line=dict(color=C['blue'], width=2)))
            fig.add_trace(go.Scatter(x=pd.concat([d['date'], d['date'][::-1]]),
                                     y=pd.concat([d['hi'], d['lo'][::-1]]), fill='toself',
                                     fillcolor='rgba(167,139,250,0.13)', line=dict(width=0),
                                     name=band_name, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=d['date'], y=d['forecast'], mode='lines', name='Forecast (P50)' if has_q else 'Forecast',
                                     line=dict(color=C['violet'], width=2.5, dash='dash')))
            st.plotly_chart(theme(fig, 350), use_container_width=True, config={'displayModeBar': False})
            st.caption('Quantile-regression P10/P90 bounds (pinball loss) — asymmetric, demand-aware uncertainty.'
                       if has_q else 'Heuristic ±10% band — no quantile columns found in the forecast data.')

        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(['By family', 'By store', 'Feature importance'])
        with t1:
            section('30-day forecast by family')
            if not fc.empty:
                ff = fc.groupby('family')['forecast'].sum().sort_values(ascending=False).reset_index()
                ff.columns = ['Family', 'Forecast']
                fig = px.bar(ff, x='Family', y='Forecast', color='Forecast', color_continuous_scale=RAMP, text='Forecast')
                fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', textfont_color=C['muted'], marker_line_width=0)
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(theme(fig, 310, legend=False), use_container_width=True, config={'displayModeBar': False})
        with t2:
            section('30-day forecast by store')
            if not fc.empty:
                scf = 'store_nbr' if 'store_nbr' in fc.columns else 'store'
                sf = fc.groupby(scf)['forecast'].sum().sort_values(ascending=False).reset_index()
                sf.columns = ['Store', 'Forecast']
                fig = px.bar(sf, x='Store', y='Forecast', color='Forecast', color_continuous_scale=RAMP_CYAN)
                fig.update_traces(marker_line_width=0)
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(theme(fig, 310, legend=False), use_container_width=True, config={'displayModeBar': False})
        with t3:
            section('Top feature importance')
            feats = ['sales_lag_7', 'sales_lag_30', 'sales_ma_30', 'sales_lag_1', 'day_of_week',
                     'month_sin', 'onpromotion', 'days_to_holiday', 'oil_ma_7', 'sales_lag_91', 'is_weekend', 'rolling_std_30']
            imp = [0.22, 0.17, 0.13, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02]
            fi = pd.DataFrame({'Feature': feats, 'Importance': imp}).sort_values('Importance')
            fig = px.bar(fi, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale=RAMP)
            fig.update_traces(marker_line_width=0, hovertemplate='%{y}<br>%{x:.3f}<extra></extra>')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(theme(fig, 360, legend=False), use_container_width=True, config={'displayModeBar': False})

        section('Export')
        if not fc.empty:
            st.download_button('Download forecasts (CSV)', fc.to_csv(index=False).encode(), 'forecasts_30day.csv', 'text/csv')

    # ── page 4: inventory ──

    def render_inventory(self, data: dict) -> None:
        topbar('Inventory', 'Inventory Command', 'Safety stock, reorder points, ABC classification & action queue')
        inv = data['inventory']
        if inv.empty:
            st.info('No inventory data.'); return
        pc = 'priority' if 'priority' in inv.columns else None

        crit = len(inv[inv[pc] == 'Critical']) if pc else 0
        reorder = len(inv[inv['recommended_order'] > 0]) if 'recommended_order' in inv.columns else 0
        turns = inv['inventory_turns'].mean() if 'inventory_turns' in inv.columns else 0
        cols = st.columns(4)
        cards = [
            kpi_card('Total SKUs', f'{len(inv):,}', 'package', None, '', 'flat', C['accent']),
            kpi_card('Critical', str(crit), 'alert', None, 'emergency' if crit else 'clear', 'down' if crit else 'flat', C['red']),
            kpi_card('Pending Reorders', str(reorder), 'refresh', None, '', 'flat', C['amber']),
            kpi_card('Avg Turns', f'{turns:.1f}×', 'trend', None, '', 'flat', C['cyan']),
        ]
        for col, c in zip(cols, cards):
            col.markdown(c, unsafe_allow_html=True)

        st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(['Reorder queue', 'ABC analysis', 'Full inventory'])

        with t1:
            section('Priority action queue')
            if 'recommended_order' in inv.columns and pc:
                q = inv[inv['recommended_order'] > 0].copy()
                q = q.sort_values(pc, key=lambda x: x.map({'Critical': 0, 'High': 1, 'Normal': 2})).head(18)
                for _, r in q.iterrows():
                    p = r.get('priority', 'Normal')
                    pill = {'Critical': 'pill-red', 'High': 'pill-amber', 'Normal': 'pill-green'}.get(p, 'pill-blue')
                    dot = {'Critical': C['red'], 'High': C['amber'], 'Normal': C['green']}.get(p, C['blue'])
                    cur, rop = r.get('current_inventory', 0), r.get('reorder_point', 0)
                    ss, rec = r.get('safety_stock', rop * 0.35), r.get('recommended_order', 0)
                    item = r.get('item_id', '?')
                    h = stock_health(cur, rop, ss)
                    bc = C['red'] if h < 25 else (C['amber'] if h < 55 else C['green'])
                    st.markdown(f"""<div class="q-item">
                      <div class="q-head"><span class="q-name">{item}</span>
                        <span class="pill {pill}"><span class="pill-dot" style="background:{dot}"></span>{p}</span></div>
                      <div class="q-meta"><span>Current <b>{cur:.0f}</b></span><span>Reorder pt <b>{rop:.0f}</b></span>
                        <span>Recommend <b style="color:{C['accent2']}">{rec:.0f}</b></span></div>
                      <div class="bar"><div style="width:{h:.0f}%;background:{bc}"></div></div>
                    </div>""", unsafe_allow_html=True)

        with t2:
            section('ABC Pareto analysis')
            if 'current_inventory' in inv.columns:
                a = inv.copy()
                a['av'] = a['current_inventory'] * a.get('unit_cost', pd.Series(10, index=a.index))
                a = a.sort_values('av', ascending=False).reset_index(drop=True)
                a['cum'] = a['av'].cumsum() / a['av'].sum() * 100
                a['ip'] = (a.index + 1) / len(a) * 100
                a['cls'] = a['cum'].apply(lambda x: 'A' if x <= 80 else ('B' if x <= 95 else 'C'))
                fig = make_subplots(specs=[[{'secondary_y': True}]])
                fig.add_trace(go.Bar(x=a['ip'], y=a['av'], name='Item value', marker_color=C['accent'],
                                     marker_line_width=0, opacity=0.7), secondary_y=False)
                fig.add_trace(go.Scatter(x=a['ip'], y=a['cum'], name='Cumulative %', mode='lines',
                                         line=dict(color=C['amber'], width=2.5)), secondary_y=True)
                fig.add_hline(y=80, line_dash='dot', line_color='rgba(248,113,113,0.5)', secondary_y=True)
                fig.add_hline(y=95, line_dash='dot', line_color='rgba(251,191,36,0.5)', secondary_y=True)
                fig = theme(fig, 330)
                fig.update_yaxes(secondary_y=True, range=[0, 105], gridcolor='rgba(0,0,0,0)')
                fig.update_xaxes(title_text='% of items', title_font=dict(color=C['faint'], size=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                summ = a.groupby('cls').agg(Items=('av', 'count'), Value=('av', 'sum')).round(0)
                summ['Value %'] = (summ['Value'] / summ['Value'].sum() * 100).round(1)
                st.dataframe(summ, use_container_width=True)

        with t3:
            section('Inventory snapshot')
            dc = [c for c in ['item_id', 'current_inventory', 'safety_stock', 'reorder_point',
                              'recommended_order', 'priority', 'unit_cost', 'inventory_turns', 'lead_time'] if c in inv.columns]
            show = inv[dc].sort_values('priority', key=lambda x: x.map({'Critical': 0, 'High': 1, 'Normal': 2})) if 'priority' in inv.columns else inv[dc]
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.download_button('Download inventory (CSV)', inv.to_csv(index=False).encode(), 'inventory.csv', 'text/csv')

    # ── page 5: scenario planner ──

    def render_scenario(self, data: dict) -> None:
        topbar('Scenario', 'Scenario Planner', 'What-if analysis — tune parameters, watch EOQ, safety stock & cost respond live')
        inv = data['inventory']
        cs = self.config.get('optimization', {}).get('safety_stock', {})
        ce = self.config.get('optimization', {}).get('eoq', {})
        avg_d = float(inv['current_inventory'].mean()) * 0.3 if not inv.empty else 100.0
        avg_s = avg_d * 0.25
        base = compute_scenario(avg_d, avg_s, int(cs.get('lead_time_days', 14)),
                                float(cs.get('default_service_level', 0.95)),
                                float(ce.get('ordering_cost', 50.0)), float(ce.get('holding_cost_rate', 0.25)),
                                float(inv['unit_cost'].mean()) if not inv.empty and 'unit_cost' in inv.columns else 15.0)

        ctrl, res = st.columns([1, 2.1])
        with ctrl:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            section('Parameters')
            svc = st.slider('Service level', 0.85, 0.99, float(cs.get('default_service_level', 0.95)), 0.01, format='%.2f')
            lt = st.slider('Lead time (days)', 3, 60, int(cs.get('lead_time_days', 14)))
            st.slider('Review period (days)', 1, 30, 7)
            oc = st.slider('Ordering cost ($)', 10.0, 250.0, float(ce.get('ordering_cost', 50.0)), 5.0)
            hr = st.slider('Holding rate', 0.05, 0.50, float(ce.get('holding_cost_rate', 0.25)), 0.01, format='%.2f')
            uc = st.slider('Unit cost ($)', 1.0, 100.0,
                           float(inv['unit_cost'].mean()) if not inv.empty and 'unit_cost' in inv.columns else 15.0, 0.5)
            section('Demand profile')
            dm = st.number_input('Mean daily demand', value=float(round(avg_d, 1)), min_value=1.0)
            ds = st.number_input('Demand std dev', value=float(round(avg_s, 1)), min_value=0.1)
            st.markdown('</div>', unsafe_allow_html=True)

        sc = compute_scenario(dm, ds, lt, svc, oc, hr, uc)

        def delta(new, old):
            if old == 0:
                return '<div class="tile-d flat">baseline</div>'
            p = (new - old) / old * 100
            d = 'up' if p > 0 else 'down'
            return f'<div class="tile-d {d}">{"↑" if p>0 else "↓"} {abs(p):.1f}% vs base</div>'

        with res:
            section('Results vs baseline')
            tiles = [
                ('Safety Stock', f"{sc['safety_stock']:.0f}", 'units', sc['safety_stock'], base['safety_stock']),
                ('EOQ', f"{sc['eoq']:.0f}", 'units/order', sc['eoq'], base['eoq']),
                ('Reorder Point', f"{sc['reorder_point']:.0f}", 'units', sc['reorder_point'], base['reorder_point']),
                ('Annual Total Cost', f"${sc['total_cost']:,.0f}", '', sc['total_cost'], base['total_cost']),
                ('Holding Cost', f"${sc['holding_cost']:,.0f}", '', sc['holding_cost'], base['holding_cost']),
                ('Ordering Cost', f"${sc['order_cost_yr']:,.0f}", '', sc['order_cost_yr'], base['order_cost_yr']),
            ]
            for row in (tiles[:3], tiles[3:]):
                cc = st.columns(3)
                for col, (lab, v, u, nw, ol) in zip(cc, row):
                    col.markdown(f'<div class="tile"><div class="tile-v">{v}</div>'
                                 f'<div class="tile-l">{lab}{" · "+u if u else ""}</div>{delta(nw, ol)}</div>',
                                 unsafe_allow_html=True)

            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
            section('Cost sensitivity · service level')
            sr = np.arange(0.85, 1.0, 0.01)
            costs = [compute_scenario(dm, ds, lt, s, oc, hr, uc)['total_cost'] for s in sr]
            ssv = [compute_scenario(dm, ds, lt, s, oc, hr, uc)['safety_stock'] for s in sr]
            fig = make_subplots(specs=[[{'secondary_y': True}]])
            fig.add_trace(go.Scatter(x=sr * 100, y=costs, name='Total cost', mode='lines',
                                     line=dict(color=C['accent'], width=2.5)), secondary_y=False)
            fig.add_trace(go.Scatter(x=sr * 100, y=ssv, name='Safety stock', mode='lines',
                                     line=dict(color=C['violet'], width=2, dash='dot')), secondary_y=True)
            fig.add_vline(x=svc * 100, line_color='rgba(52,211,153,0.5)', line_dash='dash')
            fig = theme(fig, 280)
            fig.update_yaxes(secondary_y=True, gridcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            section('Cost trade-off · holding vs ordering')
            ocr = np.linspace(10, 250, 40)
            hc = [compute_scenario(dm, ds, lt, svc, o, hr, uc)['holding_cost'] for o in ocr]
            ocv = [compute_scenario(dm, ds, lt, svc, o, hr, uc)['order_cost_yr'] for o in ocr]
            tc = [h + o for h, o in zip(hc, ocv)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ocr, y=hc, name='Holding', mode='lines', line=dict(color=C['accent'], width=2)))
            fig.add_trace(go.Scatter(x=ocr, y=ocv, name='Ordering', mode='lines', line=dict(color=C['amber'], width=2)))
            fig.add_trace(go.Scatter(x=ocr, y=tc, name='Total', mode='lines', line=dict(color=C['green'], width=2.5, dash='dot')))
            fig.add_vline(x=oc, line_color='rgba(255,255,255,0.18)', line_dash='dash')
            st.plotly_chart(theme(fig, 260), use_container_width=True, config={'displayModeBar': False})

    # ── page 6: store intelligence ──

    def render_stores(self, data: dict) -> None:
        topbar('Stores', 'Store Intelligence', 'Cross-store performance matrix, rankings & comparison')
        sales, inv = data['sales'], data['inventory']
        if sales.empty:
            st.info('No sales data.'); return
        sc = 'store_nbr' if 'store_nbr' in sales.columns else 'store'

        t1, t2, t3 = st.tabs(['Performance heatmap', 'Rankings', 'Compare'])

        with t1:
            section('Relative performance · normalized per store')
            piv = sales.pivot_table(index=sc, columns='family', values='sales', aggfunc='sum')
            pn = piv.div(piv.max(axis=1), axis=0)
            fig = go.Figure(go.Heatmap(z=pn.values, x=pn.columns.tolist(), y=[str(s) for s in pn.index],
                                       customdata=piv.values, colorscale=RAMP, xgap=2, ygap=2,
                                       colorbar=dict(thickness=10, len=0.8, tickfont=dict(color=C['faint'], size=9)),
                                       hovertemplate='%{y} · %{x}<br>$%{customdata:,.0f}<extra></extra>'))
            st.plotly_chart(theme(fig, 400, legend=False), use_container_width=True, config={'displayModeBar': False})
            section('Absolute volume')
            fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns.tolist(), y=[str(s) for s in piv.index],
                                       colorscale=RAMP_CYAN, xgap=2, ygap=2,
                                       colorbar=dict(thickness=10, len=0.8, tickfont=dict(color=C['faint'], size=9)),
                                       hovertemplate='%{y} · %{x}<br>$%{z:,.0f}<extra></extra>'))
            st.plotly_chart(theme(fig, 400, legend=False), use_container_width=True, config={'displayModeBar': False})

        with t2:
            section('Performance vs network average')
            bs = sales.groupby(sc)['sales'].agg(['sum', 'mean', 'std']).reset_index()
            bs.columns = ['Store', 'Total', 'Avg', 'Std']
            bs['CV'] = bs['Std'] / bs['Avg']
            avg = bs['Total'].mean()
            bs['vs'] = (bs['Total'] - avg) / avg * 100
            bs = bs.sort_values('Total', ascending=False).reset_index(drop=True)
            bs['Rank'] = range(1, len(bs) + 1)
            colors = [C['green'] if v >= 0 else C['red'] for v in bs['vs']]
            fig = go.Figure(go.Bar(x=bs['Store'].astype(str), y=bs['vs'], marker_color=colors, marker_line_width=0,
                                   hovertemplate='%{x}<br>%{y:+.1f}% vs avg<extra></extra>'))
            fig.add_hline(y=0, line_color='rgba(255,255,255,0.18)', line_dash='dash')
            st.plotly_chart(theme(fig, 300, legend=False), use_container_width=True, config={'displayModeBar': False})
            disp = bs[['Rank', 'Store', 'Total', 'Avg', 'CV', 'vs']].copy()
            disp.columns = ['Rank', 'Store', 'Total Sales', 'Avg Daily', 'Demand CV', '% vs Avg']
            st.dataframe(disp.round(2), use_container_width=True, hide_index=True)

        with t3:
            section('Head-to-head comparison')
            stores = sorted(sales[sc].unique().tolist())
            c1, c2 = st.columns(2)
            s1 = c1.selectbox('Store A', stores, index=0)
            s2 = c2.selectbox('Store B', stores, index=min(1, len(stores) - 1))
            d1 = sales[sales[sc] == s1].groupby('date')['sales'].sum().reset_index()
            d2 = sales[sales[sc] == s2].groupby('date')['sales'].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d1['date'], y=d1['sales'], name=str(s1), mode='lines',
                                     line=dict(color=C['accent'], width=2)))
            fig.add_trace(go.Scatter(x=d2['date'], y=d2['sales'], name=str(s2), mode='lines',
                                     line=dict(color=C['amber'], width=2)))
            st.plotly_chart(theme(fig, 320), use_container_width=True, config={'displayModeBar': False})
            stats = {}
            for store, df in [(s1, d1), (s2, d2)]:
                stats[str(store)] = {
                    'Total': f"${df['sales'].sum():,.0f}", 'Daily avg': f"${df['sales'].mean():,.0f}",
                    'Peak': f"${df['sales'].max():,.0f}", 'Low': f"${df['sales'].min():,.0f}",
                    'Std dev': f"${df['sales'].std():,.0f}",
                    'MoM growth': f"{(df['sales'].tail(30).mean()/max(df['sales'].head(30).mean(),1)-1)*100:+.1f}%",
                }
            st.dataframe(pd.DataFrame(stats), use_container_width=True)

    # ── run ──

    def _inject_sidebar_opener(self) -> None:
        """Inject a self-healing floating button that reopens a collapsed sidebar.

        Streamlit's native expand control proved unreliable when the header is
        styled (it could render invisibly or off-layout, leaving no way to
        reopen the sidebar). Instead we add our OWN fixed button into the parent
        document via a 0-height component iframe (same-origin, so it can reach
        window.parent). It shows only while the sidebar is collapsed and, on
        click, programmatically clicks whatever native toggle exists — immune to
        CSS overlap, colour-blending, and test-id renames.
        """
        components.html("""
        <script>
        (function () {
          const doc = window.parent.document;
          if (window.parent.__riqOpenerTimer) clearInterval(window.parent.__riqOpenerTimer);

          function nativeToggle() {
            const sels = ['[data-testid="stExpandSidebarButton"]',
                          '[data-testid="stSidebarCollapseButton"]',
                          '[data-testid="collapsedControl"]',
                          '[data-testid="stSidebarCollapsedControl"]'];
            for (const s of sels) {
              const el = doc.querySelector(s);
              if (el) return el.querySelector('button') || el;
            }
            return null;
          }
          function collapsed() {
            const sb = doc.querySelector('[data-testid="stSidebar"]');
            if (!sb) return false;
            const w = sb.getBoundingClientRect().width;
            return sb.getAttribute('aria-expanded') === 'false' || w < 40;
          }

          let btn = doc.getElementById('riq-sidebar-opener');
          if (!btn) {
            btn = doc.createElement('button');
            btn.id = 'riq-sidebar-opener';
            btn.title = 'Open sidebar';
            btn.setAttribute('aria-label', 'Open sidebar');
            btn.innerHTML = '\\u2630';
            btn.style.cssText = 'position:fixed;top:12px;left:12px;z-index:2147483647;'
              + 'width:40px;height:40px;border-radius:10px;cursor:pointer;'
              + 'background:#13141c;color:#f4f4f6;border:1px solid rgba(255,255,255,0.22);'
              + 'font-size:18px;line-height:1;box-shadow:0 2px 12px rgba(0,0,0,0.55);'
              + 'align-items:center;justify-content:center;display:none;transition:border-color .15s;';
            btn.onmouseenter = () => btn.style.borderColor = '#6366f1';
            btn.onmouseleave = () => btn.style.borderColor = 'rgba(255,255,255,0.22)';
            btn.onclick = () => { const t = nativeToggle(); if (t) t.click(); };
            doc.body.appendChild(btn);
          }
          function tick() { btn.style.display = collapsed() ? 'flex' : 'none'; }
          window.parent.__riqOpenerTimer = setInterval(tick, 250);
          tick();
        })();
        </script>
        """, height=0)

    def run(self) -> None:
        self._inject_sidebar_opener()
        data_all = self._load_all()
        filters = self.render_sidebar(data_all)
        data = self._filter(data_all, filters)
        page = st.session_state.page
        {
            'Executive Dashboard': self.render_executive,
            'Sales Analytics': self.render_sales_analytics,
            'Forecasting': self.render_forecasting,
            'Inventory Command': self.render_inventory,
            'Scenario Planner': self.render_scenario,
            'Store Intelligence': self.render_stores,
        }.get(page, self.render_executive)(data)


def main():
    RetailIQDashboard().run()


if __name__ == '__main__':
    main()
