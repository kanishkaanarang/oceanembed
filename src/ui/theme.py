"""Palette helpers for canvas-backed tables that do not inherit page CSS."""
import pandas as pd
import streamlit as st


def themed_table(frame: pd.DataFrame, dark: bool):
    """Style cell foreground/background without changing data or column formats.

    Streamlit supports these pandas Styler properties in its interactive grid.
    Native grid headers retain their own high-contrast header treatment.
    """
    return frame.style.format(precision=3, na_rep="—").set_properties(**{
        "background-color": "#0c2433" if dark else "#ffffff",
        "color": "#e7f4f8" if dark else "#0f172a",
    })


def render_chart(figure, **kwargs):
    """Keep Plotly labels/hover text in sync with the sidebar theme toggle."""
    dark = st.session_state.get("dark_mode", False)
    text = "#e7f4f8" if dark else "#0f172a"
    figure.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        font_color=text, legend_font_color=text,
        hoverlabel=dict(bgcolor="#113348" if dark else "#ffffff", font_color=text),
    )
    # Respect explicitly colored axes in the multivariable T-S-c plot.
    for axis in list(figure.select_xaxes()) + list(figure.select_yaxes()):
        if axis.tickfont.color is None:
            axis.tickfont.color = text
        if axis.title.font.color is None:
            axis.title.font.color = text
    # The native Streamlit theme does not follow a CSS sidebar toggle.
    kwargs.setdefault("theme", None)
    return st.plotly_chart(figure, **kwargs)


def apply_dashboard_theme(dark: bool) -> None:
    palette = (
        ("#081922", "#102832", "#183743", "#e7f2f5", "#a0bbc5", "#294955", "#45d4cc")
        if dark else
        ("#f3f7f8", "#ffffff", "#eaf1f3", "#122f3b", "#526e7a", "#d9e5e9", "#087f83")
    )
    bg, panel, muted_bg, text, muted, border, accent = palette
    badge_rules = """
    .stMarkdownBadge { color: #c8e5fa !important; }
    .st-key-mission-risk-low .stMarkdownBadge { color: #76e1af !important; }
    .st-key-mission-risk-moderate .stMarkdownBadge { color: #ffd18b !important; }
    .st-key-mission-risk-high .stMarkdownBadge { color: #ffaaa8 !important; }
    .st-key-mission-risk-undetermined .stMarkdownBadge { color: #c1ced3 !important; }
    """ if dark else ""
    st.html(f"""<style>
    .stApp {{
      --background-color: {bg}; --secondary-background-color: {muted_bg};
      --text-color: {text}; --primary-color: {accent}; --border-color: {border};
      --oe-panel: {panel}; --oe-muted: {muted};
      background: {bg}; color: {text};
    }}
    [data-testid="stHeader"] {{ background: {bg}; }}
    [data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {border}; }}
    [data-testid="stMainBlockContainer"] {{ max-width: 1600px; padding: 2.6rem 2.4rem 4rem; }}
    h1, h2, h3, [data-testid="stMarkdownContainer"], label {{ color: {text}; }}
    h1 {{ letter-spacing: -.045em; }}
    h2 {{ letter-spacing: -.025em; font-size: 1.5rem !important; }}
    h3 {{ letter-spacing: -.02em; font-size: 1.16rem !important; }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: {muted} !important; opacity: 1; }}
    [data-testid="stVerticalBlockBorderWrapper"] > div {{ border-color: {border}; }}
    div[class*="st-key-"][class*="-card"] {{
      background: {panel}; border-color: {border}; border-radius: 18px;
    }}
    [data-testid="stMetricLabel"] p {{ color: {muted}; font-size: .81rem; }}
    [data-testid="stMetricValue"] {{ color: {text}; font-size: 1.65rem; font-weight: 650; letter-spacing: -.035em; }}
    .st-key-depth-cards [data-testid="stMetric"], .st-key-surface-cards [data-testid="stMetric"],
    .st-key-comparison-cards [data-testid="stMetric"] {{
      background: {panel}; border: 1px solid {border}; border-radius: 14px; padding: 1rem;
    }}
    [data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"],
    [data-testid="stNumberInputContainer"] {{ background: {muted_bg}; color: {text}; border-color: {border}; }}
    input, [data-baseweb="select"] span {{ color: {text}; }}
    [role="group"]:has(> input[role="combobox"]), input[role="combobox"],
    [role="group"]:has(> input[role="combobox"]) button {{
      background: {muted_bg} !important; color: {text} !important; border-color: {border};
    }}
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
    [data-testid="stBaseButton-segmented_control"] {{
      background: {panel}; color: {text}; border-color: {border}; border-radius: 10px;
    }}
    [data-testid="stDownloadButton"] button:hover, [data-testid="stButton"] button:hover {{
      border-color: {accent}; color: {accent};
    }}
    [data-testid="stTabs"] [role="tablist"] {{ gap: 1.5rem; border-bottom: 1px solid {border}; }}
    [data-testid="stTabs"] [role="tab"] {{ color: {muted}; font-weight: 600; }}
    [data-testid="stTabs"] [aria-selected="true"] {{ color: {accent}; }}
    [data-testid="stExpander"] {{ background: {panel}; border-color: {border}; border-radius: 14px; }}
    [data-testid="stExpander"] details, [data-testid="stExpander"] summary {{
      background: {panel} !important; color: {text} !important; border-color: {border};
    }}
    {badge_rules}
    .st-key-hero {{
      position: relative; overflow: hidden; padding: 1.8rem 2rem; border-radius: 22px;
      background: radial-gradient(ellipse at 95% 15%, #156773 0%, transparent 48%), #0b2c3a;
      border: 1px solid #234d5a; margin-bottom: .5rem;
    }}
    .st-key-hero h1 {{ color: #f1fbfd; font-size: clamp(2rem, 3.3vw, 3.1rem); line-height: 1.08; max-width: 720px; }}
    .st-key-hero [data-testid="stMarkdownContainer"] {{ color: #c5e0e8; }}
    .st-key-hero [data-testid="stCaptionContainer"], .st-key-hero [data-testid="stCaptionContainer"] p {{ color: #b7d8e2 !important; }}
    .st-key-hero span {{ color: #c5e0e8 !important; }}
    .st-key-hero-status {{ background: #113b49; border-color: #38606a; border-radius: 14px; }}
    .st-key-hero-status [data-testid="stMetricValue"] {{ color: #ffffff; }}
    .st-key-hero-status [data-testid="stMetricLabel"] p {{ color: #b7d8e2; }}
    .st-key-mission-cyclone-card {{ border-top: 3px solid #e7a354 !important; }}
    .st-key-mission-acoustic-card {{ border-top: 3px solid {accent} !important; }}
    .st-key-mission-physics-card {{ border-top: 3px solid #41b59a !important; }}
    .oe-eyebrow {{ color: #85cdd0; font-size: .72rem; letter-spacing: .16em; font-weight: 700; margin-bottom: .7rem; }}
    .oe-scale {{ display: grid; grid-template-columns: 5fr 3fr 4fr; gap: 3px; margin: .5rem 0 1rem; font-size: .68rem; color: {muted}; }}
    .oe-scale span {{ padding-top: .4rem; border-top: 5px solid; }}
    @media (max-width: 768px) {{
      [data-testid="stMainBlockContainer"] {{ padding: 1.2rem 1rem 3rem; }}
      .st-key-hero {{ padding: 1.4rem; }}
      [data-testid="stMetricValue"] {{ font-size: 1.4rem; }}
      [data-testid="stTabs"] [role="tablist"] {{ gap: .6rem; }}
    }}
    @media (prefers-reduced-motion: no-preference) {{
      .st-key-hero {{ animation: ocean-arrive .4s ease-out; }}
      @keyframes ocean-arrive {{ from {{ opacity: .3; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}
    }}
    </style>""")
