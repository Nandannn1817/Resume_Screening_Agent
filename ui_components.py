import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Any


def inject_css(theme: str = "dark"):
    """Inject minimal theme-aware CSS into Streamlit app.

    theme: 'dark' or 'light'
    This function uses safe Python string constants and does not include
    any stray code outside strings so static analyzers won't fail.
    """
    if theme.lower() == "light":
        vars = {
            "bg": "#f7f78",
            "panel": "#ffffff",
            "muted": "#6b7280",
            "accent1": "#00CC96",
            "accent2": "#636EFA",
            "text": "#0b1220",
        }
    else:
        vars = {
            "bg": "#0f1112",
            "panel": "#111214",
            "muted": "#9aa0a6",
            "accent1": "#00CC96",
            "accent2": "#636EFA",
            "text": "#e6eef8",
        }

    css = f"""
    <style>
    :root {{ --bg: {vars['bg']}; --panel: {vars['panel']}; --muted: {vars['muted']}; --accent1: {vars['accent1']}; --accent2: {vars['accent2']}; --text: {vars['text']} }}
    body {{background: var(--bg); color: var(--text);}}
    .card {{background: var(--panel); border-radius: 10px; padding: 16px; margin-bottom: 14px; box-shadow: 0 6px 18px rgba(0,0,0,0.06);}}
    .card:hover {{transform: translateY(-4px); transition: all 0.18s ease;}}
    .card-header {{display:flex; align-items:center; justify-content:space-between}}
    .card-title {{font-size:18px; font-weight:700; color:var(--text)}}
    .skill-chip {{display:inline-block; background:rgba(0,0,0,0.06); color:var(--text); padding:6px 10px; border-radius:12px; margin:4px; font-size:12px}}
    .progress {{height:10px; background:rgba(0,0,0,0.06); border-radius:6px; overflow:hidden}}
    .progress > .bar {{height:10px; background:linear-gradient(90deg,{vars['accent1']},{vars['accent2']})}}
    .small-muted {{color:var(--muted); font-size:13px}}
    .top-title {{font-family: 'Helvetica Neue', Arial, sans-serif; font-size:36px; color:var(--text); font-weight:800; margin:6px 0}}
    .subtitle {{color:var(--muted); margin-bottom:18px}}
    .action-btn {{background:linear-gradient(90deg,{vars['accent2']},{vars['accent1']}); color:white; padding:8px 12px; border-radius:8px; text-decoration:none; border:none}}
    .grid {{display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:14px}}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


def render_match_donut(match_percent: float, title: str = "Skill Match"):
    """Render a donut chart showing match percentage (0-100)."""
    # Ensure value safe
    v = max(0.0, min(100.0, float(match_percent or 0)))
    remaining = 100.0 - v

    fig = go.Figure(data=[
        go.Pie(values=[v, remaining], hole=0.6, marker_colors=['#00CC96', 'rgba(255,255,255,0.06)'], sort=False, direction='clockwise', hoverinfo='none')
    ])
    fig.update_traces(textinfo='none')
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', annotations=[
        dict(text=f"<b>{v:.0f}%</b>", x=0.5, y=0.5, font=dict(size=20, color='var(--text)'), showarrow=False)
    ], title_text=title)

    st.plotly_chart(fig, use_container_width=True, theme='streamlit')


def render_resume_quality_gauge(score: float, title: str = "Resume Quality"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(0, min(100, float(score or 0))),
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': '#00CC96'},
               'bgcolor': '#222',
               'steps': [
                   {'range': [0, 50], 'color': '#FF6B6B'},
                   {'range': [50, 75], 'color': '#FFD166'},
                   {'range': [75, 100], 'color': '#00CC96'}
               ]}
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', title_text=title)
    st.plotly_chart(fig, use_container_width=True, theme='streamlit')


def render_skill_match_bars(skill_counts_jd: Dict[str, int], skill_counts_resume: Dict[str, int], top_n: int = 12):
    keys = list(skill_counts_jd.keys())
    keys_sorted = sorted(keys, key=lambda k: skill_counts_jd.get(k, 0), reverse=True)[:top_n]
    jd_vals = [skill_counts_jd.get(k, 0) for k in keys_sorted]
    res_vals = [skill_counts_resume.get(k, 0) for k in keys_sorted]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=keys_sorted, x=jd_vals, orientation='h', name='JD', marker_color='#636EFA',
                         hovertemplate='%{y}: %{x} occurrences in JD'))
    fig.add_trace(go.Bar(y=keys_sorted, x=res_vals, orientation='h', name='Resume', marker_color='#00CC96',
                         hovertemplate='%{y}: %{x} occurrences across resumes'))
    fig.update_layout(barmode='group', height=420, margin=dict(l=220, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, theme='streamlit')


def candidate_card(candidate: Dict[str, Any], key_prefix: str = "candidate"):
    name = candidate.get('Filename', 'Unknown')
    score = candidate.get('Score', 0)
    analysis_raw = candidate.get('AI_Analysis', '{}')
    try:
        import json as _json
        parsed = _json.loads(analysis_raw) if isinstance(analysis_raw, str) else (analysis_raw or {})
    except Exception:
        parsed = {}

        skills = parsed.get('skills', {}).get('present', []) if parsed else []
        years = parsed.get('experience', {}).get('years', 0) if parsed else 0

        # Precompute small pieces to avoid nested f-strings in the large HTML block
        initial = (name[0].upper() if name else '?')
        chips_html = ''.join([f"<span class='skill-chip'>{s}</span>" for s in skills[:8]])

        html = f"""
        <div class='card'>
            <div class='card-header'>
                <div style='display:flex; gap:12px; align-items:center'>
                    <div style='width:56px; height:56px; border-radius:8px; background:linear-gradient(90deg,#636EFA,#00CC96); display:flex; align-items:center; justify-content:center; color:white; font-weight:700'>{initial}</div>
                    <div>
                        <div class='card-title'>{name}</div>
                        <div class='small-muted'>Experience: {years} yrs</div>
                    </div>
                </div>
                <div style='text-align:right'>
                    <div style='font-weight:700; font-size:18px; color:var(--text)'>{score}%</div>
                    <div style='width:140px' class='progress'><div class='bar' style='width:{min(100, max(0, score))}%'></div></div>
                </div>
            </div>
            <div style='margin-top:10px'>
                {chips_html}
            </div>
            <div style='margin-top:12px; display:flex; gap:8px; justify-content:flex-end'>
                <button class='action-btn' type='button'>⭐ Shortlist</button>
            </div>
        </div>
        """

    st.markdown(html, unsafe_allow_html=True)
    # Inline insights expander
    with st.expander(f"Insights — {name}"):
        st.write(parsed if parsed else analysis_raw)
        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.button("⭐ Shortlist", key=f"shortlist_{key_prefix}_{name}"):
                if 'shortlisted' not in st.session_state:
                    st.session_state.shortlisted = []
                if candidate not in st.session_state.shortlisted:
                    st.session_state.shortlisted.append(candidate)
                    st.success(f"{name} shortlisted")
                else:
                    st.warning("Already shortlisted")
        with cols[1]:
            if st.button("📄 Export JSON", key=f"export_{key_prefix}_{name}"):
                import json as _json
                payload = _json.dumps(parsed if parsed else {})
                st.download_button("Download JSON", payload, file_name=f"{name}_analysis.json", mime='application/json')
        with cols[2]:
            if st.button("✉️ Share", key=f"share_{key_prefix}_{name}"):
                st.info("Share feature not yet configured.")


def render_leaderboard(candidates: List[Dict[str, Any]], top_n: int = 5):
    st.markdown("<div style='padding:6px'><b>Top Candidates</b></div>", unsafe_allow_html=True)
    for i, cand in enumerate(sorted(candidates, key=lambda x: x.get('Score', 0), reverse=True)[:top_n]):
                st.markdown(f"<div style='padding:6px 8px'>{i+1}. <b>{cand.get('Filename','Unknown')}</b> — <span style='color:#00CC96'>{cand.get('Score',0)}%</span></div>", unsafe_allow_html=True)

