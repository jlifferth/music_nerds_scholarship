"""
Music Nerds Scoring Affinity Analysis — Streamlit App
Run locally:  streamlit run app.py
Deploy:       push app.py + requirements.txt + data file to GitHub,
              then connect to share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Nerds Scoring Affinity",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
# Theme (background, text colour) is controlled by .streamlit/config.toml
st.markdown("""
<style>
  .section-note {
    font-size: 0.88rem;
    color: #555;
    background: #f4f4f4;
    border-left: 3px solid #bbb;
    padding: 8px 12px;
    margin-bottom: 14px;
    border-radius: 0 4px 4px 0;
  }
  .widget-answer {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 14px;
    margin-top: -6px;
  }
  h2 { margin-top: 0.4rem !important; margin-bottom: 2px !important; }
  .stTabs [data-baseweb="tab"] { font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# ── Marker symbols ────────────────────────────────────────────────────────────
SYMBOLS = [
    "circle", "square", "diamond", "triangle-up", "triangle-down",
    "pentagon", "hexagon", "star", "cross", "x",
    "triangle-left", "triangle-right", "hourglass",
]


# ── Helper: big question + answer heading ─────────────────────────────────────
def widget_header(question, answer):
    st.markdown(f"## {question}")
    st.markdown(f'<div class="widget-answer">{answer}</div>', unsafe_allow_html=True)


# ── Data loading & processing ─────────────────────────────────────────────────
@st.cache_data
def load_and_process(path="Music Nerds - Scores.tsv"):
    df_raw = pd.read_csv(path, sep="\t")
    df = df_raw[df_raw["Curator"].notna() & df_raw["Deadline"].notna()].copy()
    df["Deadline_dt"] = pd.to_datetime(df["Deadline"] + "/2026", format="%m/%d/%Y")
    df = df.sort_values("Deadline_dt").reset_index(drop=True)

    SCORER_COLS = [
        "Nathan", "Rachel", "Emily", "Jonathan", "Tera", "Erik", "Nana",
        "Goose", "Kristin", "Jacob", "Andreina", "George", "Jane", "Tyler",
        "Molly", "Megan", "Anna W",
    ]

    MUTUAL = sorted(set(df["Curator"].unique()) & set(SCORER_COLS))
    symbol_map = {name: SYMBOLS[i % len(SYMBOLS)] for i, name in enumerate(MUTUAL)}

    # ── Z-score normalise per scorer ──────────────────────────────────────────
    z_scores = df[SCORER_COLS].apply(lambda col: (col - col.mean()) / col.std(), axis=0)
    df_z = df.copy()
    for s in SCORER_COLS:
        df_z[f"z_{s}"] = z_scores[s]

    # ── Cross-score matrices ──────────────────────────────────────────────────
    cross_z   = pd.DataFrame(index=MUTUAL, columns=MUTUAL, dtype=float)
    cross_raw = pd.DataFrame(index=MUTUAL, columns=MUTUAL, dtype=float)
    for curator in MUTUAL:
        c_songs = df_z[df_z["Curator"] == curator]
        for scorer in MUTUAL:
            if scorer == curator:
                cross_z.loc[curator, scorer]   = np.nan
                cross_raw.loc[curator, scorer] = np.nan
            else:
                vz = c_songs[f"z_{scorer}"].dropna()
                vr = c_songs[scorer].dropna()
                cross_z.loc[curator, scorer]   = vz.mean() if len(vz) else np.nan
                cross_raw.loc[curator, scorer] = vr.mean() if len(vr) else np.nan

    # ── Directed pairs ────────────────────────────────────────────────────────
    pairs = []
    for a in MUTUAL:
        for b in MUTUAL:
            if a == b:
                continue
            agb_z   = cross_z.loc[b, a]
            bga_z   = cross_z.loc[a, b]
            agb_raw = cross_raw.loc[b, a]
            bga_raw = cross_raw.loc[a, b]
            if pd.notna(agb_z) and pd.notna(bga_z):
                pairs.append(dict(
                    scorer=a, curator=b,
                    z_given=agb_z,     z_received=bga_z,
                    raw_given=agb_raw, raw_received=bga_raw,
                    symbol=symbol_map[a],
                ))
    pair_df = pd.DataFrame(pairs)

    # ── Temporal observations ─────────────────────────────────────────────────
    events = []
    for _, row in df_z.iterrows():
        deadline = row["Deadline_dt"]
        curator  = row["Curator"]
        song     = row["Song"]
        for scorer in SCORER_COLS:
            z_given   = row[f"z_{scorer}"]
            raw_given = row[scorer]
            if pd.isna(z_given):
                continue
            prior = []
            if scorer in MUTUAL:
                prior_songs = df_z[
                    (df_z["Curator"] == scorer) & (df_z["Deadline_dt"] < deadline)
                ]
                for _, ps in prior_songs.iterrows():
                    prior += [ps[f"z_{s}"] for s in SCORER_COLS
                               if s != scorer and pd.notna(ps[f"z_{s}"])]
            events.append(dict(
                deadline=deadline, curator=curator, song=song, scorer=scorer,
                z_given=z_given, raw_given=raw_given,
                n_prior=len(prior),
                cum_received_avg=np.mean(prior) if prior else np.nan,
                symbol=symbol_map.get(scorer, "circle"),
            ))
    events_df = pd.DataFrame(events)
    has_history = events_df[events_df["n_prior"] > 0].dropna(subset=["cum_received_avg"])

    # ── Permutation tests ─────────────────────────────────────────────────────
    np.random.seed(42)
    N_PERM = 10_000

    x_t, y_t = pair_df["z_received"].values, pair_df["z_given"].values
    obs_r_t  = np.corrcoef(x_t, y_t)[0, 1]
    perm_t   = [np.corrcoef(np.random.permutation(x_t), y_t)[0, 1] for _ in range(N_PERM)]
    perm_p_t = np.mean(np.abs(perm_t) >= np.abs(obs_r_t))

    x_g, y_g = has_history["cum_received_avg"].values, has_history["z_given"].values
    obs_r_g  = np.corrcoef(x_g, y_g)[0, 1]
    perm_g   = [np.corrcoef(np.random.permutation(x_g), y_g)[0, 1] for _ in range(N_PERM)]
    perm_p_g = np.mean(np.abs(perm_g) >= np.abs(obs_r_g))

    # ── Per-scorer round trend ────────────────────────────────────────────────
    round_fx = []
    for scorer in MUTUAL:
        se = events_df[events_df["scorer"] == scorer].copy().sort_values("deadline")
        se["round_num"] = se["deadline"].rank(method="dense")
        if len(se) < 4:
            continue
        rho, p = stats.spearmanr(se["round_num"], se["z_given"])
        round_fx.append(dict(scorer=scorer, rho=rho, p=p, n=len(se)))
    re_df = pd.DataFrame(round_fx).sort_values("rho")

    # ── Weekly average given per scorer ──────────────────────────────────────
    weekly = (
        events_df.groupby(["scorer", "deadline"])
        .agg(weekly_z_mean=("z_given", "mean"), weekly_raw_mean=("raw_given", "mean"))
        .reset_index()
        .sort_values("deadline")
    )
    scorer_overall_z = events_df.groupby("scorer")["z_given"].mean().to_dict()

    # ── Overall averages + boxplot data ──────────────────────────────────────
    rows = []
    box_rows = []
    for scorer in SCORER_COLS:
        assigned_vals = df[scorer].dropna().values
        avg_assigned  = float(np.mean(assigned_vals)) if len(assigned_vals) else np.nan
        for v in assigned_vals:
            box_rows.append(dict(Participant=scorer, Type="Assigned", Score=float(v)))
        if scorer in MUTUAL:
            c_songs   = df_z[df_z["Curator"] == scorer]
            others    = [s for s in SCORER_COLS if s != scorer]
            recv_vals = c_songs[others].values.flatten()
            recv_vals = recv_vals[~np.isnan(recv_vals)]
            avg_recv  = float(np.mean(recv_vals)) if len(recv_vals) else np.nan
            for v in recv_vals:
                box_rows.append(dict(Participant=scorer, Type="Received", Score=float(v)))
        else:
            avg_recv = np.nan
        rows.append(dict(
            Participant=scorer,
            is_mutual=scorer in MUTUAL,
            avg_assigned_raw=round(avg_assigned, 2) if not np.isnan(avg_assigned) else None,
            avg_received_raw=round(avg_recv,    2) if not np.isnan(avg_recv)    else None,
        ))
    overall_df = pd.DataFrame(rows)
    box_df     = pd.DataFrame(box_rows)

    box_order = (
        box_df[box_df["Type"] == "Assigned"]
        .groupby("Participant")["Score"].median()
        .sort_values(ascending=False).index.tolist()
    )

    # ── Reciprocity table — ratio is A/B (how A scored B relative to how B scored A) ──
    seen = set()
    recip_rows = []
    for _, row in pair_df.iterrows():
        key = tuple(sorted([row["scorer"], row["curator"]]))
        if key in seen:
            continue
        seen.add(key)
        a, b    = row["scorer"], row["curator"]
        raw_a2b = cross_raw.loc[b, a]   # A's avg raw score on B's songs
        raw_b2a = cross_raw.loc[a, b]   # B's avg raw score on A's songs
        z_a2b   = cross_z.loc[b, a]
        z_b2a   = cross_z.loc[a, b]
        # Ratio = A/B: >1 means A was more generous to B than B was to A
        raw_ratio = (raw_a2b / raw_b2a
                     if pd.notna(raw_a2b) and pd.notna(raw_b2a) and raw_b2a != 0
                     else np.nan)
        z_ratio   = (z_a2b / z_b2a
                     if pd.notna(z_a2b) and pd.notna(z_b2a) and z_b2a != 0
                     else np.nan)
        recip_rows.append(dict(
            person_a=a, person_b=b,
            raw_a_to_b=round(raw_a2b,  2) if pd.notna(raw_a2b)  else None,
            raw_b_to_a=round(raw_b2a,  2) if pd.notna(raw_b2a)  else None,
            raw_ratio =round(raw_ratio, 3) if pd.notna(raw_ratio) else None,
            z_a_to_b  =round(z_a2b,   3) if pd.notna(z_a2b)    else None,
            z_b_to_a  =round(z_b2a,   3) if pd.notna(z_b2a)    else None,
            z_ratio   =round(z_ratio,  3) if pd.notna(z_ratio)   else None,
        ))
    recip_df = (pd.DataFrame(recip_rows)
                .sort_values("raw_ratio", ascending=False)
                .reset_index(drop=True))

    return dict(
        df=df, df_z=df_z,
        SCORER_COLS=SCORER_COLS, MUTUAL=MUTUAL, symbol_map=symbol_map,
        cross_z=cross_z, cross_raw=cross_raw,
        pair_df=pair_df, events_df=events_df, has_history=has_history,
        obs_r_t=obs_r_t, perm_p_t=perm_p_t, perm_t=perm_t,
        obs_r_g=obs_r_g, perm_p_g=perm_p_g, perm_g=perm_g,
        re_df=re_df, weekly=weekly, scorer_overall_z=scorer_overall_z,
        overall_df=overall_df, box_df=box_df, box_order=box_order,
        recip_df=recip_df,
    )


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    D = load_and_process()
except FileNotFoundError:
    st.error("⚠️ Data file not found. Make sure `Music Nerds - Scores.tsv` is in the same folder as `app.py`.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER & INTRO
# ══════════════════════════════════════════════════════════════════════════════
st.title("🎵 Music Nerds Scoring Affinity Analysis")

st.markdown("""
Do the scores you receive influence the scores you give?

Let's look at "targeted reciprocity" and "general mood effect".
""")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TABS  — order: w4, w3, w5, w6, w7, w1, w2, w8
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🎯 Targeted Reciprocity",
    "📋 Reciprocity Table",
    "🌡️ General Mood Effect",
    "📈 Trends Over Time",
    "⚖️ Given vs. Received",
    "📊 Overall Scores",
    "🟩 Cross-Score Matrix",
    "🔀 Permutation Tests",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 (w4) — Targeted Reciprocity Scatter
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    r, p = D["obs_r_t"], D["perm_p_t"]
    widget_header(
        "Does the score you receive from a particular person predict the score you give that person in return?",
        f"✅ Yes! Just a little bit.&nbsp;&nbsp;(r = {r:.3f})",
    )
    pair_df = D["pair_df"].copy()
    pair_df["hover_label"] = (
        "<b>" + pair_df["scorer"] + "</b> scored <b>" + pair_df["curator"] + "'s</b> songs"
    )

    fig_scat = px.scatter(
        pair_df,
        x="z_received", y="z_given",
        color="scorer",
        symbol="scorer",
        symbol_map={name: D["symbol_map"][name] for name in D["MUTUAL"]},
        custom_data=["hover_label", "scorer", "curator",
                     "z_received", "z_given", "raw_received", "raw_given"],
        labels={
            "z_received": "Score received from this person (z)",
            "z_given":    "Score given to this person (z)",
        },
        height=640,
    )
    fig_scat.update_traces(
        marker=dict(size=11, opacity=0.85, line=dict(color="white", width=0.8)),
        hovertemplate=(
            "%{customdata[0]}<br>"
            "z given: %{customdata[4]:.3f}  |  z received: %{customdata[3]:.3f}<br>"
            "raw given: %{customdata[6]:.2f}  |  raw received: %{customdata[5]:.2f}"
            "<extra></extra>"
        ),
        selector=dict(mode="markers"),
    )

    m, b_int = np.polyfit(pair_df["z_received"], pair_df["z_given"], 1)
    x_line   = np.linspace(pair_df["z_received"].min(), pair_df["z_received"].max(), 200)
    fig_scat.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b_int,
        mode="lines", name="Trend",
        line=dict(color="crimson", width=2, dash="dash"),
        hoverinfo="skip",
    ))
    fig_scat.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig_scat.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig_scat.add_annotation(
        x=0.03, y=0.97, xref="paper", yref="paper",
        text=f"r = {r:.3f}  |  perm-p = {p:.3f}",
        showarrow=False, font=dict(size=13),
        bgcolor="white", bordercolor="#aaa", borderwidth=1,
    )
    fig_scat.update_layout(
        margin=dict(t=20),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(title="Scorer", orientation="v"),
    )
    st.plotly_chart(fig_scat, use_container_width=True)
    st.markdown(
        '<div class="section-note">Each dot is a directed pair: person A scoring person B\'s music. '
        'The x-axis is how B rated A\'s music; the y-axis is how A rated B in return. '
        'A positive slope = affinity (friends reward friends); negative = retaliation. '
        'Shape and colour both encode the scorer.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"{len(pair_df)} directed pairs from {len(D['MUTUAL'])} mutual participants.  "
        f"r = {r:.3f} (positive = affinity direction), perm-p = {p:.3f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 (w3) — Reciprocity Table
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    widget_header(
        "Which pairs scored each other most — and least — symmetrically?",
        "📋 Here's every scoring pair, <em>ranked by how generously person A scored person B relative to how person B scored person A</em>",
    )
    recip = D["recip_df"].copy()
    recip.columns = [
        "Person A", "Person B",
        "A→B (raw)", "B→A (raw)", "Raw Ratio (A/B)",
        "A→B (z)",   "B→A (z)",   "Z Ratio (A/B)",
    ]
    recip.index = recip.index + 1

    def color_ratio(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        if val > 1.15:
            return "background-color: #c6efce; color: #276221"
        elif val < 0.87:
            return "background-color: #ffc7ce; color: #9c0006"
        return ""

    st.dataframe(
        recip.style.map(color_ratio, subset=["Raw Ratio (A/B)", "Z Ratio (A/B)"]),
        use_container_width=True,
        height=520,
    )
    st.markdown(
        '<div class="section-note">Each row is a pair who both curated and scored each other\'s music. '
        '<b>Raw Ratio (A/B)</b> = A\'s avg raw score toward B ÷ B\'s avg raw score toward A. '
        'Ratio > 1 means A was more generous to B than B was to A; < 1 means the reverse. '
        'Z Ratio uses z-score normalized values (note: z-ratios are less reliable when values are near zero). '
        'Sorted by raw ratio descending.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 (w5) — General Mood Effect
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    r_g, p_g = D["obs_r_g"], D["perm_p_g"]
    widget_header(
        "Do the scores you receive influence how you score everyone else in general?",
        f"🚫 Nope, not really.&nbsp;&nbsp;(r = {r_g:.3f})",
    )
    hist = D["has_history"].copy()
    hist["song_short"] = hist["song"].str[:45]
    hist["hover_label"] = (
        "<b>" + hist["scorer"] + "</b> scored <b>" + hist["curator"] + "'s</b> songs"
    )

    fig_mood = px.scatter(
        hist,
        x="cum_received_avg", y="z_given",
        color="scorer",
        symbol="scorer",
        symbol_map={name: D["symbol_map"][name] for name in D["MUTUAL"]
                    if name in hist["scorer"].unique()},
        custom_data=["hover_label", "scorer", "curator", "song_short",
                     "cum_received_avg", "z_given"],
        labels={
            "cum_received_avg": "Scorer's cumulative avg received (z)",
            "z_given": "Score given (z)",
        },
        height=540,
    )
    fig_mood.update_traces(
        marker=dict(size=8, opacity=0.75, line=dict(color="white", width=0.6)),
        hovertemplate=(
            "%{customdata[0]}<br>"
            "Song: %{customdata[3]}<br>"
            "z given: %{customdata[5]:.3f}  |  cum. avg received: %{customdata[4]:.3f}"
            "<extra></extra>"
        ),
        selector=dict(mode="markers"),
    )

    m3, b3 = np.polyfit(hist["cum_received_avg"], hist["z_given"], 1)
    x3 = np.linspace(hist["cum_received_avg"].min(), hist["cum_received_avg"].max(), 200)
    fig_mood.add_trace(go.Scatter(
        x=x3, y=m3 * x3 + b3, mode="lines", name="Trend",
        line=dict(color="crimson", width=2, dash="dash"), hoverinfo="skip",
    ))
    fig_mood.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig_mood.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig_mood.add_annotation(
        x=0.03, y=0.97, xref="paper", yref="paper",
        text=f"r = {r_g:.3f}  |  perm-p = {p_g:.3f}",
        showarrow=False, font=dict(size=13),
        bgcolor="white", bordercolor="#aaa", borderwidth=1,
    )
    fig_mood.update_layout(
        margin=dict(t=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_mood, use_container_width=True)
    st.markdown(
        '<div class="section-note">Each dot is one (song, scorer) observation where the scorer '
        'had already received feedback on their own curated songs before this deadline. '
        'X-axis = scorer\'s cumulative average received z-score up to that point; '
        'Y-axis = the z-score they gave this song. '
        'A negative slope would mean low received scores → lower scores given to others.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"r = {r_g:.3f}, perm-p = {p_g:.3f}.  "
        "The slope is essentially flat — no evidence that cumulative received scores "
        "shift how generously participants score others."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 (w6) — Trends Over Time
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    widget_header(
        "Do your scores change over time?",
        "↕️ Just a little! And not everyone moved in the same direction.",
    )
    re_df = D["re_df"].copy()
    re_df["bar_color"] = re_df["rho"].apply(lambda x: "#d62728" if x < 0 else "#2ca02c")

    fig_bar = go.Figure(go.Bar(
        x=re_df["rho"], y=re_df["scorer"],
        orientation="h",
        marker_color=re_df["bar_color"],
        customdata=re_df[["rho", "p", "n", "scorer"]].values,
        hovertemplate=(
            "<b>%{customdata[3]}</b><br>"
            "ρ = %{customdata[0]:.3f}<br>"
            "p = %{customdata[1]:.3f}<br>"
            "n observations = %{customdata[2]}<extra></extra>"
        ),
    ))
    fig_bar.add_vline(x=0, line_width=1.5, line_color="black")
    fig_bar.update_layout(
        title="Spearman ρ: Score Given vs. Round Number",
        xaxis_title="Spearman ρ  (red = scores lower over time, green = higher)",
        yaxis_title="",
        height=400, margin=dict(t=50),
        xaxis=dict(range=[-0.5, 0.5]),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Weekly Average Score Given (z-score)")

    all_scorers_options = D["MUTUAL"] + [s for s in D["SCORER_COLS"] if s not in D["MUTUAL"]]
    selected_scorers = st.multiselect(
        "Select participants to display",
        options=all_scorers_options,
        default=D["MUTUAL"][:6],
        key="line_select",
    )

    weekly   = D["weekly"].copy()
    scorer_z = D["scorer_overall_z"]

    fig_line = go.Figure()
    for scorer in selected_scorers:
        sw = weekly[weekly["scorer"] == scorer].sort_values("deadline")
        if sw.empty:
            continue
        overall_mean = scorer_z.get(scorer, 0)
        fig_line.add_trace(go.Scatter(
            x=sw["deadline"], y=sw["weekly_z_mean"],
            mode="lines+markers", name=scorer,
            hovertemplate=(
                f"<b>{scorer}</b><br>"
                "Week: %{x|%b %-d}<br>"
                "Avg z this week: %{y:.3f}<extra></extra>"
            ),
        ))
        fig_line.add_shape(
            type="line",
            x0=sw["deadline"].min(), x1=sw["deadline"].max(),
            y0=overall_mean, y1=overall_mean,
            line=dict(dash="dot", width=1, color="lightgray"),
        )

    fig_line.add_hline(y=0, line_dash="dash", line_color="black",
                        opacity=0.25, annotation_text="Group zero-line")
    fig_line.update_layout(
        title="Weekly Average Z-Score Given per Scorer",
        xaxis_title="Deadline", yaxis_title="Avg z-score given",
        height=480, margin=dict(t=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown(
        '<div class="section-note">Two views of how scoring behaviour evolves across rounds. '
        'The bar chart shows each participant\'s Spearman ρ between round number and score given '
        '(positive = more generous over time, negative = stricter). '
        'The line chart shows each scorer\'s weekly average z-score — use the selector to focus on specific people.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Dotted grey lines mark each scorer's personal overall average — "
        "deviations above/below indicate relatively generous or strict weeks for that person."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 (w7) — Scorer-Level Given vs. Received
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    pair_df_t5 = D["pair_df"]
    ss_rows = []
    for scorer in D["MUTUAL"]:
        given    = pair_df_t5[pair_df_t5["scorer"] == scorer]["z_given"].mean()
        received = pair_df_t5[pair_df_t5["scorer"] == scorer]["z_received"].mean()
        n_songs  = D["df"][D["df"]["Curator"] == scorer].shape[0]
        ss_rows.append(dict(scorer=scorer, avg_given=given, avg_received=received, n_songs=n_songs))
    ss_df = pd.DataFrame(ss_rows).dropna()
    r7, p7 = stats.pearsonr(ss_df["avg_received"], ss_df["avg_given"])

    widget_header(
        "Do people who receive high scores tend to give high scores?",
        f"📊 There's a positive trend, but it's noisy with only {len(ss_df)} people.&nbsp;&nbsp;(r = {r7:.3f})",
    )
    m7, b7 = np.polyfit(ss_df["avg_received"], ss_df["avg_given"], 1)
    x7     = np.linspace(ss_df["avg_received"].min(), ss_df["avg_received"].max(), 200)

    fig_gvr = go.Figure()
    fig_gvr.add_trace(go.Scatter(
        x=x7, y=m7 * x7 + b7, mode="lines", name="Trend",
        line=dict(color="crimson", width=2, dash="dash"), hoverinfo="skip",
    ))
    fig_gvr.add_trace(go.Scatter(
        x=ss_df["avg_received"], y=ss_df["avg_given"],
        mode="markers+text",
        text=ss_df["scorer"],
        textposition="top right",
        marker=dict(size=13, color="#7b2d8b", line=dict(color="white", width=1.2)),
        customdata=ss_df[["scorer", "avg_received", "avg_given", "n_songs"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Avg z received: %{customdata[1]:.3f}<br>"
            "Avg z given:    %{customdata[2]:.3f}<br>"
            "Songs curated:  %{customdata[3]}<extra></extra>"
        ),
        name="Participants",
    ))
    fig_gvr.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig_gvr.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig_gvr.update_layout(
        xaxis_title="Avg z-score received from others",
        yaxis_title="Avg z-score given to others",
        height=540, margin=dict(t=20), showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_gvr, use_container_width=True)
    st.markdown(
        '<div class="section-note">Aggregated to the person level — each dot is one participant. '
        'X-axis = average z-score they received on their curated songs; '
        'Y-axis = average z-score they gave to others. '
        'Upper-right = both received high and gave high; could reflect taste alignment, '
        'genuine affinity, or just curation quality.</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Person-level correlation: r = {r7:.3f}, p = {p7:.3f}.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 (w1) — Overall Scores
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    widget_header(
        "How do the scores people give compare to the scores they receive?",
        "📦 Most people score in a similar range — but there are interesting outliers in both directions.",
    )
    _box_colors = {"Assigned": "#5b9bd5", "Received": "#ed7d31"}
    fig_box = go.Figure()
    for _btype in ["Assigned", "Received"]:
        _x, _q1, _med, _q3, _lo, _hi = [], [], [], [], [], []
        for _bp in D["box_order"]:
            _bvals = D["box_df"][
                (D["box_df"]["Participant"] == _bp) & (D["box_df"]["Type"] == _btype)
            ]["Score"].dropna().values
            if len(_bvals) == 0:
                continue
            _x.append(_bp)
            _q1.append(float(np.percentile(_bvals, 25)))
            _med.append(float(np.median(_bvals)))
            _q3.append(float(np.percentile(_bvals, 75)))
            _lo.append(float(_bvals.min()))
            _hi.append(float(_bvals.max()))
        fig_box.add_trace(go.Box(
            x=_x, q1=_q1, median=_med, q3=_q3,
            lowerfence=_lo, upperfence=_hi,
            name=_btype,
            marker_color=_box_colors[_btype],
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Median: %{median:.1f}<br>"
                "Min: %{lowerfence:.1f}<br>"
                "Max: %{upperfence:.1f}"
                "<extra></extra>"
            ),
        ))
    fig_box.update_layout(
        yaxis_range=[0, 10.5],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40),
        plot_bgcolor="white", paper_bgcolor="white",
        boxmode="group",
        hovermode="x unified",
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown(
        '<div class="section-note">Score distributions (0–10 scale). '
        '"Assigned" = spread of all scores this person gave to others. '
        '"Received" = spread of all scores their own curated songs got back. '
        'Participants who never curated only have an Assigned box.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Summary Table")
    df_show = D["overall_df"].copy()
    df_show.columns = ["Participant", "Also Curated", "Avg Score Assigned", "Avg Score Received"]
    df_show["Also Curated"] = df_show["Also Curated"].map({True: "✓", False: ""})
    df_show = df_show.sort_values("Avg Score Assigned", ascending=False).reset_index(drop=True)
    df_show.index = df_show.index + 1
    st.dataframe(df_show, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 (w2) — Cross-Score Matrix
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    widget_header(
        "How did each person score each other person's music?",
        "🔥 Some clear patterns — a few curators were consistently scored above or below average.",
    )
    mode = st.radio("Score type", ["Z-score (normalized)", "Raw (0–10)"],
                    horizontal=True, key="heatmap_mode")

    mat    = (D["cross_z"] if "Z-score" in mode else D["cross_raw"]).astype(float)
    MUTUAL = D["MUTUAL"]

    hover_text = []
    for curator in MUTUAL:
        row_hover = []
        for scorer in MUTUAL:
            if curator == scorer:
                row_hover.append("(same person)")
            else:
                v = mat.loc[curator, scorer]
                if pd.isna(v):
                    row_hover.append("No data")
                else:
                    label = "z-score" if "Z-score" in mode else "raw avg"
                    row_hover.append(
                        f"<b>{scorer}</b> scored <b>{curator}'s</b> songs<br>{label}: {v:.2f}"
                    )
        hover_text.append(row_hover)

    cmin, cmax = (-1.5, 1.5) if "Z-score" in mode else (3, 10)
    fig_heat = go.Figure(go.Heatmap(
        z=mat.values,
        x=list(mat.columns),
        y=list(mat.index),
        text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in mat.values],
        texttemplate="%{text}",
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale="RdYlGn",
        zmid=0 if "Z-score" in mode else 6.5,
        zmin=cmin, zmax=cmax,
        colorbar=dict(title="z-score" if "Z-score" in mode else "raw avg"),
    ))
    fig_heat.update_layout(
        title="Cross-Score Matrix (rows = curator, columns = scorer)",
        xaxis_title="Scorer", yaxis_title="Curator",
        height=520, margin=dict(t=60),
        xaxis=dict(tickangle=-40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown(
        '<div class="section-note">Each cell = average score the <b>column scorer</b> gave to the '
        '<b>row curator\'s</b> songs. Green = above average; red = below average. '
        'Toggle between z-score (accounts for individual scoring style) and raw 0–10.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 (w8) — Permutation Tests
# ─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    widget_header(
        "How confident are we that these results aren't just random noise?",
        "🎲 Not very — neither effect is statistically significant, though the sample is small.",
    )
    col_a, col_b = st.columns(2)

    def perm_fig(perm_rs, obs_r, perm_p, title, color):
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=perm_rs, nbinsx=80,
            marker_color=color, opacity=0.7,
            name="Permuted r values",
            hovertemplate="r = %{x:.3f}<br>count = %{y}<extra></extra>",
        ))
        fig.add_vline(
            x=obs_r, line_width=2.5, line_color="crimson",
            annotation_text=f"Observed r = {obs_r:.3f}",
            annotation_position="top right",
            annotation_font_size=12,
        )
        fig.add_vline(x=-obs_r, line_width=1.5, line_color="crimson", line_dash="dot")
        fig.update_layout(
            title=title,
            xaxis_title="Pearson r (permuted)",
            yaxis_title="Count",
            height=420, margin=dict(t=60),
            showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
        )
        return fig

    with col_a:
        st.plotly_chart(
            perm_fig(D["perm_t"], D["obs_r_t"], D["perm_p_t"],
                     f"Targeted Reciprocity — perm-p = {D['perm_p_t']:.3f}", "#5b9bd5"),
            use_container_width=True,
        )
        st.caption(
            f"Observed r = {D['obs_r_t']:.3f}, perm-p = {D['perm_p_t']:.3f}. "
            "The observed r sits comfortably within the range of chance variation."
        )

    with col_b:
        st.plotly_chart(
            perm_fig(D["perm_g"], D["obs_r_g"], D["perm_p_g"],
                     f"General Mood Effect — perm-p = {D['perm_p_g']:.3f}", "#ed7d31"),
            use_container_width=True,
        )
        st.caption(
            f"Observed r = {D['obs_r_g']:.3f}, perm-p = {D['perm_p_g']:.3f}. "
            "Essentially no relationship between scores received and scores given."
        )

    st.markdown(
        '<div class="section-note">A permutation test shuffles the x-values 10,000 times and '
        'recomputes r each time, building a null distribution of "what r looks like by chance." '
        'The vertical line is our actual observed r. '
        'If it sat far in the tail, we\'d have evidence of a real effect.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("""
    **Interpreting permutation p-values:** A p-value of 0.05 means only 5% of random shuffles
    produced a correlation as large as ours — i.e., our result is in the most extreme 5% of the
    null distribution. Neither test clears that bar. With only 13 mutual participants and 48 songs,
    the dataset is too small to reliably detect subtle effects either way.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Z-score explainer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📐 What is a z-score, and why does it matter here?"):
    st.markdown("""
    A **z-score** re-expresses a number as *how many standard deviations above or below
    that person's own average* it sits. A z-score of +1 means "one standard deviation
    above this person's typical rating"; −1 means one below.

    **Why it matters here:** some people are generous scorers (rarely go below 7) and some
    are strict (5 is high praise). If we compare raw scores directly, a 7 from a strict
    scorer and a 7 from a generous scorer mean very different things. Z-scores put everyone
    on the same footing, so we're measuring *relative enthusiasm* rather than absolute numbers.

    Plots that show z-scores are measuring **deviation from each person's own baseline**.
    Raw scores are preserved for the tables and places where the absolute number is intuitive.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Music Nerds Scholarship · Scoring Affinity Analysis · "
    f"{D['df']['Deadline_dt'].dt.strftime('%b %Y').iloc[0]}–"
    f"{D['df']['Deadline_dt'].dt.strftime('%b %Y').iloc[-1]}"
)
