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
st.markdown("""
<style>
  .section-note {
    font-size: 0.92rem;
    color: #555;
    background: #f8f8f8;
    border-left: 3px solid #ccc;
    padding: 8px 12px;
    margin-bottom: 12px;
    border-radius: 0 4px 4px 0;
  }
  h2 { margin-top: 1.6rem !important; }
  .stTabs [data-baseweb="tab"] { font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loading & processing ─────────────────────────────────────────────────
@st.cache_data
def load_and_process(path="Music Nerds - Scores.tsv"):
    df_raw = pd.read_csv(path, sep="\t")
    df = df_raw[df_raw["Curator"].notna() & df_raw["Deadline"].notna()].copy()
    df["Deadline_dt"] = pd.to_datetime(df["Deadline"] + "/2025", format="%m/%d/%Y")
    df = df.sort_values("Deadline_dt").reset_index(drop=True)

    SCORER_COLS = [
        "Nathan", "Rachel", "Emily", "Jonathan", "Tera", "Erik", "Nana",
        "Goose", "Kristin", "Jacob", "Andreina", "George", "Jane", "Tyler",
        "Molly", "Megan", "Anna W",
    ]

    # Participants who both curate and score
    MUTUAL = sorted(set(df["Curator"].unique()) & set(SCORER_COLS))

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
                vz  = c_songs[f"z_{scorer}"].dropna()
                vr  = c_songs[scorer].dropna()
                cross_z.loc[curator, scorer]   = vz.mean()  if len(vz)  else np.nan
                cross_raw.loc[curator, scorer] = vr.mean()  if len(vr)  else np.nan

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
            ))
    events_df = pd.DataFrame(events)
    has_history = events_df[events_df["n_prior"] > 0].dropna(subset=["cum_received_avg"])

    # ── Permutation tests ─────────────────────────────────────────────────────
    np.random.seed(42)
    N_PERM = 10_000

    x_t, y_t = pair_df["z_received"].values, pair_df["z_given"].values
    obs_r_t   = np.corrcoef(x_t, y_t)[0, 1]
    perm_t    = [np.corrcoef(np.random.permutation(x_t), y_t)[0, 1] for _ in range(N_PERM)]
    perm_p_t  = np.mean(np.abs(perm_t) >= np.abs(obs_r_t))

    x_g, y_g = has_history["cum_received_avg"].values, has_history["z_given"].values
    obs_r_g   = np.corrcoef(x_g, y_g)[0, 1]
    perm_g    = [np.corrcoef(np.random.permutation(x_g), y_g)[0, 1] for _ in range(N_PERM)]
    perm_p_g  = np.mean(np.abs(perm_g) >= np.abs(obs_r_g))

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

    # ── Overall averages table ────────────────────────────────────────────────
    rows = []
    for scorer in SCORER_COLS:
        avg_assigned_raw = df[scorer].mean()
        if scorer in MUTUAL:
            c_songs    = df_z[df_z["Curator"] == scorer]
            others     = [s for s in SCORER_COLS if s != scorer]
            avg_recv   = np.nanmean(c_songs[others].values)
        else:
            avg_recv   = np.nan
        rows.append(dict(
            Participant=scorer,
            is_mutual=scorer in MUTUAL,
            avg_assigned_raw=round(avg_assigned_raw, 2),
            avg_received_raw=round(avg_recv, 2) if not np.isnan(avg_recv) else None,
        ))
    overall_df = pd.DataFrame(rows)

    # ── Reciprocity table (unordered pairs) ───────────────────────────────────
    seen = set()
    recip_rows = []
    for _, row in pair_df.iterrows():
        key = tuple(sorted([row["scorer"], row["curator"]]))
        if key in seen:
            continue
        seen.add(key)
        a, b = row["scorer"], row["curator"]
        raw_a2b = cross_raw.loc[b, a]
        raw_b2a = cross_raw.loc[a, b]
        z_a2b   = cross_z.loc[b, a]
        z_b2a   = cross_z.loc[a, b]
        ratio   = raw_b2a / raw_a2b if (pd.notna(raw_a2b) and pd.notna(raw_b2a)
                                         and raw_a2b != 0) else np.nan
        recip_rows.append(dict(
            person_a=a, person_b=b,
            raw_a_to_b=round(raw_a2b, 2) if pd.notna(raw_a2b) else None,
            raw_b_to_a=round(raw_b2a, 2) if pd.notna(raw_b2a) else None,
            ratio_b_over_a=round(ratio, 3) if pd.notna(ratio) else None,
            z_a_to_b=round(z_a2b, 3) if pd.notna(z_a2b) else None,
            z_b_to_a=round(z_b2a, 3) if pd.notna(z_b2a) else None,
        ))
    recip_df = (pd.DataFrame(recip_rows)
                .sort_values("ratio_b_over_a", ascending=False)
                .reset_index(drop=True))

    return dict(
        df=df, df_z=df_z,
        SCORER_COLS=SCORER_COLS, MUTUAL=MUTUAL,
        cross_z=cross_z, cross_raw=cross_raw,
        pair_df=pair_df, events_df=events_df, has_history=has_history,
        obs_r_t=obs_r_t, perm_p_t=perm_p_t, perm_t=perm_t,
        obs_r_g=obs_r_g, perm_p_g=perm_p_g, perm_g=perm_g,
        re_df=re_df, weekly=weekly, scorer_overall_z=scorer_overall_z,
        overall_df=overall_df, recip_df=recip_df,
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
Each week, one person curates a short playlist and everyone else rates each song out of 10.
Because the same people both submit music **and** score others, an interesting question emerges:
**do the scores you receive from others influence the scores you give back?**

This dashboard explores two versions of that idea:

- **Targeted reciprocity** — does person A score person B's music higher (or lower) specifically
  because of how B has scored A's own picks?
- **General mood effect** — after a week where your music received low scores, do you
  tend to score everyone more harshly the next time you rate?

A third, broader framing is **scoring affinity**: do friends or people with aligned taste
systematically reward each other, regardless of any strategic motive?
""")

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

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Overall Averages",
    "🟩 Cross-Score Matrix",
    "📋 Reciprocity Table",
    "🎯 Targeted Reciprocity",
    "🌡️ General Mood Effect",
    "📈 Trends Over Time",
    "⚖️ Given vs. Received",
    "🔀 Permutation Tests",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Overall Averages
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-note">Raw averages (0–10 scale) for each participant. '
                '"Avg Score Assigned" is their mean rating across all songs they scored. '
                '"Avg Score Received" is the mean rating their curated songs received from others '
                '(blank for participants who never curated).</div>', unsafe_allow_html=True)

    df_show = D["overall_df"].copy()
    df_show.columns = ["Participant", "Also Curated", "Avg Score Assigned", "Avg Score Received"]
    df_show["Also Curated"] = df_show["Also Curated"].map({True: "✓", False: ""})
    df_show = df_show.sort_values("Avg Score Assigned", ascending=False).reset_index(drop=True)
    df_show.index = df_show.index + 1

    # Plotly bar chart: assigned vs received side-by-side
    fig_avg = go.Figure()
    fig_avg.add_trace(go.Bar(
        name="Avg Score Assigned",
        x=df_show["Participant"],
        y=df_show["Avg Score Assigned"],
        marker_color="#5b9bd5",
        hovertemplate="<b>%{x}</b><br>Avg assigned: %{y:.2f}<extra></extra>",
    ))
    fig_avg.add_trace(go.Bar(
        name="Avg Score Received",
        x=df_show["Participant"],
        y=df_show["Avg Score Received"],
        marker_color="#ed7d31",
        hovertemplate="<b>%{x}</b><br>Avg received: %{y:.2f}<extra></extra>",
    ))
    fig_avg.update_layout(
        barmode="group", title="Scores Assigned vs. Received per Participant",
        xaxis_title="Participant", yaxis_title="Score (0–10)",
        yaxis_range=[0, 10], legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420, margin=dict(t=60),
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    st.subheader("Table")
    st.dataframe(df_show, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Cross-Score Matrix
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-note">Each cell shows the average score that the '
                '<b>column scorer</b> gave to the <b>row curator\'s</b> songs. '
                'Green = above average; red = below average. '
                'Toggle between z-score (deviation from scorer\'s own mean) and raw 0–10 scores.</div>',
                unsafe_allow_html=True)

    mode = st.radio("Score type", ["Z-score (normalized)", "Raw (0–10)"],
                    horizontal=True, key="heatmap_mode")

    mat = (D["cross_z"] if "Z-score" in mode else D["cross_raw"]).astype(float)
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
                    row_hover.append(f"Scorer: {scorer}<br>Curator: {curator}<br>{label}: {v:.2f}")
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
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Reciprocity Table
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-note">Each row is one pair of participants who both '
                'curated and scored each other\'s music. The <b>Ratio</b> is '
                'B\'s raw average score toward A divided by A\'s raw average toward B — '
                'a ratio > 1 means B was more generous to A than A was to B; < 1 means the reverse. '
                'Sorted by ratio (most lopsided at top).</div>', unsafe_allow_html=True)

    recip = D["recip_df"].copy()
    recip.columns = [
        "Person A", "Person B",
        "A→B (raw)", "B→A (raw)", "Ratio (B→A / A→B)",
        "A→B (z)", "B→A (z)",
    ]
    recip.index = recip.index + 1

    # Colour-code ratio column
    def color_ratio(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        if val > 1.15:
            return "background-color: #c6efce; color: #276221"  # green
        elif val < 0.87:
            return "background-color: #ffc7ce; color: #9c0006"  # red
        return ""

    st.dataframe(
        recip.style.applymap(color_ratio, subset=["Ratio (B→A / A→B)"]),
        use_container_width=True, height=520,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Targeted Reciprocity Scatter
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-note">Each dot is a directed pair: person A scoring '
                'person B\'s music. The x-axis is how B rated A (the "treatment" A received); '
                'the y-axis is how A rated B in return. A positive slope would indicate '
                'scoring affinity; a negative slope would indicate retaliation.</div>',
                unsafe_allow_html=True)

    pair_df = D["pair_df"].copy()
    pair_df["label"] = pair_df["scorer"] + " → " + pair_df["curator"]

    fig_scat = px.scatter(
        pair_df,
        x="z_received", y="z_given",
        hover_name="label",
        hover_data={"z_received": ":.3f", "z_given": ":.3f",
                    "raw_received": ":.2f", "raw_given": ":.2f",
                    "label": False},
        labels={"z_received": "Score B gave A (z)", "z_given": "Score A gave B (z)",
                "raw_received": "B→A (raw)", "raw_given": "A→B (raw)"},
        color="scorer",
        height=520,
    )

    # Regression line
    m, b = np.polyfit(pair_df["z_received"], pair_df["z_given"], 1)
    x_line = np.linspace(pair_df["z_received"].min(), pair_df["z_received"].max(), 200)
    fig_scat.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b,
        mode="lines", name="Trend",
        line=dict(color="red", width=2, dash="dash"),
        hoverinfo="skip",
    ))
    fig_scat.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_scat.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)

    r, p = D["obs_r_t"], D["perm_p_t"]
    fig_scat.add_annotation(
        x=0.02, y=0.97, xref="paper", yref="paper",
        text=f"r = {r:.3f} | permutation p = {p:.3f}",
        showarrow=False, font=dict(size=13),
        bgcolor="white", bordercolor="gray", borderwidth=1,
    )
    fig_scat.update_layout(title="Targeted Reciprocity: Does How B Treated A Predict How A Treats B?",
                            margin=dict(t=60))
    st.plotly_chart(fig_scat, use_container_width=True)

    st.caption(f"**{len(pair_df)} directed pairs** from {len(D['MUTUAL'])} mutual participants.  "
               f"r = {r:.3f} (slight positive / affinity direction), permutation p = {p:.3f} "
               f"(not statistically significant at conventional thresholds).")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — General Mood Effect
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-note">Each dot is one (song, scorer) observation where '
                'the scorer had already received feedback on their own curated songs before '
                'this deadline. The x-axis is the scorer\'s cumulative average received z-score '
                'up to that point; the y-axis is the z-score they gave. A negative slope '
                'would mean "scorers who have received low scores give lower scores to others."</div>',
                unsafe_allow_html=True)

    hist = D["has_history"].copy()
    hist["song_short"] = hist["song"].str[:40]

    fig_mood = px.scatter(
        hist,
        x="cum_received_avg", y="z_given",
        hover_data={
            "scorer": True, "curator": True, "song_short": True,
            "cum_received_avg": ":.3f", "z_given": ":.3f",
            "song": False,
        },
        labels={
            "cum_received_avg": "Scorer's cumulative avg received (z)",
            "z_given": "Score given (z)",
            "song_short": "Song",
        },
        color="scorer",
        opacity=0.6,
        height=520,
    )

    m3, b3 = np.polyfit(hist["cum_received_avg"], hist["z_given"], 1)
    x3 = np.linspace(hist["cum_received_avg"].min(), hist["cum_received_avg"].max(), 200)
    fig_mood.add_trace(go.Scatter(
        x=x3, y=m3 * x3 + b3, mode="lines", name="Trend",
        line=dict(color="red", width=2, dash="dash"), hoverinfo="skip",
    ))
    fig_mood.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_mood.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)

    r_g, p_g = D["obs_r_g"], D["perm_p_g"]
    fig_mood.add_annotation(
        x=0.02, y=0.97, xref="paper", yref="paper",
        text=f"r = {r_g:.3f} | permutation p = {p_g:.3f}",
        showarrow=False, font=dict(size=13),
        bgcolor="white", bordercolor="gray", borderwidth=1,
    )
    fig_mood.update_layout(title="General Mood Effect: Does Receiving Low Scores Make You Score Others Lower?",
                            margin=dict(t=60))
    st.plotly_chart(fig_mood, use_container_width=True)

    st.caption(f"r = {r_g:.3f}, permutation p = {p_g:.3f}.  "
               f"The slope is essentially flat — no evidence that cumulative received scores "
               f"shift how generously participants score others.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Trends Over Time
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-note">Two views of how scoring behaviour evolves across rounds. '
                'The bar chart shows each participant\'s Spearman correlation between round number '
                'and score given (positive = more generous over time, negative = stricter). '
                'The line chart below shows each scorer\'s weekly average z-score over time.</div>',
                unsafe_allow_html=True)

    re_df = D["re_df"].copy()

    # ── Bar chart ────────────────────────────────────────────────────────────
    re_df["color"] = re_df["rho"].apply(lambda x: "#d62728" if x < 0 else "#2ca02c")
    re_df["sig"] = re_df["p"].apply(lambda x: " ★" if x < 0.05 else "")
    re_df["label"] = re_df.apply(
        lambda r: f"<b>{r['scorer']}</b>{r['sig']}<br>"
                  f"ρ = {r['rho']:.2f}, p = {r['p']:.2f}, n = {r['n']}",
        axis=1,
    )

    fig_bar = go.Figure(go.Bar(
        x=re_df["rho"], y=re_df["scorer"],
        orientation="h",
        marker_color=re_df["color"],
        customdata=re_df[["rho", "p", "n", "scorer"]].values,
        hovertemplate=(
            "<b>%{customdata[3]}</b><br>"
            "ρ = %{customdata[0]:.3f}<br>"
            "p = %{customdata[1]:.3f}<br>"
            "n = %{customdata[2]}<extra></extra>"
        ),
    ))
    fig_bar.add_vline(x=0, line_width=1.5, line_color="black")
    fig_bar.update_layout(
        title="Spearman ρ: Score Given vs. Round Number",
        xaxis_title="Spearman ρ", yaxis_title="",
        height=400, margin=dict(t=60),
        xaxis=dict(range=[-0.5, 0.5]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Line chart ───────────────────────────────────────────────────────────
    st.subheader("Weekly Average Score Given (z-score)")

    MUTUAL = D["MUTUAL"]
    all_scorers_options = MUTUAL + [s for s in D["SCORER_COLS"] if s not in MUTUAL]
    default_sel = MUTUAL[:6]
    selected_scorers = st.multiselect(
        "Select participants to display",
        options=all_scorers_options,
        default=default_sel,
        key="line_select",
    )

    weekly = D["weekly"].copy()
    scorer_z = D["scorer_overall_z"]

    fig_line = go.Figure()
    for scorer in selected_scorers:
        sw = weekly[weekly["scorer"] == scorer].sort_values("deadline")
        if sw.empty:
            continue
        overall_mean = scorer_z.get(scorer, 0)
        fig_line.add_trace(go.Scatter(
            x=sw["deadline"],
            y=sw["weekly_z_mean"],
            mode="lines+markers",
            name=scorer,
            hovertemplate=(
                f"<b>{scorer}</b><br>"
                "Week: %{x|%b %-d}<br>"
                "Avg z this week: %{y:.3f}<extra></extra>"
            ),
        ))
        # Overall mean reference line (dashed, same color, lighter)
        fig_line.add_shape(
            type="line",
            x0=sw["deadline"].min(), x1=sw["deadline"].max(),
            y0=overall_mean, y1=overall_mean,
            line=dict(dash="dot", width=1, color="gray"),
        )

    fig_line.add_hline(y=0, line_dash="dash", line_color="black",
                        opacity=0.3, annotation_text="Group zero-line")
    fig_line.update_layout(
        title="Weekly Average Z-Score Given per Scorer",
        xaxis_title="Deadline", yaxis_title="Avg z-score given",
        height=480, margin=dict(t=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.caption("Dashed grey lines show each scorer's overall mean — "
               "deviations above/below indicate relatively generous or strict weeks.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — Scorer-Level Given vs. Received
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="section-note">Aggregated to the person level: each dot is one '
                'participant. The x-axis is the average z-score they received from others on '
                'their curated songs; the y-axis is the average z-score they gave to others. '
                'People in the upper-right both received high scores and gave high scores — '
                'this could reflect aligned taste or affinity.</div>',
                unsafe_allow_html=True)

    # Build per-person summary
    pair_df = D["pair_df"]
    ss_rows = []
    for scorer in D["MUTUAL"]:
        given    = pair_df[pair_df["scorer"]   == scorer]["z_given"].mean()
        received = pair_df[pair_df["scorer"]   == scorer]["z_received"].mean()
        n_songs  = D["df"][D["df"]["Curator"] == scorer].shape[0]
        ss_rows.append(dict(
            scorer=scorer, avg_given=given, avg_received=received, n_songs=n_songs
        ))
    ss_df = pd.DataFrame(ss_rows).dropna()

    m7, b7 = np.polyfit(ss_df["avg_received"], ss_df["avg_given"], 1)
    x7     = np.linspace(ss_df["avg_received"].min(), ss_df["avg_received"].max(), 200)

    fig_gvr = go.Figure()
    fig_gvr.add_trace(go.Scatter(
        x=x7, y=m7 * x7 + b7, mode="lines", name="Trend",
        line=dict(color="red", width=2, dash="dash"), hoverinfo="skip",
    ))
    fig_gvr.add_trace(go.Scatter(
        x=ss_df["avg_received"], y=ss_df["avg_given"],
        mode="markers+text",
        text=ss_df["scorer"],
        textposition="top right",
        marker=dict(size=12, color="#7b2d8b",
                    line=dict(color="black", width=1)),
        customdata=ss_df[["scorer", "avg_received", "avg_given", "n_songs"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Avg z received: %{customdata[1]:.3f}<br>"
            "Avg z given: %{customdata[2]:.3f}<br>"
            "Songs curated: %{customdata[3]}<extra></extra>"
        ),
        name="Participants",
    ))
    fig_gvr.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig_gvr.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig_gvr.update_layout(
        title="Scorer-Level: Average Z-Score Given vs. Received",
        xaxis_title="Avg z-score received from others",
        yaxis_title="Avg z-score given to others",
        height=520, margin=dict(t=60), showlegend=False,
    )
    st.plotly_chart(fig_gvr, use_container_width=True)

    r7, p7 = stats.pearsonr(ss_df["avg_received"], ss_df["avg_given"])
    st.caption(f"Person-level correlation: r = {r7:.3f}, p = {p7:.3f}.  "
               f"People who receive higher scores do tend to give higher scores — "
               f"but with n={len(ss_df)} this is noisy.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — Permutation Tests
# ─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown('<div class="section-note">A permutation test answers: "how unusual is our '
                'observed correlation if the relationship were purely random?" We shuffle the '
                'x-values 10,000 times and recompute r each time. The red/blue lines show '
                'where our actual observed r falls in that null distribution. '
                'A p-value close to 0 would mean the result is very unlikely by chance.</div>',
                unsafe_allow_html=True)

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
        fig.add_vline(
            x=-obs_r, line_width=1.5, line_color="crimson",
            line_dash="dot",
        )
        fig.update_layout(
            title=title,
            xaxis_title="Pearson r (permuted)",
            yaxis_title="Count",
            height=400,
            margin=dict(t=60),
            showlegend=False,
        )
        return fig

    with col_a:
        st.plotly_chart(
            perm_fig(D["perm_t"], D["obs_r_t"], D["perm_p_t"],
                     f"Targeted Reciprocity<br>perm-p = {D['perm_p_t']:.3f}",
                     "#5b9bd5"),
            use_container_width=True,
        )
        st.caption(f"Observed r = {D['obs_r_t']:.3f}. "
                   f"Permutation p = {D['perm_p_t']:.3f}. "
                   f"The observed r is within the normal range of chance variation.")

    with col_b:
        st.plotly_chart(
            perm_fig(D["perm_g"], D["obs_r_g"], D["perm_p_g"],
                     f"General Mood Effect<br>perm-p = {D['perm_p_g']:.3f}",
                     "#ed7d31"),
            use_container_width=True,
        )
        st.caption(f"Observed r = {D['obs_r_g']:.3f}. "
                   f"Permutation p = {D['perm_p_g']:.3f}. "
                   f"Essentially no relationship between scores received and scores given.")

    st.markdown("---")
    st.markdown("""
    **Interpreting permutation p-values:** A p-value of 0.05 means only 5% of random shuffles
    produced a correlation as large as ours — i.e., ours is in the most extreme 5% of the
    null distribution. Neither test clears that bar, meaning we cannot rule out that the
    patterns we see are simply chance variation in a small dataset.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Music Nerds Scholarship · Scoring Affinity Analysis · "
           f"{D['df']['Deadline_dt'].dt.strftime('%b %Y').iloc[0]}–"
           f"{D['df']['Deadline_dt'].dt.strftime('%b %Y').iloc[-1]}")
