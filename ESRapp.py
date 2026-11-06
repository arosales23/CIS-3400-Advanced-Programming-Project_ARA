import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Effort, Sleep, and Risk", layout="wide")
sns.set_theme(style="whitegrid", context="notebook")

RAW_FILE = "student_exam_performance.csv"
DASHBOARD_FILE = "dashboard_data.csv"

USEFUL_COLUMNS = [
    "student_id", "exam_score", "pass_status", "study_hours_per_day", "sleep_hours",
    "attendance_percentage", "stress_level", "education_level", "family_income",
    "exam_difficulty", "previous_exam_score", "exam_preparation_days",
    "assignment_completion_rate", "performance_grade",
]
QUARTILE_LABELS = ["Lowest 25%", "Low-Mid", "High-Mid", "Highest 25%"]
BLUE, RED = "#2a6f97", "#c0392b"
FACTOR_LABELS = {
    "sleep_group": "sleep duration", "effort_group": "daily study time",
    "stress_group": "stress level", "attendance_group": "attendance",
    "prev_score_group": "previous exam score", "completion_group": "assignment completion",
    "prep_group": "days of exam preparation", "exam_difficulty": "exam difficulty",
    "family_income": "family income", "education_level": "education level",
}


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def read_csv(source):
    return pd.read_csv(source)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Select columns and add every derived column the notebook creates."""
    d = raw[USEFUL_COLUMNS].copy()

    d["effort_group"] = pd.qcut(
        d["study_hours_per_day"], 4,
        labels=["Low", "Medium-Low", "Medium-High", "High"])
    d["sleep_group"] = pd.cut(
        d["sleep_hours"], bins=[0, 6, 8, 24],
        labels=["Short (<6h)", "Normal (6-8h)", "Long (>8h)"], right=False)
    d["failed"] = (d["pass_status"] == "Fail").astype(int)
    d["stress_group"] = pd.cut(
        d["stress_level"], bins=[0, 7, 9, 10],
        labels=["Lower (<=7)", "High (8-9)", "Max (10)"])
    d["attendance_group"] = pd.qcut(d["attendance_percentage"], 4, labels=QUARTILE_LABELS)
    d["prev_score_group"] = pd.qcut(d["previous_exam_score"], 4, labels=QUARTILE_LABELS)
    d["completion_group"] = pd.qcut(d["assignment_completion_rate"], 4, labels=QUARTILE_LABELS)
    d["prep_group"] = pd.cut(
        d["exam_preparation_days"], bins=[0, 10, 20, 30],
        labels=["1-10 days", "11-20 days", "21-30 days"])
    return d


def load_data():
    """Use a local CSV if present, otherwise ask for an upload."""
    for path in (RAW_FILE, DASHBOARD_FILE):
        if os.path.exists(path):
            return read_csv(path), path
    up = st.sidebar.file_uploader("Upload student_exam_performance.csv", type="csv")
    if up is not None:
        return read_csv(up), up.name
    return None, None


raw, source_name = load_data()

st.title("Effort, Sleep, and Risk")
st.caption("Analysis of the Student Exam Performance & Success dataset (synthetic data)")

if raw is None:
    st.info("Place `student_exam_performance.csv` next to `app.py`, or upload it in the sidebar.")
    st.stop()

missing_cols = [c for c in USEFUL_COLUMNS if c not in raw.columns]
if missing_cols:
    st.error(f"The file is missing required columns: {missing_cols}")
    st.stop()

data = prepare(raw)

# ------------------------------------------------------------- sidebar
st.sidebar.success(f"Loaded {source_name}")
st.sidebar.header("Filters")
edu = st.sidebar.multiselect(
    "Education level", sorted(data["education_level"].dropna().unique()),
    default=sorted(data["education_level"].dropna().unique()))
diff = st.sidebar.multiselect(
    "Exam difficulty", ["Easy", "Medium", "Hard"], default=["Easy", "Medium", "Hard"])
incomes = list(data["family_income"].dropna().unique())
inc = st.sidebar.multiselect("Family income", incomes, default=incomes)

# Thresholds come from the full dataset so flags stay comparable when filtering.
hi_effort = data["study_hours_per_day"].quantile(0.75)
lo_score = data["exam_score"].quantile(0.25)
hi_completion = data["assignment_completion_rate"].quantile(0.75)
data["overworked_no_payoff"] = (data["study_hours_per_day"] >= hi_effort) & (data["exam_score"] <= lo_score)
data["overworked_strict"] = data["overworked_no_payoff"] & (data["assignment_completion_rate"] >= hi_completion)

df = data[
    data["education_level"].isin(edu)
    & data["exam_difficulty"].isin(diff)
    & data["family_income"].isin(inc)
].copy()

if df.empty:
    st.warning("No students match the current filters.")
    st.stop()

overall_fail = df["failed"].mean() * 100

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Q1: Overworking", "Q2: Risk profiles", "Q3: Sleep vs. study", "Data"])

# ------------------------------------------------------------ overview
with tab0:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(df):,}")
    c2.metric("Fail rate", f"{overall_fail:.1f}%")
    c3.metric("Median exam score", f"{df['exam_score'].median():.1f}")
    c4.metric("Overworked, no payoff", f"{df['overworked_no_payoff'].mean() * 100:.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col, title in [
        (axes[0, 0], "study_hours_per_day", "Study hours per day"),
        (axes[0, 1], "sleep_hours", "Sleep hours per night"),
        (axes[1, 0], "exam_score", "Exam score"),
        (axes[1, 1], "assignment_completion_rate", "Assignment completion rate"),
    ]:
        sns.histplot(df[col].dropna(), bins=40, ax=ax, color=BLUE)
        ax.axvline(df[col].median(), color="red", ls="--", label="median")
        ax.set(title=f"Distribution of {title.lower()}", xlabel=title)
        ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        f"**Finding:** The typical student in this selection studies "
        f"{df['study_hours_per_day'].median():.1f} hours a day, sleeps "
        f"{df['sleep_hours'].median():.1f} hours a night, and scores "
        f"{df['exam_score'].median():.1f} on the exam. {overall_fail:.1f}% of students fail. "
        f"Dashed red lines mark each median.")

# ------------------------------------------------------------------ Q1
with tab1:
    st.header("Which students are overworking without payoff?")
    st.markdown(
        f"Flagged = study hours in the top quartile (≥ {hi_effort:.2f} h/day) **and** "
        f"exam score in the bottom quartile (≤ {lo_score:.1f}).")

    a, b = st.columns(2)
    a.metric("Original flag", f"{df['overworked_no_payoff'].sum():,}",
             f"{df['overworked_no_payoff'].mean() * 100:.2f}% of students", delta_color="off")
    b.metric("Stricter flag (+ high assignment completion)", f"{df['overworked_strict'].sum():,}",
             f"{df['overworked_strict'].mean() * 100:.2f}% of students", delta_color="off")

    rho_all = df["study_hours_per_day"].corr(df["exam_score"], method="spearman")
    rho_by_diff = [
        df.loc[df["exam_difficulty"] == d_, "study_hours_per_day"]
        .corr(df.loc[df["exam_difficulty"] == d_, "exam_score"], method="spearman")
        for d_ in ["Easy", "Medium", "Hard"] if (df["exam_difficulty"] == d_).sum() > 2]
    link = "more study time generally goes with higher scores" if rho_all >= 0.2 else \
        "study time has only a weak link to scores"
    st.info(
        f"**Finding:** Only {df['overworked_no_payoff'].mean() * 100:.1f}% of students "
        f"({df['overworked_no_payoff'].sum():,}) study a lot but score poorly, so overworking "
        f"without payoff is the exception. Across all students, {link} "
        f"(Spearman r = {rho_all:.2f})"
        + (f", and the link is similar across exam difficulty levels "
           f"(r from {min(rho_by_diff):.2f} to {max(rho_by_diff):.2f}). "
           if len(rho_by_diff) > 1 else ". ") +
        f"Advisors can use the flagged group's profile table below to see what else sets "
        f"these students apart, such as sleep, stress, or attendance.")

    sample = df.sample(min(8000, len(df)), random_state=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=sample, x="study_hours_per_day", y="exam_score",
                    hue="overworked_no_payoff", palette={False: "#9bb7d4", True: "red"},
                    alpha=0.4, s=12, ax=ax)
    ax.axvline(hi_effort, ls="--", color="gray")
    ax.axhline(lo_score, ls="--", color="gray")
    ax.set_title("Effort vs. score: students working hard without payoff")
    st.pyplot(fig)

    st.subheader("Flagged vs. everyone else")
    profile = (df.groupby("overworked_no_payoff")
               .agg(students=("student_id", "count"),
                    avg_study_hours=("study_hours_per_day", "mean"),
                    avg_exam_score=("exam_score", "mean"),
                    avg_assignment_completion=("assignment_completion_rate", "mean"),
                    avg_sleep_hours=("sleep_hours", "mean"),
                    avg_stress=("stress_level", "mean"),
                    avg_attendance=("attendance_percentage", "mean"),
                    avg_prev_score=("previous_exam_score", "mean"),
                    pct_hard_exam=("exam_difficulty", lambda s: (s == "Hard").mean() * 100))
               .round(2).T)
    profile.columns = ["Everyone else" if not c else "Overworked, no payoff" for c in profile.columns]
    st.dataframe(profile, use_container_width=True)

    st.subheader("Median exam score: effort × exam difficulty")
    q1 = (df.pivot_table(index="effort_group", columns="exam_difficulty",
                         values="exam_score", aggfunc="median", observed=True)
          .reindex(columns=[c for c in ["Easy", "Medium", "Hard"] if c in diff]).round(1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(q1, annot=True, fmt=".1f", cmap="YlGnBu",
                cbar_kws={"label": "Median exam score"}, ax=ax)
    ax.set(xlabel="Exam difficulty", ylabel="Effort group")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Spearman correlation: study hours vs. exam score")
    rows = [("All exams", df["study_hours_per_day"].corr(df["exam_score"], method="spearman"))]
    for d_ in ["Easy", "Medium", "Hard"]:
        sub = df[df["exam_difficulty"] == d_]
        if len(sub) > 2:
            rows.append((d_, sub["study_hours_per_day"].corr(sub["exam_score"], method="spearman")))
    st.dataframe(pd.DataFrame(rows, columns=["Exam difficulty", "Spearman"]).round(3),
                 hide_index=True)

    st.subheader("% flagged by group")
    col = st.selectbox("Group by", ["sleep_group", "education_level", "family_income", "exam_difficulty"])
    flagged = (df.groupby(col, observed=True)["overworked_no_payoff"].mean() * 100).round(1)
    st.dataframe(flagged.to_frame("% flagged"), use_container_width=True)

# ------------------------------------------------------------------ Q2
with tab2:
    st.header("Which student profile is most at risk of failing?")
    st.markdown(f"Overall fail rate: **{overall_fail:.1f}%** "
                f"({df['failed'].sum():,} of {len(df):,} students)")

    factors = ["sleep_group", "effort_group", "stress_group", "attendance_group",
               "prev_score_group", "completion_group", "prep_group",
               "exam_difficulty", "family_income", "education_level"]

    st.subheader("Which factors separate pass from fail the most?")
    rows = []
    for f in factors:
        g = df.groupby(f, observed=True)["failed"].mean() * 100
        if len(g) < 2:  # a factor with a single group (e.g. one difficulty selected) has no gap
            continue
        rows.append({"factor": f, "lowest_fail_rate": g.min(), "highest_fail_rate": g.max(),
                     "gap_(pts)": g.max() - g.min(), "riskiest_group": str(g.idxmax())})
    gap_tbl = (pd.DataFrame(rows).sort_values("gap_(pts)", ascending=False)
               .round(1).reset_index(drop=True))
    st.dataframe(gap_tbl, use_container_width=True)

    g1, g2, g3, gl = gap_tbl.iloc[0], gap_tbl.iloc[1], gap_tbl.iloc[2], gap_tbl.iloc[-1]
    st.info(
        f"**Finding:** The factor that separates pass from fail the most is "
        f"{FACTOR_LABELS[g1['factor']]}: the riskiest group ({g1['riskiest_group']}) fails "
        f"{g1['highest_fail_rate']:.1f}% of the time vs {g1['lowest_fail_rate']:.1f}% for the "
        f"safest group. Next are {FACTOR_LABELS[g2['factor']]} ({g2['gap_(pts)']:.0f}-point gap) "
        f"and {FACTOR_LABELS[g3['factor']]} ({g3['gap_(pts)']:.0f}-point gap). "
        f"{FACTOR_LABELS[gl['factor']].capitalize()} matters least ({gl['gap_(pts)']:.1f} points). "
        f"These are associations, not proven causes, and the data is synthetic.")

    with st.expander("Fail rate by each factor", expanded=False):
        fig, axes = plt.subplots(5, 2, figsize=(13, 20))
        for ax, f in zip(axes.ravel(), factors):
            rate = df.groupby(f, observed=True)["failed"].mean() * 100
            sns.barplot(x=rate.index.astype(str), y=rate.values, color=BLUE, ax=ax)
            ax.axhline(overall_fail, ls="--", color="red")
            for i, v in enumerate(rate.values):
                ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
            ax.set(title=f"Fail rate by {f}", xlabel="", ylabel="% failing")
            ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Fail rate for pairs of factors")
    pairs = [("sleep_group", "effort_group"), ("prev_score_group", "attendance_group"),
             ("exam_difficulty", "prev_score_group"), ("sleep_group", "stress_group")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (r, c) in zip(axes.ravel(), pairs):
        tbl = (df.pivot_table(index=r, columns=c, values="failed",
                              aggfunc="mean", observed=True) * 100).round(1)
        sns.heatmap(tbl, annot=True, fmt=".1f", cmap="Reds",
                    cbar_kws={"label": "% failing"}, ax=ax)
        ax.set(title=f"Fail rate: {r} x {c}", xlabel="", ylabel="")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Top risky profiles")
    profile_factors = st.multiselect(
        "Factors that define a profile", factors,
        default=["sleep_group", "effort_group", "prev_score_group", "exam_difficulty"])
    min_students = st.slider("Minimum students per profile", 10, 1000, 200, step=10)
    top_n = st.slider("Profiles to show", 3, 20, 10)

    if not profile_factors:
        st.info("Pick at least one factor.")
    else:
        profiles = (df.groupby(profile_factors, observed=True)
                    .agg(students=("student_id", "count"),
                         fail_rate=("failed", lambda s: s.mean() * 100))
                    .reset_index())
        profiles = profiles[profiles["students"] >= min_students].copy()
        if profiles.empty:
            st.warning("No profiles meet the minimum size. Lower the minimum or use fewer factors.")
        else:
            profiles["lift"] = profiles["fail_rate"] / overall_fail
            profiles = profiles.sort_values("fail_rate", ascending=False).round(2)
            st.caption(f"{len(profiles)} profiles with at least {min_students} students.")

            top = profiles.head(top_n).copy()
            top["label"] = top[profile_factors].astype(str).agg(" | ".join, axis=1)
            fig, ax = plt.subplots(figsize=(11, max(3, 0.55 * len(top) + 1.5)))
            sns.barplot(data=top, y="label", x="fail_rate", color=RED, ax=ax)
            ax.axvline(overall_fail, ls="--", color="black", label=f"Overall ({overall_fail:.1f}%)")
            for i, (v, n) in enumerate(zip(top["fail_rate"], top["students"])):
                ax.text(v + 0.3, i, f"{v:.1f}%  (n={n:,})", va="center", fontsize=9)
            ax.set(title=f"Top {len(top)} student profiles by fail rate",
                   xlabel="% failing", ylabel="")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            worst = top.iloc[0]
            st.info(
                f"**Finding:** The highest-risk profile is **{worst['label']}**: "
                f"{worst['fail_rate']:.1f}% fail (n={int(worst['students']):,}), "
                f"about {worst['lift']:.1f}x the overall rate of {overall_fail:.1f}%."
                + (" That is higher than any single factor above, so counselors should look "
                   "at combinations of factors, not one factor alone."
                   if worst["fail_rate"] > g1["highest_fail_rate"] else ""))

            l, r = st.columns(2)
            l.markdown("**Most at-risk**")
            l.dataframe(profiles.head(10).reset_index(drop=True))
            r.markdown("**Least at-risk**")
            r.dataframe(profiles.tail(5).reset_index(drop=True))

# ------------------------------------------------------------------ Q3
with tab3:
    st.header("Does sleep matter as much as study time?")

    c1, c2 = st.columns(2)
    for box, col, label in [(c1, "study_hours_per_day", "Study hours"),
                            (c2, "sleep_hours", "Sleep hours")]:
        r = df[col].corr(df["exam_score"], method="spearman")
        box.metric(f"{label} vs. exam score (Spearman)", f"{r:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    mid = df["exam_score"].median()
    for ax, col, title in [(axes[0], "effort_group", "Study time (hours per day)"),
                           (axes[1], "sleep_group", "Sleep (hours per night)")]:
        med = df.groupby(col, observed=True)["exam_score"].median()
        sns.barplot(x=med.index.astype(str), y=med.values, color=BLUE, ax=ax)
        for i, v in enumerate(med.values):
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center")
        ax.set(title=title, xlabel="", ylabel="Median exam score")
        ax.set_ylim(mid - 15, mid + 15)
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Median exam score: study time vs. sleep", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)

    rho_study = df["study_hours_per_day"].corr(df["exam_score"], method="spearman")
    rho_sleep = df["sleep_hours"].corr(df["exam_score"], method="spearman")
    verdict = ("Sleep matters, but not as much as study time"
               if abs(rho_study) > abs(rho_sleep) else "Sleep matters at least as much as study time")
    sleep_fail = df.groupby("sleep_group", observed=True)["failed"].mean() * 100
    sleep_note = ""
    if {"Short (<6h)", "Long (>8h)"} <= set(sleep_fail.index):
        sleep_note = (f" Students sleeping under 6 hours fail {sleep_fail['Short (<6h)']:.1f}% "
                      f"of the time vs {sleep_fail['Long (>8h)']:.1f}% for those sleeping over 8 hours.")
    st.info(
        f"**Finding:** {verdict}. Study time has a Spearman correlation of {rho_study:.2f} "
        f"with exam score, compared with {rho_sleep:.2f} for sleep.{sleep_note} "
        f"Both study habits and sleep are worth discussing with students.")

    for g in ["effort_group", "sleep_group"]:
        t = (df.groupby(g, observed=True)
             .agg(median_score=("exam_score", "median"),
                  fail_rate=("failed", lambda s: s.mean() * 100)).round(1))
        st.markdown(f"**{g}**")
        st.dataframe(t)

# ---------------------------------------------------------------- data
with tab4:
    st.write(f"{len(df):,} rows after filters")
    st.dataframe(df.head(500), use_container_width=True)
    st.download_button("Download filtered data (CSV)", df.to_csv(index=False).encode(),
                       "dashboard_data.csv", "text/csv")
