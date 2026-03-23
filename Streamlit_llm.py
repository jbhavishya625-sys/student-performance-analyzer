# Commit 10: Streamlit Dashboard — Full App with LLM Insights integrated
# Run: streamlit run commit10_streamlit_llm.py
# Requires: ANTHROPIC_API_KEY set in environment

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import anthropic

st.set_page_config(page_title="StudentLens — AI Analyzer", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        border-top: 3px solid #e6a817;
    }
    .kpi-val { font-size: 2rem; font-weight: 700; color: #e6a817; margin: 4px 0; }
    .kpi-lbl { font-size: 0.72rem; color: #7d8590; letter-spacing: 0.12em; text-transform: uppercase; }
    .ai-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        line-height: 1.8;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────
@st.cache_data
def load():
    return pd.read_csv("student_performance_cleaned.csv")

df = load()
subjects = ["Math", "Physics", "Chemistry", "English", "Computer_Science"]
grade_colors = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/student-male.png", width=55)
st.sidebar.title("StudentLens")
st.sidebar.caption("AI-Powered Performance Analyzer")
st.sidebar.divider()

grade_filter = st.sidebar.multiselect("Filter by Grade", ["A", "B", "C"], default=["A", "B", "C"])
att_range = st.sidebar.slider("Attendance Range (%)", 55, 100, (55, 100))
study_range = st.sidebar.slider("Study Hours/Day", 0.0, 5.0, (0.0, 5.0))

fdf = df[
    df["Final_Grade"].isin(grade_filter) &
    df["Attendance_Percent"].between(*att_range) &
    df["Study_Hours_Per_Day"].between(*study_range)
]
st.sidebar.divider()
st.sidebar.metric("Students Shown", len(fdf))
st.sidebar.metric("Pass Rate (≥50)", f"{(fdf['Average_Marks'] >= 50).mean()*100:.1f}%")

# ── Header ────────────────────────────────────────────────────
st.title("📊 Student Performance Analyzer")
st.caption("Visualizations + Claude AI Insights · Filtered view updates all charts")
st.divider()

# ── KPIs ──────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    ("Total Students", len(fdf)),
    ("Avg Score", f"{fdf['Average_Marks'].mean():.1f}"),
    ("Avg Attendance", f"{fdf['Attendance_Percent'].mean():.1f}%"),
    ("Avg Study Hrs", f"{fdf['Study_Hours_Per_Day'].mean():.2f}"),
    ("Dominant Grade", fdf["Final_Grade"].value_counts().idxmax() if len(fdf) else "—"),
]
for col, (lbl, val) in zip([c1,c2,c3,c4,c5], kpis):
    col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>',
                 unsafe_allow_html=True)

st.divider()

# ── Charts ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Subject Analysis", "📊 Attendance & Study", "🔥 Correlations"])

with tab1:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Average Marks per Subject")
        fig, ax = plt.subplots(figsize=(8, 4))
        avg = fdf[subjects].mean().sort_values()
        bars = ax.barh(avg.index, avg.values,
                       color=sns.color_palette("Blues_d", len(subjects)), edgecolor="white")
        for bar, val in zip(bars, avg.values):
            ax.text(val + 0.4, bar.get_y() + bar.get_height()/2, f"{val:.1f}", va="center", fontsize=9)
        ax.set_xlim(0, 110); ax.set_xlabel("Average Marks")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Grade Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        gc = fdf["Final_Grade"].value_counts()
        ax.pie(gc.values, labels=gc.index, autopct="%1.1f%%",
               colors=[grade_colors.get(g,"gray") for g in gc.index],
               startangle=90, wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2))
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Score Distribution per Subject (Box Plot)")
    fig, ax = plt.subplots(figsize=(12, 4))
    fdf[subjects].plot(kind="box", ax=ax, patch_artist=True,
                       boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                       medianprops=dict(color="red", linewidth=2))
    ax.set_ylabel("Marks"); ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Study Hours vs Avg Marks")
        fig, ax = plt.subplots(figsize=(7, 4))
        for g, grp in fdf.groupby("Final_Grade"):
            ax.scatter(grp["Study_Hours_Per_Day"], grp["Average_Marks"],
                       label=f"Grade {g}", color=grade_colors[g], alpha=0.75, s=55,
                       edgecolors="white", linewidths=0.4)
        if len(fdf) > 2:
            z = np.polyfit(fdf["Study_Hours_Per_Day"], fdf["Average_Marks"], 1)
            xl = np.linspace(fdf["Study_Hours_Per_Day"].min(), fdf["Study_Hours_Per_Day"].max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), "k--", alpha=0.4, label="Trend")
        ax.set_title(f"r = {fdf['Study_Hours_Per_Day'].corr(fdf['Average_Marks']):.2f}")
        ax.set_xlabel("Study Hours/Day"); ax.set_ylabel("Average Marks"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Attendance vs Avg Marks")
        fig, ax = plt.subplots(figsize=(7, 4))
        for g, grp in fdf.groupby("Final_Grade"):
            ax.scatter(grp["Attendance_Percent"], grp["Average_Marks"],
                       label=f"Grade {g}", color=grade_colors[g], alpha=0.75, s=55,
                       edgecolors="white", linewidths=0.4)
        if len(fdf) > 2:
            z2 = np.polyfit(fdf["Attendance_Percent"], fdf["Average_Marks"], 1)
            xl2 = np.linspace(fdf["Attendance_Percent"].min(), fdf["Attendance_Percent"].max(), 100)
            ax.plot(xl2, np.poly1d(z2)(xl2), "k--", alpha=0.4, label="Trend")
        ax.set_title(f"r = {fdf['Attendance_Percent'].corr(fdf['Average_Marks']):.2f}")
        ax.set_xlabel("Attendance %"); ax.set_ylabel("Average Marks"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Study Hours × Attendance Bubble Chart")
    fig, ax = plt.subplots(figsize=(12, 4))
    sc = ax.scatter(fdf["Study_Hours_Per_Day"], fdf["Attendance_Percent"],
                    c=fdf["Average_Marks"], cmap="RdYlGn", s=90, alpha=0.85,
                    edgecolors="gray", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Avg Marks")
    ax.set_xlabel("Study Hours/Day"); ax.set_ylabel("Attendance %")
    plt.tight_layout(); st.pyplot(fig); plt.close()

with tab3:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    num_cols = subjects + ["Attendance_Percent", "Study_Hours_Per_Day", "Average_Marks"]
    corr = fdf[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, linewidths=0.4, annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Student Table ─────────────────────────────────────────────
st.divider()
st.subheader("📋 Student Records")
search = st.text_input("Search by Student ID", placeholder="e.g. S001")
display = fdf[fdf["Student_ID"].str.contains(search, case=False)] if search else fdf
st.dataframe(display.reset_index(drop=True), use_container_width=True, height=260)

# ── LLM Insights ─────────────────────────────────────────────
st.divider()
st.subheader("✦ AI Insights — Powered by Claude")

PROMPTS = {
    "📋 Class Performance Summary": (
        "You are an expert educational data analyst. "
        "Give a concise 5-point summary of this class performance, highlighting key trends.\n\n{stats}"
    ),
    "💡 Teacher Recommendations": (
        "You are an experienced teacher. "
        "Give 5 specific, data-driven recommendations to improve student outcomes based on:\n\n{stats}"
    ),
    "⚠️ At-Risk Student Profile": (
        "You are an academic advisor. "
        "Describe the at-risk student profile and list 3-4 early warning indicators based on:\n\n{stats}"
    ),
    "📚 Subject Improvement Focus": (
        "You are a curriculum specialist. "
        "Based on subject-wise scores below, identify which subjects need the most attention and suggest strategies.\n\n{stats}"
    ),
}

def build_stats(df):
    subjects = ["Math", "Physics", "Chemistry", "English", "Computer_Science"]
    return f"""
Total Students: {len(df)}
Grade Distribution: {df["Final_Grade"].value_counts().to_dict()}
Subject Averages: { {s: round(df[s].mean(),1) for s in subjects} }
Average Attendance: {df["Attendance_Percent"].mean():.1f}%
Average Study Hours/Day: {df["Study_Hours_Per_Day"].mean():.2f}
Average Score: {df["Average_Marks"].mean():.1f}
Study Hours ↔ Score correlation: {df["Study_Hours_Per_Day"].corr(df["Average_Marks"]):.3f}
Attendance ↔ Score correlation: {df["Attendance_Percent"].corr(df["Average_Marks"]):.3f}
""".strip()

selected = st.selectbox("Choose insight type", list(PROMPTS.keys()))

if st.button("🔍 Generate AI Insight", type="primary"):
    stats_text = build_stats(fdf)
    prompt = PROMPTS[selected].format(stats=stats_text)
    with st.spinner("Claude is analyzing your data..."):
        try:
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}]
            )
            result = msg.content[0].text
            st.markdown(f'<div class="ai-box">{result}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ API Error: {e}\n\nMake sure ANTHROPIC_API_KEY is set.")

st.divider()
st.caption("StudentLens · Data Visualization + Claude AI · Student Performance Analyzer Project")
