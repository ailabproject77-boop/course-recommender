"""
app.py - Streamlit demo.   Run:  streamlit run app.py
"""
from pathlib import Path
import pandas as pd
import streamlit as st

import json
from generate_dataset import save
from recommenders import ContentBased, CollaborativeFiltering, AcademicHybrid, parse_list

st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------- data + models
@st.cache_data
def load_data():
    if not Path("data/courses.csv").exists():
        save("data", seed=42)
    kw = dict(keep_default_na=False)
    return (pd.read_csv("data/courses.csv", **kw), pd.read_csv("data/students.csv", **kw),
            pd.read_csv("data/ratings.csv", **kw))


@st.cache_resource
def load_models():
    courses, _, ratings = load_data()
    cb = ContentBased(courses)
    cf = CollaborativeFiltering(courses, ratings)
    w = (0.35, 0.35, 0.15, 0.15)                         # default; train.py saves tuned weights
    p = Path("results/best_weights.json")
    if p.exists():
        d = json.loads(p.read_text())["weights"]
        w = (d["content"], d["collab"], d["semester"], d["difficulty"])
    return {"Content-Based": cb, "Collaborative Filtering": cf,
            "Academic-Aware Hybrid": AcademicHybrid(cb, cf, w)}


courses, students, ratings = load_data()
models = load_models()
label = {r.course_id: f"{r.course_id} · {r.name} (Sem {r.level})" for r in courses.itertuples()}
all_interests = sorted(courses["category"].unique())
all_skills = sorted({s for x in courses["skills"] for s in parse_list(x)})

# ---------------------------------------------------------------- session defaults
for key, val in dict(interests=[], skills=[], completed=[], semester=5, difficulty=3, sample="—").items():
    st.session_state.setdefault(key, val)


def load_sample():
    """Fill the form from a student in the dataset."""
    sid = st.session_state["sample"]
    if sid == "—":
        return
    s = students[students.student_id == sid].iloc[0]
    hist = ratings[(ratings.student_id == sid) & (ratings.semester_taken < s.semester)]
    st.session_state.update(interests=parse_list(s.interests), skills=parse_list(s.skills),
                            semester=int(s.semester), difficulty=int(s.preferred_difficulty),
                            completed=hist.course_id.tolist())
    for r in hist.itertuples():
        st.session_state[f"rate_{r.course_id}"] = int(r.rating)


# ---------------------------------------------------------------- sidebar (student input)
with st.sidebar:
    st.header("🎓 Student profile")
    st.selectbox("Load a sample student (optional)", ["—"] + students.student_id.tolist(),
                 key="sample", on_change=load_sample)
    st.multiselect("Interests", all_interests, key="interests")
    st.multiselect("Skills", all_skills, key="skills")
    st.multiselect("Completed courses", courses.course_id.tolist(), key="completed",
                   format_func=lambda c: label[c])
    st.slider("Current semester", 1, 8, key="semester")
    st.slider("Preferred difficulty (1 = easy, 5 = hard)", 1, 5, key="difficulty")
    if st.session_state["completed"]:
        with st.expander("Rate completed courses (optional, default 4)"):
            for c in st.session_state["completed"]:
                st.session_state.setdefault(f"rate_{c}", 4)
                st.slider(label[c], 1, 5, key=f"rate_{c}")

student = dict(
    interests=st.session_state["interests"], skills=st.session_state["skills"],
    history={c: st.session_state.get(f"rate_{c}", 4) for c in st.session_state["completed"]},
    semester=st.session_state["semester"], preferred_difficulty=st.session_state["difficulty"])

# ---------------------------------------------------------------- main page
st.title("Personalized Course Recommendation System")
st.caption("AI Lab project · Content-Based vs Collaborative Filtering vs our Academic-Aware Hybrid · "
           "⚠️ Dataset is SYNTHETIC (generated), not real student data.")

tab_rec, tab_res, tab_about = st.tabs(["Recommendations", "Experiment results", "How it works"])


def show(recs):
    st.dataframe(
        recs.assign(prerequisites=recs.missing_prereqs.map(lambda m: f"❌ missing {m}" if m else "✅ met"))
            [["rank", "name", "score", "level", "difficulty", "prerequisites", "reason"]],
        hide_index=True, width="stretch",
        column_config={"score": st.column_config.ProgressColumn("Score (1.0 = top)", min_value=0, max_value=1, format="%.2f"),
                       "name": "Course", "level": "Level", "difficulty": "Diff.",
                       "reason": st.column_config.TextColumn("Why recommended", width="large")})


with tab_rec:
    if not (student["interests"] or student["skills"] or student["history"]):
        st.info("👈 Choose interests, skills or completed courses in the sidebar "
                "(or load a sample student) to get recommendations.")
    else:
        mode = st.radio("Approach", ["Academic-Aware Hybrid", "Content-Based", "Collaborative Filtering",
                                     "Compare all three"], horizontal=True)
        if mode != "Compare all three":
            st.subheader(f"Top 5 – {mode}")
            recs = models[mode].recommend(student, 5)
            show(recs) if len(recs) else st.warning("No eligible courses found.")
        else:
            outs = {m: models[m].recommend(student, 5) for m in models}
            for m, recs in outs.items():
                st.subheader(m)
                show(recs)
            names = pd.DataFrame({m: recs.name.reset_index(drop=True) for m, recs in outs.items()})
            names.index = [f"#{i + 1}" for i in names.index]
            st.subheader("Side-by-side")
            st.dataframe(names, width="stretch")
            st.caption("❌ prerequisites 'missing' means the student could not actually enrol yet. "
                       "Only the Hybrid removes such courses.")

with tab_res:
    st.write("Results produced by `python evaluate.py` (temporal hold-out on the synthetic dataset).")
    res = Path("results")
    if (res / "summary.csv").exists():
        st.dataframe(pd.read_csv(res / "summary.csv").round(3), hide_index=True, width="stretch")
        for img in ["metrics_comparison.png", "prereq_violations.png", "ablation.png", "precision_by_semester.png"]:
            if (res / img).exists():
                st.image(str(res / img))
    else:
        st.warning("Run `python evaluate.py` first to create the results.")

with tab_about:
    st.markdown("""
**Content-Based:** TF-IDF turns each course (name, category, skills, description) into a vector. The student's
profile = interests + skills + liked completed courses. Courses are ranked by **cosine similarity**.

**Collaborative Filtering (item-based):** builds a student × course rating matrix, computes course-to-course
cosine similarity, and scores a course by its similarity to courses the student liked.

**Academic-Aware Hybrid (our improvement):** `w1·CB + w2·CF + w3·semester fit + w4·difficulty fit`
(weights **tuned on a validation set** by `train.py`) and courses with unmet **prerequisites are removed**.
""")
