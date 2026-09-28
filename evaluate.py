"""
evaluate.py
-----------
Runs the real experiment and writes tables + charts to ./results

Protocol (temporal hold-out):
  * For each student, everything taken BEFORE their current semester = training history.
  * The courses they actually took IN their current semester and rated >= 4 = "relevant" (hidden test set).
  * All models are trained only on training history (no leakage).
  * Each model recommends top-5 courses; we compare with the relevant set.

Metrics (per student, then averaged):
  Precision@5 = hits / 5      Recall@5 = hits / #relevant      F1@5 = harmonic mean
  Prereq-violation rate = share of recommended courses whose prerequisites are NOT completed
  Time = milliseconds per recommendation (includes generating the reason text)
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_dataset import generate, save
from recommenders import (ContentBased, CollaborativeFiltering, AcademicHybrid,
                          RandomBaseline, parse_list)

K = 5
RESULTS = Path("results")
MAIN = ["Content-Based", "Collaborative Filtering", "Academic-Aware Hybrid"]


def split(students, ratings):
    r = ratings.merge(students[["student_id", "semester"]], on="student_id")
    return r[r.semester_taken < r.semester], r[r.semester_taken == r.semester]


def hybrid_weights():
    """Weights tuned by train.py (on the validation set) if available, else the hand-picked ones."""
    p = RESULTS / "best_weights.json"
    if p.exists():
        w = json.loads(p.read_text())["weights"]
        return (w["content"], w["collab"], w["semester"], w["difficulty"])
    return (0.35, 0.35, 0.15, 0.15)


def build_models(courses, train):
    cb = ContentBased(courses)
    cf = CollaborativeFiltering(courses, train)
    hybrid = AcademicHybrid(cb, cf, hybrid_weights())
    return cb, cf, hybrid


def evaluate(models, students, train, test, k=K, with_reason=True):
    history = {sid: dict(zip(g.course_id, g.rating)) for sid, g in train.groupby("student_id")}
    relevant = {sid: set(g[g.rating >= 4].course_id) for sid, g in test.groupby("student_id")}
    rows = []
    for s in students.itertuples():
        rel = relevant.get(s.student_id, set())
        if not rel:
            continue
        student = dict(interests=parse_list(s.interests), skills=parse_list(s.skills),
                       history=history.get(s.student_id, {}), semester=int(s.semester),
                       preferred_difficulty=int(s.preferred_difficulty))
        for m in models:
            t0 = time.perf_counter()
            recs = m.recommend(student, k, with_reason=with_reason)
            ms = (time.perf_counter() - t0) * 1000
            hits = len(set(recs.course_id) & rel)
            p, r = hits / k, hits / len(rel)
            rows.append(dict(model=m.name, student_id=s.student_id, semester=int(s.semester),
                             precision=p, recall=r, f1=(2 * p * r / (p + r) if hits else 0.0),
                             prereq_violation=float((recs.missing_prereqs != "").mean()) if len(recs) else 0.0,
                             time_ms=ms))
    return pd.DataFrame(rows)


def summarize(res, order):
    out = res.groupby("model")[["precision", "recall", "f1", "prereq_violation", "time_ms"]].mean()
    out = out.loc[order].reset_index()
    out.columns = ["Model", "Precision@5", "Recall@5", "F1@5", "Prereq violation rate", "Time (ms/student)"]
    return out


def to_md(df, fmt=".3f"):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:{fmt}}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ charts
def bar_chart(summary, path, title):
    metrics = ["Precision@5", "Recall@5", "F1@5"]
    x, w = np.arange(len(summary)), 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(metrics):
        bars = ax.bar(x + (i - 1) * w, summary[m], w, label=m)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Model"], rotation=12, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def violation_chart(summary, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(summary["Model"], summary["Prereq violation rate"] * 100, color=["#4c72b0", "#dd8452", "#55a868", "#999999"][:len(summary)])
    ax.bar_label(bars, fmt="%.1f%%")
    ax.set_ylabel("% recommended courses with missing prerequisites")
    ax.set_title("Prerequisite violations in Top-5 (lower is better)")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def semester_chart(res, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in MAIN:
        g = res[res.model == m].groupby("semester")["precision"].mean()
        ax.plot(g.index, g.values, marker="o", label=m)
    ax.set_xlabel("Student's semester")
    ax.set_ylabel("Precision@5")
    ax.set_title("Precision@5 by semester")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    RESULTS.mkdir(exist_ok=True)
    courses, students, ratings = save("data", seed=42)         # regenerate -> always consistent
    train, test = split(students, ratings)
    cb, cf, hybrid = build_models(courses, train)

    # 1) Main comparison (seed 42) + random baseline
    rnd = RandomBaseline(courses)
    res = evaluate([cb, cf, hybrid, rnd], students, train, test)
    n_eval = res.student_id.nunique()
    main_tab = summarize(res, MAIN + [rnd.name])
    res.to_csv(RESULTS / "per_student_results.csv", index=False)
    main_tab.to_csv(RESULTS / "summary.csv", index=False)
    bar_chart(main_tab, RESULTS / "metrics_comparison.png", f"Top-{K} recommendation quality (seed 42, {n_eval} students)")
    violation_chart(main_tab, RESULTS / "prereq_violations.png")
    semester_chart(res, RESULTS / "precision_by_semester.png")

    # 2) Ablation: which part of the improvement helps?
    variants = [
        cb, cf,
        AcademicHybrid(cb, cf, (0.5, 0.5, 0, 0), use_prereq=False, name="CB + CF only"),
        AcademicHybrid(cb, cf, (0.5, 0.5, 0, 0), use_prereq=True, name="+ Prerequisite filter"),
        AcademicHybrid(cb, cf, (0.4, 0.4, 0.2, 0), use_prereq=True, name="+ Semester fit"),
        AcademicHybrid(cb, cf, (0.35, 0.35, 0.15, 0.15), use_prereq=True, name="+ Difficulty fit (hand-picked weights)"),
    ]
    abl = evaluate(variants, students, train, test)
    abl_tab = summarize(abl, [v.name for v in variants])
    abl_tab.to_csv(RESULTS / "ablation.csv", index=False)
    bar_chart(abl_tab, RESULTS / "ablation.png", "Ablation: adding academic rules step by step")

    # 3) Robustness: 5 other random datasets
    per_seed = []
    for sd in [1, 2, 3, 4, 5]:
        c2, s2, r2 = generate(seed=sd)
        tr2, te2 = split(s2, r2)
        a, b, h = build_models(c2, tr2)
        t = summarize(evaluate([a, b, h], s2, tr2, te2), MAIN)
        t["seed"] = sd
        per_seed.append(t)
    allseeds = pd.concat(per_seed)
    g = allseeds.groupby("Model")[["Precision@5", "Recall@5", "F1@5"]].agg(["mean", "std"])
    multi = pd.DataFrame({"Model": MAIN})
    for m in ["Precision@5", "Recall@5", "F1@5"]:
        multi[m] = [f"{g.loc[x, (m, 'mean')]:.3f} ± {g.loc[x, (m, 'std')]:.3f}" for x in MAIN]
    multi.to_csv(RESULTS / "multi_seed.csv", index=False)

    # 4) Save everything as markdown for the report
    md = [f"## Main results (synthetic data, seed 42, {n_eval} evaluated students, {len(train)} training ratings)",
          f"Hybrid weights (content, collab, semester, difficulty) = {hybrid_weights()}", "",
          to_md(main_tab), "", "## Ablation study", to_md(abl_tab), "",
          "## Robustness: mean ± std over 5 other random datasets (seeds 1-5)", to_md(multi)]
    (RESULTS / "results.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nCharts and CSV files saved in ./results")


if __name__ == "__main__":
    main()
