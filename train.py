"""
train.py
--------
The TRAINING step of the project. Run:  python train.py

What is "trained" here
  1. Content-Based : TF-IDF is FITTED on course text (learns vocabulary + word weights).
  2. Collaborative : rating matrix + course-course similarity are FITTED from training ratings.
  3. Hybrid        : its 4 weights are TUNED by grid search on a VALIDATION set.

Data split (three parts, no leakage):
  TRAIN       = each student's ratings before their last training semester   -> fit CB / CF while tuning
  VALIDATION  = each student's last training semester                       -> choose the hybrid weights
  TEST        = the student's current semester (never used while tuning)    -> final honest score

Outputs: results/training_log.txt, results/tuning_top10.csv, results/tuning_chart.png,
         results/final_test_comparison.csv, results/best_weights.json, models/*.joblib
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_dataset import save
from recommenders import ContentBased, CollaborativeFiltering, AcademicHybrid, parse_list
from evaluate import split, evaluate, summarize, to_md, K

RESULTS, MODELS = Path("results"), Path("models")
HAND_PICKED = (0.35, 0.35, 0.15, 0.15)
STEP = 0.05                                                    # weight grid resolution
LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def weight_grid(step=STEP):
    """All (w_content, w_collab, w_semester, w_difficulty) that are multiples of `step` and sum to 1."""
    n = round(1 / step)
    return np.array([(a, b, c, n - a - b - c) for a in range(n + 1) for b in range(n + 1 - a)
                     for c in range(n + 1 - a - b)]) / n


def main():
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    courses, students, ratings = save("data", seed=42)
    train_full, test = split(students, ratings)

    # ---------------- 1. create TRAIN / VALIDATION split inside the training data
    last = train_full.groupby("student_id")["semester_taken"].transform("max")
    sub = train_full[train_full.semester_taken < last]           # TRAIN
    val = train_full[train_full.semester_taken == last]          # VALIDATION
    val_sem = last.groupby(train_full.student_id).first().rename("val_sem")
    val_students = students.merge(val_sem, left_on="student_id", right_index=True)
    val_students = val_students.assign(semester=val_students.val_sem)
    log("=== DATA SPLIT ===")
    log(f"Train (fit CB/CF while tuning): {len(sub):>5} ratings")
    log(f"Validation (tune weights)     : {len(val):>5} ratings")
    log(f"Test (final score only)       : {len(test):>5} ratings")

    # ---------------- 2. fit CB and CF on TRAIN only
    log("\n=== FITTING MODELS (on TRAIN part) ===")
    t0 = time.perf_counter(); cb = ContentBased(courses); t_cb = time.perf_counter() - t0
    t0 = time.perf_counter(); cf = CollaborativeFiltering(courses, sub); t_cf = time.perf_counter() - t0
    log(f"Content-Based : TF-IDF fitted on {cb.fit_info['courses']} courses, vocabulary = "
        f"{cb.fit_info['vocabulary_size']} terms  ({t_cb * 1000:.0f} ms)")
    i = cf.fit_info
    log(f"Collaborative : rating matrix {i['students']} students x {i['courses']} courses, "
        f"{i['ratings_used']} ratings, density {i['matrix_density']:.1%}; similarity matrix "
        f"{i['courses']}x{i['courses']}  ({t_cf * 1000:.0f} ms)")

    # ---------------- 3. tune hybrid weights by grid search on VALIDATION
    log("\n=== TUNING HYBRID WEIGHTS (grid search on VALIDATION) ===")
    hybrid = AcademicHybrid(cb, cf)
    hist = {sid: dict(zip(g.course_id, g.rating)) for sid, g in sub.groupby("student_id")}
    relevant = {sid: set(g[g.rating >= 4].course_id) for sid, g in val.groupby("student_id")}
    comps, rel_masks = [], []
    for s in val_students.itertuples():
        rel = relevant.get(s.student_id)
        if not rel:
            continue
        st = dict(interests=parse_list(s.interests), skills=parse_list(s.skills),
                  history=hist.get(s.student_id, {}), semester=int(s.semester),
                  preferred_difficulty=int(s.preferred_difficulty))
        cand, c1, c2, c3, c4 = hybrid.components(st)
        comps.append((cand, np.stack([c1, c2, c3, c4])))
        rel_masks.append(np.isin(hybrid.ids, list(rel)))

    W = weight_grid()
    f1s, precs = np.zeros(len(W)), np.zeros(len(W))
    for (cand, C), rel in zip(comps, rel_masks):
        S = np.where(cand[None, :], W @ C, -np.inf)                       # (combos, courses)
        top = np.argsort(-S, axis=1, kind="stable")[:, :K]
        hit = rel[top] & np.isfinite(np.take_along_axis(S, top, axis=1))
        h = hit.sum(1)
        p, r = h / K, h / rel.sum()
        f1s += np.where(h > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
        precs += p
    f1s /= len(comps)
    precs /= len(comps)
    log(f"{len(W)} weight combinations (step {STEP}) x {len(comps)} validation students")

    order = np.lexsort((-precs, -f1s))                                   # best F1, tie -> best precision
    best = W[order[0]]
    hp_row = int(np.where((np.abs(W - np.array(HAND_PICKED)) < 1e-9).all(axis=1))[0][0])
    wcols = ["w_content", "w_collab", "w_semester", "w_difficulty"]
    top10 = pd.DataFrame(W[order[:10]], columns=wcols)
    top10["Validation F1@5"] = f1s[order[:10]]
    top10.insert(0, "Rank", range(1, 11))
    top10.to_csv(RESULTS / "tuning_top10.csv", index=False)
    log(f"Hand-picked {HAND_PICKED} -> validation F1@5 = {f1s[hp_row]:.3f}")
    log(f"Best found  {tuple(float(x) for x in best)} -> validation F1@5 = {f1s[order[0]]:.3f}")
    log(f"Range over the whole grid: {f1s.min():.3f} to {f1s.max():.3f}")

    # sanity check: the fast grid result must equal the normal evaluate() on validation
    chk = evaluate([AcademicHybrid(cb, cf, HAND_PICKED)], val_students, sub, val)
    assert abs(chk.f1.mean() - f1s[hp_row]) < 1e-9, "grid search and evaluate() disagree"

    labels = [f"#{r}\n{w[0]:.2f}\n{w[1]:.2f}\n{w[2]:.2f}\n{w[3]:.2f}"
              for r, w in zip(top10.Rank, top10[wcols].to_numpy())]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(10), top10["Validation F1@5"], label="Top-10 tuned combinations")
    ax.axhline(f1s[hp_row], color="red", ls="--", label=f"Hand-picked {HAND_PICKED}")
    ax.set_xticks(range(10)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(min(f1s[order[:10]].min(), f1s[hp_row]) - 0.03, f1s[order[0]] + 0.02)
    ax.set_ylabel("Validation F1@5")
    ax.set_title("Top-10 weight combinations (rows under each bar: content / collab / semester / difficulty)", fontsize=10)
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / "tuning_chart.png", dpi=150); plt.close(fig)

    # ---------------- 4. refit on FULL training data and score once on TEST
    log("\n=== FINAL MODELS (fitted on ALL training data) - TEST SCORES ===")
    cb, cf = ContentBased(courses), CollaborativeFiltering(courses, train_full)
    tuned_w = tuple(float(x) for x in best)
    tuned = AcademicHybrid(cb, cf, tuned_w, name="Hybrid (tuned weights)")
    hand = AcademicHybrid(cb, cf, HAND_PICKED, name="Hybrid (hand-picked weights)")
    models = [cb, cf, hand, tuned]
    res = evaluate(models, students, train_full, test)
    tab = summarize(res, [m.name for m in models])
    tab.to_csv(RESULTS / "final_test_comparison.csv", index=False)
    log(to_md(tab))

    # ---------------- 5. save trained artifacts
    joblib.dump({"vectorizer": cb.vec, "course_vectors": cb.X}, MODELS / "content_based.joblib")
    joblib.dump({"item_similarity": cf.sim, "course_ids": cf.ids}, MODELS / "collaborative.joblib")
    (RESULTS / "best_weights.json").write_text(json.dumps(
        dict(weights=dict(zip(["content", "collab", "semester", "difficulty"], tuned_w)),
             validation_f1=float(f1s[order[0]]), hand_picked_validation_f1=float(f1s[hp_row]),
             test_f1_tuned=float(tab.iloc[3]["F1@5"]), test_f1_hand_picked=float(tab.iloc[2]["F1@5"])), indent=2))
    log("\nSaved: models/*.joblib, results/best_weights.json, tuning_top10.csv, tuning_chart.png, "
        "final_test_comparison.csv, training_log.txt")
    (RESULTS / "training_log.txt").write_text("\n".join(LOG))


if __name__ == "__main__":
    main()
