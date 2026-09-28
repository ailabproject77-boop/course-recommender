# Report content, PPT outline, work division and viva Q&A

All numbers below come from `python evaluate.py` (see `results/results.md`).

---
## PART A – PROJECT REPORT STRUCTURE (with content)

**1. Introduction** – Choosing courses is hard: many electives, prerequisites, different difficulty levels.
Goal: recommend the 5 best next courses for a student.

**2. Problem statement & objectives** – (a) build and compare two AI approaches, (b) design one improvement,
(c) measure with Precision/Recall/F1@5, (d) deliver a working web demo.

**3. Dataset (synthetic)** – 48 CSE courses (8 semesters, difficulty 1-5, prerequisites, 3 skills, description),
300 students (1-3 interests, 3-6 skills, semester 3-8, preferred difficulty 2-4), 7,437 ratings (1-5).
Generation rules: a student may only take a course if prerequisites were completed in an *earlier* semester and the
course level is within [semester-2, semester+1]; choice probability rises with interest match, skill overlap and
difficulty fit; ratings = 3 + interest bonus + skill bonus − difficulty mismatch + noise. Seeded (42) → reproducible.
**Must be declared synthetic** in the report and PPT.

**4. Methodology**
* 4.1 *Content-Based*: text = name + category + skills + description → TF-IDF → student profile (interests, skills,
  liked completed courses) → cosine similarity → rank.
* 4.2 *Collaborative (item-based)*: rating matrix → course-course cosine similarity →
  score(i) = Σ sim(i, j) × liking(j) over courses j the student liked (liking = rating − 2, min 0).
* 4.3 *Academic-Aware Hybrid (our innovation)*:
  `score = w1·CB + w2·CF + w3·semester_fit + w4·difficulty_fit`, then remove courses with unmet prerequisites.
  Weights were first hand-picked (0.35/0.35/0.15/0.15) and then **tuned on a validation set** (see 4.4).
  semester_fit = max(0, 1 − 0.25·|course level − semester|); difficulty_fit = 1 − |course difficulty − preferred| / 4.
  CB and CF scores are min-max normalised to 0-1 first.

* 4.4 *Training and tuning (`train.py`)*: CB and CF are classical, so training = fitting (TF-IDF vocabulary of 344 terms;
  300×48 rating matrix with 31% density; 48×48 similarity matrix). Training data is split temporally into
  TRAIN / VALIDATION / TEST (4,509 / 1,456 / 1,472 ratings). A grid search over 1,771 weight combinations (step 0.05,
  sum = 1) on VALIDATION selects w = (0.30, 0.20, 0.25, 0.25): validation F1 0.447 vs 0.422 for the hand-picked weights.
  The TEST set is used only once, after tuning.

**5. Experimental setup** – temporal hold-out (history before current semester → predict courses in current semester,
relevant = rating ≥ 4), 243 students with ≥1 relevant course, models trained on training ratings only, K = 5.
Metrics: Precision@5, Recall@5, F1@5, prerequisite-violation rate, time.

**6. Results**

| Model | Precision@5 | Recall@5 | F1@5 | Prereq violation | Time (ms) |
|---|---|---|---|---|---|
| Content-Based | 0.191 | 0.547 | 0.271 | 43.2% | 2.96 |
| Collaborative Filtering | 0.132 | 0.341 | 0.183 | 16.9% | 0.90 |
| Academic-Aware Hybrid (tuned) | **0.304** | **0.815** | **0.425** | **0.0%** | 3.55 |
| Random baseline | 0.058 | 0.167 | 0.083 | 52.5% | 0.68 |

Hybrid with hand-picked weights: P 0.286, R 0.775, F1 0.402. Tuning gave +0.023 F1 on the untouched test set.

Ablation (F1@5, hand-picked weights): CB+CF only 0.323 → + prerequisite filter 0.380 → + semester fit 0.397 → + difficulty fit 0.402.
Robustness over 5 other random datasets (mean ± std): CB F1 0.260 ± 0.011, CF 0.185 ± 0.022, Hybrid 0.422 ± 0.013 (weights tuned on seed 42 only, so these datasets are unseen).
Charts: `results/metrics_comparison.png`, `ablation.png`, `prereq_violations.png`, `precision_by_semester.png`.

**7. Discussion** –
* Hybrid F1 is about 57% higher than CB (0.425 vs 0.271) and more than 2× CF (0.183). Random baseline is 0.083, so all methods learn something.
* Most of the gain comes from the **prerequisite filter** (+0.057 F1). Semester fit adds +0.017, difficulty fit +0.005 (small).
* CB recommends un-enrollable courses 43% of the time; the hybrid never does – a benefit that accuracy alone does not show.
* Weight tuning: the top-10 combinations differ by only ~0.001 validation F1 and some give CF weight 0.00–0.10, so the exact weights are not unique and CF contributes little on this data; the gain comes mainly from a larger semester/difficulty weight.
* CF is weakest: it has no ratings for new/advanced courses (cold start) and the data is small.
* Hybrid is slightly slower (≈4 ms) but still real-time.

**8. Limitations** – synthetic data whose generator follows the same academic rules the hybrid uses (so the gain is
partly by design); weights tuned only on synthetic data; only 1.86 relevant courses per student (max Precision@5 ≈ 0.37); no significance test;
no real-student user study.

**9. Conclusion & future work** – Combining content, collaboration and academic rules gave the best and always valid
recommendations. Future: real university data, matrix factorisation, tuning more parameters (e.g. semester-fit slope), grade/GPA and career goals, user study.

**10. References** – Ricci et al., *Recommender Systems Handbook*; scikit-learn docs (TfidfVectorizer, cosine_similarity);
Streamlit docs; Sarwar et al. (2001), *Item-based collaborative filtering recommendation algorithms*.

---
## PART B – PPT OUTLINE (12 slides)

1. Title, team members
2. Problem & motivation
3. Objectives + the teacher's 4 requirements ticked
4. Dataset (**SYNTHETIC**) – table of sizes + a sample of courses
5. System architecture diagram (from README)
6. Approach 1: Content-Based (TF-IDF + cosine, tiny example)
7. Approach 2: Collaborative (item-based cosine, tiny example)
8. Our improvement: Academic-Aware Hybrid (formula + 3 rules)
9. Training: what is fitted + train/validation/test split + `tuning_chart.png`
10. Evaluation protocol and results: table + `metrics_comparison.png` + `prereq_violations.png`
11. Ablation + robustness + limitations
12. Live demo screenshots (or live run), conclusion, future work, Q&A

---
## PART C – DIVISION OF WORK (5 students)

| Student | Responsibility | Files / deliverable |
|---|---|---|
| 1 | Dataset design & generation, data description | `generate_dataset.py`, `data/`, report §3 |
| 2 | Content-Based (TF-IDF, cosine, explanations) | `ContentBased` in `recommenders.py`, report §4.1 |
| 3 | Collaborative Filtering + baseline | `CollaborativeFiltering`, `RandomBaseline`, report §4.2 |
| 4 | Hybrid improvement, training/tuning, evaluation/graphs | `AcademicHybrid`, `train.py`, `evaluate.py`, `results/`, report §4.3, §5-7 |
| 5 | Streamlit demo, integration, README/PPT | `app.py`, `requirements.txt`, README, PPT, testing |

Everyone reads the whole code and prepares the viva answers for their own part.

---
## PART D – VIVA QUESTIONS AND ANSWERS

**Q1. What is your project in one line?** A system that recommends the top 5 next courses using content-based filtering,
collaborative filtering and our academic-aware hybrid, compared with Precision/Recall/F1@5.

**Q2. Which dataset? Is it real?** No, it is synthetic, generated with seeded random rules (48 courses, 300 students,
7,437 ratings). We could not get real, private student records. Results therefore show relative behaviour, not real-world accuracy.

**Q3. What is TF-IDF?** Term Frequency × Inverse Document Frequency: words frequent in one course but rare across courses
get high weight (e.g. "neural"), common words get low weight.

**Q4. What is cosine similarity and why use it?** Cosine of the angle between two vectors; 1 = same direction, 0 = unrelated.
It ignores text length, so long and short descriptions compare fairly.

**Q5. How is the student profile built in content-based?** Average of (a) TF-IDF vector of interests + skills and
(b) rating-weighted average of the vectors of completed courses the student liked.

**Q6. How does your collaborative filtering work?** Item-based: build a student×course rating matrix, compute cosine
similarity between course columns, then score each unseen course by its similarity to courses the student liked.
Explanation: "students who liked X also took this".

**Q7. Why item-based and not user-based / matrix factorisation?** Simple to explain, explainable, stable with
few courses (48) and many students; matrix factorisation is harder to explain and needs tuning.

**Q8. What is your improvement?** The Academic-Aware Hybrid: weighted mix of CB and CF, plus semester fit, difficulty
fit and a hard prerequisite filter, which standard recommenders ignore.

**Q9. How did you choose the hybrid weights?** We first guessed 0.35/0.35/0.15/0.15, then tuned them in `train.py`: grid
search over 1,771 combinations on a validation set (each student's last training semester). Best = 0.30/0.20/0.25/0.25.
Test F1 rose from 0.402 to 0.425. The test set was never used for tuning.

**Q9b. Do you "train" the models?** Yes, in the classical sense: TF-IDF is fitted on course text, the rating matrix and
course similarity are fitted on training ratings, and the hybrid weights are tuned. There are no epochs/loss curves
because these methods have no gradient-based training. `results/training_log.txt` shows the fitting log.

**Q9c. Isn't tuning on 1,771 combinations overfitting?** It can be, so we tuned on validation and reported on a separate
test set (gain held: 0.402 → 0.425) and on 5 unseen random datasets (F1 0.422 ± 0.013).

**Q10. Define Precision@5, Recall@5, F1@5.** Precision = correct recommendations / 5. Recall = correct / all relevant
courses of the student. F1 = 2PR/(P+R).

**Q11. Why is Precision so low (0.29)?** Each student has only 1.86 relevant courses on average, so even a perfect
system could reach about 0.37 Precision@5. Compare methods relatively (and against random = 0.058).

**Q12. How did you avoid data leakage?** Temporal split: models see only courses taken before the student's current
semester; test = what they took in that semester; CF similarity is computed from training ratings only.

**Q13. Which method won and why?** Hybrid: F1 0.425 vs CB 0.271 vs CF 0.183. The prerequisite filter contributed most
(ablation), because CB/CF recommend unavailable courses (43% / 17% violation rate).

**Q14. Isn't the comparison unfair since your data follows prerequisite rules?** Partly yes – we state it as a limitation.
The generator encodes real academic rules, so a rule-aware model benefits. On real data gains would be smaller,
but the prerequisite-violation result would still hold (recommending courses one cannot enrol in is always wrong).

**Q15. Why is CF the weakest?** Cold start: advanced courses have few earlier ratings, and the dataset is small/sparse.
CB does not suffer because it uses course text.

**Q16. What is cold start and how does your system handle it?** New student/course with no ratings. CB handles it via
declared interests/skills and course text; the hybrid inherits this. The demo works with no ratings (default 4).

**Q17. How does the demo explain its recommendations?** Each row shows prerequisites met, level vs semester, difficulty vs
preference, top matching keywords (CB) and the liked courses behind the CF score.

**Q18. What is the time complexity/speed?** ≈1-4 ms per student for 48 courses; similarity matrices are precomputed once.

**Q19. What are the limitations?** Synthetic data, weights tuned only on synthetic data, no significance test, no GPA/career goals, small catalogue.

**Q20. How would you extend it?** Real university data, matrix factorisation, hyperparameter tuning, credit-limit and
timetable constraints, A/B test with real students.

**Q21. How do I run it?** `pip install -r requirements.txt` → `python train.py` → `python evaluate.py` → `streamlit run app.py`.
