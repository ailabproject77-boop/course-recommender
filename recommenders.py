"""
recommenders.py
---------------
All recommendation approaches used in the project.

A "student" is a plain dict:
    {"interests": [str], "skills": [str],
     "history": {course_id: rating 1-5},      # completed courses
     "semester": int, "preferred_difficulty": int (1-5)}

Every recommender has:  recommend(student, k=5) -> DataFrame
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_list(s):
    """'a;b;c' -> ['a','b','c'] (empty string -> [])."""
    return [x for x in str(s).split(";") if x] if isinstance(s, str) else []


def pref_weight(rating):
    """How much a rating counts as 'liked': 1->0, 2->0, 3->1, 4->2, 5->3."""
    return max(float(rating) - 2.0, 0.0)


# ----------------------------------------------------------------------------
class Recommender:
    """Base class: shared course info + the recommend() routine."""
    name = "base"

    def __init__(self, courses):
        self.courses = courses.reset_index(drop=True)
        self.ids = self.courses["course_id"].tolist()
        self.idx = {c: i for i, c in enumerate(self.ids)}
        self.prereqs = {r.course_id: parse_list(r.prerequisites) for r in self.courses.itertuples()}
        self.level = self.courses["level"].to_numpy()
        self.difficulty = self.courses["difficulty"].to_numpy()

    def score_all(self, student):          # -> array, one score per course
        raise NotImplementedError

    def explain(self, student, i):         # -> short text reason
        return ""

    def recommend(self, student, k=5, with_reason=True):
        completed = set(student["history"])
        scores = np.asarray(self.score_all(student), dtype=float).copy()
        scores[np.isin(self.ids, list(completed))] = -np.inf      # never re-recommend
        order = np.argsort(-scores, kind="stable")
        top = [i for i in order if np.isfinite(scores[i])][:k]
        best = scores[top[0]] if top else 0.0
        rows = []
        for rank, i in enumerate(top, 1):
            cid = self.ids[i]
            missing = [p for p in self.prereqs[cid] if p not in completed]
            rows.append({
                "rank": rank, "course_id": cid, "name": self.courses.at[i, "name"],
                "category": self.courses.at[i, "category"],
                "level": int(self.level[i]), "difficulty": int(self.difficulty[i]),
                "score": float(scores[i] / best) if best > 0 else 0.0,   # 1.0 = top course
                "missing_prereqs": ", ".join(missing),
                "reason": self.explain(student, i) if with_reason else "",
            })
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
class ContentBased(Recommender):
    """Approach 1: TF-IDF + cosine similarity between student profile and courses."""
    name = "Content-Based"

    def __init__(self, courses):
        super().__init__(courses)
        c = self.courses
        skills_txt = c["skills"].str.replace(";", " ")
        text = c["name"] + " " + c["category"] + " " + skills_txt + " " + skills_txt + " " + c["description"]
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.X = self.vec.fit_transform(text).toarray()       # courses x terms (rows are unit length)
        self.terms = np.array(self.vec.get_feature_names_out())
        self.fit_info = {"courses": self.X.shape[0], "vocabulary_size": self.X.shape[1]}

    def _profile(self, student):
        """Student profile = average of (declared interests+skills) and (liked completed courses)."""
        parts = []
        query = " ".join(student["interests"]) + " " + " ".join(student["skills"])
        q = self.vec.transform([query]).toarray()[0]
        if q.any():
            parts.append(q)
        w = np.array([pref_weight(r) for r in student["history"].values()])
        if len(w) and w.sum() > 0:
            rows = [self.idx[c] for c in student["history"]]
            parts.append((w[:, None] * self.X[rows]).sum(axis=0) / w.sum())
        return sum(parts) / len(parts) if parts else np.zeros(self.X.shape[1])

    def score_all(self, student):
        return cosine_similarity(self._profile(student)[None, :], self.X)[0]

    def explain(self, student, i):
        contrib = self._profile(student) * self.X[i]
        top = [self.terms[j] for j in np.argsort(-contrib)[:3] if contrib[j] > 0]
        return ("Matches your profile on: " + ", ".join(top)) if top else "Low similarity to your profile"


# ----------------------------------------------------------------------------
class CollaborativeFiltering(Recommender):
    """Approach 2: item-based CF. Similar courses = courses liked by the same students."""
    name = "Collaborative Filtering"

    def __init__(self, courses, train_ratings):
        super().__init__(courses)
        users = sorted(train_ratings["student_id"].unique())
        u_idx = {u: i for i, u in enumerate(users)}
        R = np.zeros((len(users), len(self.ids)))             # students x courses rating matrix
        for r in train_ratings.itertuples():
            R[u_idx[r.student_id], self.idx[r.course_id]] = r.rating
        self.sim = cosine_similarity(R.T)                     # course x course similarity
        np.fill_diagonal(self.sim, 0.0)
        self.fit_info = {"students": R.shape[0], "courses": R.shape[1], "ratings_used": int((R > 0).sum()),
                         "matrix_density": float((R > 0).mean())}

    def _weights(self, student):
        w = np.zeros(len(self.ids))
        for c, r in student["history"].items():
            w[self.idx[c]] = pref_weight(r)
        return w

    def score_all(self, student):
        # score(course i) = sum over liked courses j of  sim(i, j) * how much the student liked j
        return self.sim @ self._weights(student)

    def explain(self, student, i):
        contrib = self.sim[i] * self._weights(student)
        top = [self.courses.at[j, "name"] for j in np.argsort(-contrib)[:2] if contrib[j] > 0]
        return ("Students who liked " + " & ".join(top) + " also took this") if top else "No similar liked courses"


# ----------------------------------------------------------------------------
class AcademicHybrid(Recommender):
    """
    OUR IMPROVEMENT: Academic-Aware Hybrid.
        score = w_cb*CB + w_cf*CF + w_sem*semester_fit + w_diff*difficulty_fit
    and courses whose prerequisites are not completed are removed (hard filter).
    CB and CF scores are min-max normalised to 0-1 among the candidate courses first.
    """
    def __init__(self, cb, cf, weights=(0.35, 0.35, 0.15, 0.15), use_prereq=True,
                 name="Academic-Aware Hybrid"):
        super().__init__(cb.courses)
        self.cb, self.cf = cb, cf
        self.w_cb, self.w_cf, self.w_sem, self.w_diff = weights
        self.use_prereq = use_prereq
        self.name = name

    def _prereq_ok(self, completed):
        return np.array([all(p in completed for p in self.prereqs[c]) for c in self.ids])

    @staticmethod
    def _minmax(x, mask):
        out = np.zeros_like(x, dtype=float)
        if mask.any():
            lo, hi = x[mask].min(), x[mask].max()
            out[mask] = (x[mask] - lo) / (hi - lo) if hi > lo else 0.0
        return out

    def components(self, student):
        """The four normalised parts (0-1) of the hybrid score + the candidate mask."""
        completed = set(student["history"])
        cand = ~np.isin(self.ids, list(completed))
        if self.use_prereq:
            cand &= self._prereq_ok(completed)                # rule 1: prerequisites
        cb = self._minmax(self.cb.score_all(student), cand)
        cf = self._minmax(self.cf.score_all(student), cand)
        sem_fit = np.maximum(0.0, 1 - 0.25 * np.abs(self.level - student["semester"]))          # rule 2
        diff_fit = 1 - np.abs(self.difficulty - student["preferred_difficulty"]) / 4.0          # rule 3
        return cand, cb, cf, sem_fit, diff_fit

    def score_all(self, student):
        cand, cb, cf, sem_fit, diff_fit = self.components(student)
        s = self.w_cb * cb + self.w_cf * cf + self.w_sem * sem_fit + self.w_diff * diff_fit
        return np.where(cand, s, -np.inf)

    def explain(self, student, i):
        cid = self.ids[i]
        pre = self.prereqs[cid]
        parts = [("Prerequisites done (" + ", ".join(pre) + ")") if pre else "No prerequisites",
                 f"Level {self.level[i]} vs your semester {student['semester']}",
                 f"Difficulty {self.difficulty[i]} vs preferred {student['preferred_difficulty']}"]
        for extra in (self.cb.explain(student, i), self.cf.explain(student, i)):
            if extra and not extra.startswith(("Low", "No similar")):
                parts.append(extra)
        return " | ".join(parts)


# ----------------------------------------------------------------------------
class RandomBaseline(Recommender):
    """Sanity-check baseline: random courses."""
    name = "Random (baseline)"

    def __init__(self, courses, seed=0):
        super().__init__(courses)
        self.rng = np.random.default_rng(seed)

    def score_all(self, student):
        return self.rng.random(len(self.ids))
