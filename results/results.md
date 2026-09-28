## Main results (synthetic data, seed 42, 243 evaluated students, 5965 training ratings)
Hybrid weights (content, collab, semester, difficulty) = (0.3, 0.2, 0.25, 0.25)

| Model | Precision@5 | Recall@5 | F1@5 | Prereq violation rate | Time (ms/student) |
|---|---|---|---|---|---|
| Content-Based | 0.191 | 0.547 | 0.271 | 0.432 | 3.666 |
| Collaborative Filtering | 0.132 | 0.341 | 0.183 | 0.169 | 1.114 |
| Academic-Aware Hybrid | 0.304 | 0.815 | 0.425 | 0.000 | 4.223 |
| Random (baseline) | 0.058 | 0.167 | 0.083 | 0.525 | 1.063 |

## Ablation study
| Model | Precision@5 | Recall@5 | F1@5 | Prereq violation rate | Time (ms/student) |
|---|---|---|---|---|---|
| Content-Based | 0.191 | 0.547 | 0.271 | 0.432 | 3.353 |
| Collaborative Filtering | 0.132 | 0.341 | 0.183 | 0.169 | 1.020 |
| CB + CF only | 0.231 | 0.622 | 0.323 | 0.262 | 3.821 |
| + Prerequisite filter | 0.272 | 0.732 | 0.380 | 0.000 | 3.815 |
| + Semester fit | 0.283 | 0.767 | 0.397 | 0.000 | 3.885 |
| + Difficulty fit (hand-picked weights) | 0.286 | 0.775 | 0.402 | 0.000 | 3.848 |

## Robustness: mean ± std over 5 other random datasets (seeds 1-5)
| Model | Precision@5 | Recall@5 | F1@5 |
|---|---|---|---|
| Content-Based | 0.184 ± 0.008 | 0.517 ± 0.022 | 0.260 ± 0.011 |
| Collaborative Filtering | 0.133 ± 0.017 | 0.346 ± 0.037 | 0.185 ± 0.022 |
| Academic-Aware Hybrid | 0.300 ± 0.011 | 0.822 ± 0.020 | 0.422 ± 0.013 |