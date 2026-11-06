"""Central configuration for ExamLens.

Every tunable value lives here so that nothing is hard-coded across the project.
"""

SBERT_MODEL_NAME="all-MiniLM-L6-v2"
RANDOM_STATE=42
UNMAPPED="Unmapped"

# ---- question types (canonical names, used everywhere) ----
QUESTION_TYPES=["Definition","Theory / Explanation","Numerical / Problem Solving",
                "Derivation","Application","Comparison","Proof","Other"]
TYPE_SHORT={"Definition":"Defn","Theory / Explanation":"Theory",
            "Numerical / Problem Solving":"Numerical","Derivation":"Deriv",
            "Application":"Applic","Comparison":"Compar","Proof":"Proof","Other":"Other"}

# ---- semantic similarity (repeated / rephrased questions) ----
SIM_HIGH=0.85                  # >= this -> highly similar
SIM_MID=0.70                   # >= this (and < SIM_HIGH) -> similar
INCLUDE_SAME_YEAR_PAIRS=False  # compare only across different years
MAX_PAIRS_SHOWN=25

# ---- topic / concept matching ----
CONCEPT_THRESHOLD=0.35         # minimum semantic score for a concept match
LEXICAL_CONCEPT_SCORE=0.90     # score given when the concept phrase appears literally
MAX_CONCEPTS_PER_QUESTION=3
LEXICAL_TOPIC_BONUS=0.15       # bonus when a topic/concept phrase literally appears
TOPIC_MIN_SIM=0.20             # below this a question is labelled Unmapped

# ---- SVM topic classifier ----
PSEUDO_LABEL_MARGIN=0.05       # min gap between best and 2nd best topic score
MIN_SVM_LABELS=30              # minimum confident labels to train SVM
MIN_SVM_PER_CLASS=3            # minimum labels per topic to be a class
MIN_SVM_CLASSES=2
SVM_MIN_MARGIN=0.15            # SVM decision margin needed to override semantic label

# ---- Random Forest difficulty ----
MIN_RF_SAMPLES=30
DIFF_EASY_MAX=0.35             # weak-label score thresholds
DIFF_HARD_MIN=0.60
RF_PARAMS=dict(n_estimators=200,max_depth=6,min_samples_leaf=2,
               class_weight="balanced",random_state=RANDOM_STATE)

# ---- K-Means ----
KMEANS_FIXED_K=None            # set an int to force the number of clusters
KMEANS_MIN_K=2
KMEANS_MAX_K=8
MIN_QUESTIONS_FOR_KMEANS=8

# ---- trends / importance ----
RECENT_WINDOW=2                # number of latest years counted as "recent"
RECENCY_DECAY=0.7              # weight multiplier per year of age
TREND_PP_THRESHOLD=1.0         # percentage points per year to call a trend
TREND_MIN_YEARS=3
IMPORTANCE_WEIGHTS=dict(frequency=0.35,marks=0.30,recency=0.20,coverage=0.15)
TOP_TOPICS_IN_TREND=8
TOP_CONCEPTS_PER_TOPIC=5
MAX_TOPICS_IN_CONCEPT_REPORT=8

# ---- future prediction (XGBoost) ----
MIN_TRAIN_YEARS=2              # first walk-forward fold trains on this many years
MIN_XGB_ROWS=12                # minimum training rows (topic x year)
MIN_XGB_PER_CLASS=2
TOP_N_PREDICTIONS=8
XGB_PARAMS=dict(n_estimators=120,max_depth=3,learning_rate=0.08,subsample=0.9,
                colsample_bytree=0.9,reg_lambda=1.0,eval_metric="logloss",
                random_state=RANDOM_STATE,verbosity=0,n_jobs=1)

# ---- report output ----
REPORT_DIR="reports"                 # the full text report is saved here (a new timestamped file every run)
REPORT_PREFIX="exam_lens_report"
CONSOLE_HIDDEN_SECTIONS=("similar_questions","hidden_groups","future_prediction")   # saved to the .txt only

# ---- syllabus parsing ----
CONCEPT_MAX_WORDS=8
