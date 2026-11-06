# ExamLens - Exam Paper Intelligence System

An ML-driven exam intelligence system with an NLP semantic layer, run as a Python console application.

## Project Overview
Exam preparation is guesswork when previous-year papers are just unstructured PDFs. ExamLens reads a syllabus PDF and several previous-year question-paper PDFs, turns them into structured data, learns historical patterns, and reports which topics, concepts and question patterns matter and which topics are likely to appear next. All results are calculated from the PDFs you supply; nothing is fabricated.

## Objectives
- Structure raw PDFs into questions, marks, sections and years.
- Measure topic, concept, question-type and difficulty distributions and their trends over years.
- Detect repeated and rephrased questions by meaning.
- Estimate which topics are likely in a future paper, with probabilities and supporting evidence.
- Be honest about small datasets and missing labels.

## Features
PDF processing (PyMuPDF) | Sentence-BERT embeddings | SVM topic classification | Random Forest difficulty estimation | XGBoost future-topic prediction | K-Means clustering | cosine-similarity search | walk-forward temporal validation | NetworkX knowledge graph | topic analysis | concept analysis | question-type analysis | trend analysis | weighted topic importance | future prediction.

## Architecture
```text
PDF input -> validation -> year detection -> PyMuPDF extraction -> question parsing
   -> preprocessing -> feature engineering
        NLP / semantic layer: Sentence-BERT, cosine similarity, concept matching, NetworkX graph
        ML layer: SVM (topic), Random Forest (difficulty), XGBoost (future), K-Means (groups)
        Temporal validation (walk-forward)
   -> analysis engine -> one complete report (saved as .txt) + filtered console view
```
Folders: `extractor/`, `preprocessing/`, `nlp/`, `models/`, `analysis/`, `knowledge_graph/`, `prediction/`, `utils/`. All thresholds are in `config.py`.

NLP transforms and semantically represents exam text, while machine learning performs classification, prediction, clustering and model-based analysis.

## Why Each Algorithm Is Used
- **Sentence-BERT (all-MiniLM-L6-v2):** captures meaning, so rephrased questions and syllabus concepts match even with different words.
- **Cosine similarity:** simple, scale-free comparison of embeddings; thresholds are configurable (`SIM_HIGH`, `SIM_MID`).
- **SVM (TF-IDF + LinearSVC):** strong, fast text classifier for small datasets.
- **Random Forest:** robust on small tabular feature sets.
- **XGBoost:** gradient-boosted trees suit small, mixed numeric topic-level features and give probabilities.
- **K-Means:** unsupervised discovery of question groups not defined by the syllabus; k is chosen by silhouette score.
- **Temporal validation:** random splits would leak future information; walk-forward training mimics real forecasting.
- **NetworkX:** lightweight in-memory graph for Topic -> Concept -> Question -> Year / Marks / Type, with no database server.

## Installation
```text
pip install -r requirements.txt
```
The first run downloads the Sentence-BERT model (internet needed once).

## Run
```text
python main.py
```

## User Input
No file paths are hard-coded. At runtime you enter the syllabus PDF path, the number of papers, and each paper's path. Every path is validated and re-requested if invalid. The year is detected from the filename, then first-page text, then PDF title/subject; if still unclear you are asked.

## PDF Requirements
PDFs need a text layer. Nothing about topics, years or question counts is hard-coded; everything is read from the PDFs.

**Question papers.** Question numbers may be written `1.`, `Q1.`, `Question 1` or `1)`; sub-parts as `(a)`/`a)` and `(i)`/`i)` (for example `1. a) i)`), on the same line as the text or on lines of their own. Sub-parts such as `a) i) ... ii) ...` become separate questions; a part that is a scenario followed by `i) ... ii) ...` stays one question. Marks are read from, in order: marks written next to the question (`(4)`, `[5]`, `(5 marks)`), a rule in the instructions (for example "subdivision (b) carries 6 marks each", or "Each question carries 20 marks"), then `Section A ... 2 marks each` headers. Instructions, page headers/footers, Bloom-level/CO tags such as `(L4)`, table debris and `OR` lines are ignored. The year is taken from the filename, then the examination line (month + year) on the first page, then the PDF metadata; if still unclear you are asked.

**Syllabus.** Either numbered units (`Unit 1: Topic`, `Module 2 - Topic`, `3. Topic`) followed by concept lists, or blocks written as `TOPIC TITLE IN CAPITALS: concept - concept - concept (hours)`. A block with no title is named after its first concept phrase. Concepts are split on commas, semicolons, dashes and colons; hour counts and short acronyms in brackets are dropped.

Scanned/image-only PDFs are reported as unreadable because OCR is not implemented. A malformed PDF is skipped and reported; it does not stop the run.

## ML Workflow
1. **Topics (SVM):** the only labels available are high-confidence pseudo-labels (a literal syllabus phrase match that agrees with the semantic winner by a clear margin). These are NOT human-verified. If enough exist, TF-IDF + LinearSVC is trained on them and used for lower-confidence questions; otherwise SBERT semantic matching alone is used. The report states which mode ran.
2. **Difficulty (Random Forest):** real difficulty labels do not exist. Weak labels come from a documented heuristic (question type, relative marks, length, numeric content, concept count). The Random Forest learns from them, so results are an estimate, never an objective label. Below the sample threshold only the heuristic estimate is shown.
3. **K-Means:** clusters embeddings; clusters are reported by number with sizes, representative questions and dominant topic/type, not invented names.

## NLP Workflow
Questions, topics and concepts are embedded with Sentence-BERT and L2-normalised, so a dot product is cosine similarity. Pairs above the thresholds across different years are listed once each. Concept matching combines cosine similarity with literal phrase matches. If Sentence-BERT cannot load, TF-IDF vectors are used and clearly labelled as a fallback.

## Future Prediction
XGBoost predicts, per syllabus topic, the probability of appearing in the next paper. Features (frequency, recent frequency, year coverage, marks share/trend, years since last appearance, streaks, concept recurrence, repeated questions, difficulty, type mix and more) for a target year use only earlier years. Validation is walk-forward (e.g. train 2021-2022, test 2023; then train 2021-2023, test 2024), with years taken dynamically from the papers. Metrics (accuracy, precision, recall, F1, ROC-AUC where meaningful) are compared against a naive "appeared last year" baseline. If there are too few years or rows, metrics are not invented; the report says validation is limited and falls back to a clearly labelled recency-weighted score. Likely concepts, question type and difficulty are recency-weighted historical statistics, not XGBoost outputs. Exact question wording is never predicted.

## Knowledge Graph
Nodes: topics, concepts, questions, years, marks, question types. Edges: topic has concept, concept asked in question, question in year / carries marks / has type, and similar-to between repeated questions. It powers the concept analysis and the concept-recurrence and repeated-question features. It is a representation layer, not a predictor.

## Report Output
One complete report is generated after the analysis and saved automatically to `reports/exam_lens_report_<timestamp>.txt` (a new file every run; earlier reports are never overwritten). The same report is printed to the console, except for three sections that are written to the text file only (set in `config.CONSOLE_HIDDEN_SECTIONS`).

Sections in the text file: Dataset Summary, Difficulty Analysis, Topic Analysis, Concept Analysis, Question Type Analysis, Topic Trend by Year, Similar or Repeated Questions, Hidden Question Groups, Important Topics, Future Exam Prediction, Summary.

Console view: the same sections without Similar or Repeated Questions, Hidden Question Groups and Future Exam Prediction (sections are numbered consecutively in each output). Importance score = 0.35 frequency + 0.30 marks + 0.20 recency + 0.15 year coverage (analytical, not ML). Only aggregates are shown for topics and question types.

## Limitations
- Small datasets: few papers give weak models and reduced confidence.
- PDF extraction depends on layout; unusual numbering or tables may be mis-parsed.
- No true difficulty labels; topic "labels" are pseudo-labels.
- Topic assignment is computed once over all papers, a mild transductive effect on temporal validation.
- Few historical years make validation limited or impossible.
- Future predictions are probabilistic and uncertain.
- Scanned PDFs are unsupported without OCR.
- Concepts come from the syllabus text; concepts not listed there are not discovered.

## Future Improvements
OCR for scanned PDFs, human-verified topic and difficulty labels, larger datasets, fine-tuned transformer models, richer question-type classifiers, advanced time-series models, ensemble prediction, better automatic concept extraction.

## Viva Description
ExamLens is an ML-driven exam intelligence system that analyzes syllabus and historical question-paper PDFs. NLP techniques such as Sentence-BERT are used to convert unstructured exam questions into semantic representations and identify relationships between questions and concepts. Machine-learning models including SVM, Random Forest, XGBoost and K-Means are then used for topic classification, difficulty prediction, future topic prediction and unsupervised pattern discovery. Temporal validation prevents future-data leakage. A NetworkX knowledge graph represents relationships among topics, concepts, questions, years, marks and question types.
