"""ExamLens - Exam Paper Intelligence System (console application)."""
import sys

import numpy as np
import pandas as pd

import config
from analysis import (concept_analysis,dataset_analysis,difficulty_analysis,important_topics,
                      question_type_analysis,repeated_questions,topic_analysis,trend_analysis)
from extractor import pdf_extractor as px
from knowledge_graph import graph_builder
from models import clustering,difficulty_model,future_predictor,topic_classifier
from nlp.concept_matcher import ConceptMatcher
from nlp.embeddings import Embedder
from nlp.semantic_similarity import find_similar_pairs,semantic_search
from preprocessing import text_processor as tp
from prediction import prediction_report
from utils import helpers,report
from utils.file_validation import ask_pdf_path,ask_positive_int,ask_year


def load_syllabus():
    while True:
        path=ask_pdf_path("Enter the path of the syllabus PDF: ")
        info=px.extract_pdf(path)
        if not info["ok"]:
            print("Could not read text from this PDF: "+"; ".join(info["issues"]))
            continue
        topics=px.parse_syllabus_topics(info["text"])
        if not topics:
            print("No syllabus topics could be detected in this PDF. Please provide another syllabus PDF.")
            continue
        print(f"  Syllabus read: {len(topics)} topics detected, {sum(len(t['concepts']) for t in topics)} concept phrases.")
        return info,topics


def load_papers(count):
    papers=[]
    issues=[]
    for i in range(1,count+1):
        path=ask_pdf_path(f"Enter path for question paper {i}: ")
        info=px.extract_pdf(path)
        for msg in info["issues"]:
            issues.append(f"{info['name']}: {msg}")
        if not info["ok"]:
            print(f"  Skipped (unreadable): {info['name']}")
            continue
        questions=px.parse_questions(info["text"])
        if not questions:
            issues.append(f"{info['name']}: no questions could be detected (unusual layout?)")
            print(f"  Skipped (no questions detected): {info['name']}")
            continue
        year,source=px.detect_year(info)
        if year is None:
            year=ask_year("Enter the year for this question paper: ")
            source="user input"
        else:
            print(f"  Year {year} detected from {source}.")
        no_marks=sum(1 for q in questions if q["marks"] is None)
        if no_marks:
            issues.append(f"{info['name']}: marks missing for {no_marks} of {len(questions)} questions")
        info.update(questions=questions,year=year,year_source=source,stated_total=px.stated_total_marks(info["text"]))
        papers.append(info)
        print(f"  Loaded {info['name']}: {len(questions)} questions.")
    papers.sort(key=lambda p:p["year"])
    return papers,issues


def build_question_table(papers):
    rows=[]
    for idx,p in enumerate(papers):
        for q in p["questions"]:
            rows.append(dict(paper_idx=idx,year=p["year"],paper=p["name"],qid=q["qid"],section=q["section"],
                             text=q["text"],marks=q["marks"],is_sub=q["is_sub"]))
    df=pd.DataFrame(rows)
    df["marks"]=df["marks"].astype(float)
    df["embed_text"]=df["text"].apply(tp.embedding_ready)
    df["tfidf_text"]=df["text"].apply(tp.tfidf_ready)
    df["words"]=df["text"].apply(lambda t:len(t.split()))
    df["numeric_flag"]=df["text"].apply(tp.numeric_flag)
    df["qtype"]=df["text"].apply(tp.classify_question_type)
    return df.reset_index(drop=True)


def run_pipeline(syllabus,topics,papers,n_requested,issues):
    print("\nProcessing: embeddings, topic matching, ML models ...")
    df=build_question_table(papers)
    years=sorted(df["year"].unique().tolist())

    embedder=Embedder()
    if not embedder.is_sbert:
        print(f"  Note: the sentence-embedding model could not be loaded ({embedder.load_error}); "
              "a TF-IDF text representation is being used instead, so similarity scores are less semantic.")
    embedder.prepare(list(df["embed_text"])+[t["name"] for t in topics]+[c for t in topics for c in t["concepts"]])
    emb=embedder.encode(df["embed_text"])

    matcher=ConceptMatcher(topics,embedder)
    scores,lex=matcher.topic_scores(emb,list(df["embed_text"]))
    df["topic"]=topic_classifier.classify_topics(df,scores,lex,matcher.names)
    df["concepts"]=[matcher.match_concepts(df["topic"].iloc[i],emb[i],df["embed_text"].iloc[i]) for i in range(len(df))]

    pairs=repeated_questions.enrich(find_similar_pairs(df,emb),df)
    recurrence=repeated_questions.pair_counts_per_question(pairs,len(df))
    difficulty_model.run_difficulty(df,emb,recurrence)
    clusters=clustering.cluster_questions(df,emb)
    G=graph_builder.build_graph(df,topics,pairs)

    prediction=future_predictor.predict_future(df,topics,years,pairs)
    topic_rows=topic_analysis.compute(df,years)

    ctx=dict(df=df,years=years,topics=topics,pairs=pairs,clusters=clusters,graph=G,prediction=prediction,
             topic_rows=topic_rows,embedder=embedder,emb=emb,has_marks=bool(df["marks"].notna().any()))
    ctx["dataset"]=dataset_analysis.compute(syllabus,topics,papers,n_requested,issues,df,years)
    ctx["difficulty_res"]=difficulty_analysis.compute(df,years)
    ctx["concepts"]=concept_analysis.compute(G,years)
    ctx["qtype_dist"]=question_type_analysis.compute(df)
    ctx["topic_trend"]=trend_analysis.compute_topic_trend(df,years,[r["topic"] for r in topic_rows])
    ctx["importance"]=important_topics.compute(df,years)
    return ctx


def build_sections(ctx):
    years=ctx["years"]
    S=report.section
    return [
        S("dataset","Dataset Summary",dataset_analysis.build_report(ctx["dataset"])),
        S("difficulty","Difficulty Analysis",difficulty_analysis.build_report(ctx["difficulty_res"],years)),
        S("topics","Topic Analysis",topic_analysis.build_report(ctx["topic_rows"],years,ctx["has_marks"])),
        S("concepts","Concept Analysis",concept_analysis.build_report(ctx["concepts"],ctx["topic_rows"],years,ctx["has_marks"])),
        S("question_types","Question Type Analysis",question_type_analysis.build_report(ctx["qtype_dist"])),
        S("topic_trend","Topic Trend by Year",trend_analysis.build_topic_trend_report(ctx["topic_trend"],years,ctx["topic_rows"])),
        S("similar_questions","Similar or Repeated Questions",repeated_questions.build_similar_report(ctx["pairs"])),
        S("hidden_groups","Hidden Question Groups",repeated_questions.build_groups_report(ctx["clusters"],ctx["df"])),
        S("important_topics","Important Topics",important_topics.build_report(ctx["importance"],years)),
        S("future_prediction","Future Exam Prediction",prediction_report.build_prediction_report(ctx["prediction"])),
        S("summary","Summary",prediction_report.build_summary(ctx)),
    ]


def optional_search(ctx):
    print("\nOptional semantic search over the question bank (press Enter to skip).")
    while True:
        query=input("Search by meaning: ").strip()
        if not query:
            return
        for idx,score in semantic_search(query,ctx["embedder"],ctx["emb"],ctx["df"]):
            row=ctx["df"].iloc[idx]
            print(f"  {score:.2f}  {row['year']} Q{row['qid']}: {helpers.truncate(row['text'],100)}")


def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    print("ExamLens - Exam Paper Intelligence System\n")
    try:
        syllabus,topics=load_syllabus()
        count=ask_positive_int("Enter the number of previous-year question papers: ")
        papers,issues=load_papers(count)
        if not papers:
            print("\nNo usable question papers were loaded, so nothing can be analysed.")
            for i in issues:
                print("  - "+i)
            return
        if len(papers)<3:
            print(f"\nWarning: only {len(papers)} usable paper(s). Trends and future prediction will have limited reliability.")
        ctx=run_pipeline(syllabus,topics,papers,count,issues)
        sections=build_sections(ctx)
        full_report=report.render(sections)
        path=report.save_report(full_report)
        print(report.render(sections,hidden=config.CONSOLE_HIDDEN_SECTIONS))
        print(f"The complete report was saved to: {path}")
        optional_search(ctx)
    except (KeyboardInterrupt,EOFError):
        print("\nExamLens stopped.")


if __name__=="__main__":
    main()
