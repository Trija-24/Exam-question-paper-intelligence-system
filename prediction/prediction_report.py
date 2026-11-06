"""Future prediction report and dynamic summary."""
import config
from utils import helpers


def build_prediction_report(pred):
    out=[f"Target: probability that a topic appears in the {pred['next_label']}. "
         "Exact question wording is never predicted.",""]
    if pred["folds"]:
        out.append("Accuracy on past years (trained on earlier years, tested on the next year)")
        rows=[[f"{f['train_years'][0]}-{f['train_years'][-1]}",f["test_year"],f["n_train"],f["n_test"],
               f"{f['accuracy']:.2f}",f"{f['precision']:.2f}",f"{f['recall']:.2f}",f"{f['f1']:.2f}",
               "n/a" if f["auc"] is None else f"{f['auc']:.2f}",f"{f['baseline']:.2f}"] for f in pred["folds"]]
        out.append(helpers.format_table(["Train years","Test","TrainRows","TestRows","Acc","Prec","Recall","F1","ROC-AUC","Acc(last-yr rule)"],rows))
        out.append("Acc(last-yr rule) = a naive baseline that predicts 'appears again if it appeared last year'.")
    for y,reason in pred["skipped"]:
        out.append(f"  Test year {y} skipped: {reason}")
    for n in pred["notes"]:
        out.append("  Note: "+n)
    if pred["trained"] and pred["reduced_confidence"]:
        out.append("  Confidence: REDUCED (small dataset / few past-year checks). Treat likelihoods as rough guidance.")
    out.append("")
    for rank,p in enumerate(pred["predictions"],1):
        out.append(f"{rank}. Topic: {p['topic']}")
        out.append(f"   Likelihood: {p['likelihood']*100:.0f}%  (high predicted likelihood based on historical patterns)"
                   if p["likelihood"]>=0.6 else f"   Likelihood: {p['likelihood']*100:.0f}%")
        out.append("   Likely concepts: "+(", ".join(p["concepts"]) if p["concepts"] else "none matched in history"))
        if p["qtype"]:
            out.append(f"   Likely question type: {p['qtype']} ({p['qtype_share']:.0f}% of recency-weighted history)")
            out.append(f"   Likely difficulty: {p['difficulty']} (estimated)")
        out.append("   Historical evidence:")
        for e in p["evidence"]:
            out.append("     - "+e)
        out.append("")
    return "\n".join(out).rstrip()


def build_summary(ctx):
    lines=[]
    imp=ctx["importance"]["rows"]
    if imp:
        lines.append("Top important topics: "+", ".join(r["topic"] for r in imp[:3])+".")
    dist=ctx["qtype_dist"]
    top_type=max(dist,key=dist.get)
    lines.append(f"Most common question type: {top_type} ({dist[top_type]:.0f}%).")
    if len(ctx["years"])<config.TREND_MIN_YEARS:
        lines.append("Topic trends: not enough years to judge direction.")
    else:
        pg=ctx["topic_trend"]["groups"]
        lines.append("Increasing topics: "+(", ".join(pg["increasing"][:4]) if pg["increasing"] else "none detected")
                     +"; decreasing topics: "+(", ".join(pg["decreasing"][:4]) if pg["decreasing"] else "none detected")+".")
    recs=sorted([r for rs in ctx["concepts"].values() for r in rs],key=lambda r:(-len(r["years"]),-r["freq"]))
    recur=[r for r in recs if len(r["years"])>=2][:4]
    lines.append("Major recurring concepts: "+(", ".join(f"{r['concept']} ({r['topic']})" for r in recur) if recur
                 else "none appeared in more than one year")+".")
    pairs=ctx["pairs"]
    high=sum(1 for p in pairs if p["level"]=="Highly similar")
    lines.append(f"Repeated/rephrased questions: {len(pairs)} pair(s) across years ({high} highly similar).")
    preds=ctx["prediction"]["predictions"][:3]
    if preds:
        lines.append("Likely future topics: "+", ".join(f"{p['topic']} ({p['likelihood']*100:.0f}%)" for p in preds)+".")
    dt=ctx["difficulty_res"]["trend"]
    if len(ctx["years"])>=config.TREND_MIN_YEARS:
        lines.append(f"Estimated difficulty trend: Hard questions {dt['Hard']}, Easy questions {dt['Easy']}.")
    else:
        lines.append("Estimated difficulty trend: not enough years to judge direction.")
    out=["- "+l for l in lines]
    out.append("")
    out.append("Predictions are probabilistic estimates from historical patterns, not guarantees.")
    return "\n".join(out)
