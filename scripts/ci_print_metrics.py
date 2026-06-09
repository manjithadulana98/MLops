"""Print a summary of evaluation metrics from the evaluation report JSON."""
import json
import sys

try:
    with open("reports/evaluation_report.json") as f:
        r = json.load(f)
    b = r["binary_metrics"]
    o = r.get("optimal_threshold_metrics", {})
    print(f"Sensitivity:  {b['sensitivity']:.4f}")
    print(f"Specificity:  {b['specificity']:.4f}")
    print(f"Precision:    {b['precision']:.4f}")
    print(f"F1-Score:     {b['f1_score']:.4f}")
    print(f"ROC AUC:      {r['roc_metrics']['roc_auc']:.4f}")
    if o:
        print(
            f"Best threshold F1: {o['threshold']:.3f} "
            f"-> Sensitivity {o['sensitivity']:.4f}, F1 {o['f1_score']:.4f}"
        )
except Exception as e:
    print(f"Could not read report: {e}")
    sys.exit(0)
