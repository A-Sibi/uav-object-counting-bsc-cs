# src/eval/plots.py
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from pathlib import Path

def save_pr_curve(recalls, precisions, ap: float, best: dict, out_path: str, title: str | None = None) -> str:
    """
    Shrani PR krivuljo (Recall vs Precision). 'best' pričakujemo z 'precision','recall','F1'.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4), dpi=150)
    # step krivulja (post) je standard za PR
    plt.step(recalls, precisions, where="post")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    if best and "recall" in best and "precision" in best:
        plt.scatter([best["recall"]], [best["precision"]], s=30)
        plt.annotate(f"F1={best.get('F1', 0):.3f}", (best["recall"], best["precision"]),
                     xytext=(5, -10), textcoords="offset points", fontsize=8)
    t = title or f"PR curve (AP={ap:.3f})"
    plt.title(t)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
