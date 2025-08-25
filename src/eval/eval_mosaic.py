# src/eval/eval_mosaic.py
from __future__ import annotations
import argparse, statistics as st
from .metrics import load_boxes, greedy_counts, pr_curve_and_ap50

def eval_pair(pred_path: str, gt_path: str, iou: float):
    preds = load_boxes(pred_path)
    gts   = load_boxes(gt_path)
    TP, FP, FN = greedy_counts(preds, gts, iou)
    P = TP / max(TP+FP, 1); R = TP / max(TP+FN, 1)
    AP, _, _, best = pr_curve_and_ap50(preds, gts, iou)
    return {
        "pred": pred_path, "gt": gt_path,
        "TP":TP, "FP":FP, "FN":FN,
        "precision":P, "recall":R, "F1":best["F1"],
        "precision@bestF1":best["precision"], "recall@bestF1":best["recall"],
        "AP50":AP
    }

def evaluate_mosaic(pairs: str, iou: float):
    with open(pairs, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    rows = []
    for ln in lines:
        pred, gt = [s.strip() for s in ln.split(";")]
        rows.append(eval_pair(pred, gt, iou))

    for r in rows:
        print(f"{r['pred']} vs {r['gt']}: TP={r['TP']} FP={r['FP']} FN={r['FN']} | "
              f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['F1']:.3f} | AP@{iou:.2f}={r['AP50']:.3f}")

    if rows:
        def avg(key): return st.mean([r[key] for r in rows])
        print("\n== AVERAGE over all ==")
        print(f"P={avg('precision'):.3f} R={avg('recall'):.3f} F1={avg('F1'):.3f} AP@{iou:.2f}={avg('AP50'):.3f}")
    return None