# src/eval/eval_mosaic.py
from __future__ import annotations
import statistics as st
from .metrics import load_boxes, greedy_counts, pr_curve_and_ap50

def eval_pair(pred_path: str, gt_path: str, iou: float):
    preds = load_boxes(pred_path)
    gts   = load_boxes(gt_path)

    TP, FP, FN = greedy_counts(preds, gts, iou)
    P = TP / max(TP + FP, 1)
    R = TP / max(TP + FN, 1)
    AP, _, _, best = pr_curve_and_ap50(preds, gts, iou)

    # counting diagnostics tied to this eval (so via TP/FP/FN)
    cnt_pred = TP + FP
    cnt_gt   = TP + FN
    diff     = cnt_pred - cnt_gt
    rel_err  = diff / max(cnt_gt, 1)

    return {
        "pred": pred_path, "gt": gt_path,
        "TP": TP, "FP": FP, "FN": FN,
        "precision": P, "recall": R, "F1": best["F1"],
        "precision@bestF1": best["precision"], "recall@bestF1": best["recall"],
        "AP50": AP,
        "cnt_pred": cnt_pred, "cnt_gt": cnt_gt, "diff": diff, "rel_err": rel_err,
    }

def _print_aggregates(tag: str, rows: list[dict], iou: float) -> None:
    """Pretty aggregate print for a subset (macro + micro + count summaries)."""
    if not rows:
        print(f"\n== {tag} ==\n(no rows)")
        return

    def avg(key): return st.mean([r[key] for r in rows])

    TP_sum = sum(r["TP"] for r in rows)
    FP_sum = sum(r["FP"] for r in rows)
    FN_sum = sum(r["FN"] for r in rows)
    pred_sum = sum(r["cnt_pred"] for r in rows)
    gt_sum   = sum(r["cnt_gt"] for r in rows)
    diff_sum = pred_sum - gt_sum
    rel_err_micro = diff_sum / max(gt_sum, 1)

    P_micro = TP_sum / max(TP_sum + FP_sum, 1)
    R_micro = TP_sum / max(TP_sum + FN_sum, 1)
    F1_micro = (2 * P_micro * R_micro / max(P_micro + R_micro, 1e-9)) if (P_micro or R_micro) else 0.0

    print(f"\n== {tag} ==")
    print(f"macro: P={avg('precision'):.3f} R={avg('recall'):.3f} F1={avg('F1'):.3f} AP@{iou:.2f}={avg('AP50'):.3f}")
    print(f"counts: #pred(avg)={avg('cnt_pred'):.2f} #gt(avg)={avg('cnt_gt'):.2f} "
          f"diff(avg)={avg('diff'):.2f} rel_err(avg)={avg('rel_err'):.2%}")
    print(f"micro:  P={P_micro:.3f} R={R_micro:.3f} F1={F1_micro:.3f}  |  "
          f"#pred={pred_sum} #gt={gt_sum} diff={diff_sum} rel_err={rel_err_micro:.2%}")

def evaluate_mosaic(pairs: str, iou: float):
    with open(pairs, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    rows: list[dict] = []
    rows_p1: list[dict] = []
    rows_p2: list[dict] = []

    for idx, ln in enumerate(lines, start=1):
        parts = [s.strip() for s in ln.split(";")]
        if len(parts) < 2:
            print(f"[WARN] skipping malformed line {idx}: {ln!r}")
            continue
        pred, gt = parts[0], parts[1]
        r = eval_pair(pred, gt, iou)
        # tag pipeline by parity: odd -> p1, even -> p2
        pipe = "p1" if (idx % 2 == 1) else "p2"
        r["pipeline"] = pipe
        rows.append(r)
        (rows_p1 if pipe == "p1" else rows_p2).append(r)

        print(f"[{pipe}] {r['pred']} vs {r['gt']}: "
              f"TP={r['TP']} FP={r['FP']} FN={r['FN']} | "
              f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['F1']:.3f} | "
              f"AP@{iou:.2f}={r['AP50']:.3f} | "
              f"#pred={r['cnt_pred']} #gt={r['cnt_gt']} diff={r['diff']} rel_err={r['rel_err']:.2%}")

    # Overall (all rows), then per-pipeline
    if rows:
        _print_aggregates("AVERAGE over all", rows, iou)
        _print_aggregates("PIPELINE p1 (odd lines)", rows_p1, iou)
        _print_aggregates("PIPELINE p2 (even lines)", rows_p2, iou)

    return rows
