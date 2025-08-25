# src/eval/metrics.py
from __future__ import annotations
from typing import List, Dict, Tuple
import json
import math

def iou(a: Dict, b: Dict) -> float:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    w, h = max(0.0, xx2-xx1), max(0.0, yy2-yy1)
    inter = w*h
    area_a = max(0.0,(ax2-ax1))*max(0.0,(ay2-ay1))
    area_b = max(0.0,(bx2-bx1))*max(0.0,(by2-by1))
    denom = area_a + area_b - inter
    return inter/denom if denom>0 else 0.0

def load_boxes(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for d in data:
        out.append({
            "x1": float(d["x1"]), "y1": float(d["y1"]),
            "x2": float(d["x2"]), "y2": float(d["y2"]),
            "conf": float(d.get("conf", 1.0))
        })
    return out

def greedy_counts(preds: List[Dict], gts: List[Dict], iou_thr: float=0.5) -> Tuple[int,int,int]:
    used = set()
    TP = 0
    # sort by conf desc
    order = sorted(range(len(preds)), key=lambda i: preds[i].get("conf",1.0), reverse=True)
    for idx in order:
        p = preds[idx]
        best = -1; best_iou = 0.0
        for j, g in enumerate(gts):
            if j in used: continue
            I = iou(p, g)
            if I >= iou_thr and I > best_iou:
                best_iou = I; best = j
        if best >= 0:
            used.add(best); TP += 1
    FP = len(preds) - TP
    FN = len(gts) - TP
    return TP, FP, FN

def pr_curve_and_ap50(preds: List[Dict], gts: List[Dict], iou_thr: float=0.5):
    preds_sorted = sorted(preds, key=lambda d: d.get("conf", 1.0), reverse=True)
    used = set()
    tp_flags = []
    for p in preds_sorted:
        best = -1; best_iou = 0.0
        for j, g in enumerate(gts):
            if j in used: continue
            I = iou(p, g)
            if I >= iou_thr and I > best_iou:
                best_iou = I; best = j
        if best >= 0:
            used.add(best); tp_flags.append(1)
        else:
            tp_flags.append(0)

    tp_c, fp_c = [], []
    tp = fp = 0
    for f in tp_flags:
        if f: tp += 1
        else: fp += 1
        tp_c.append(tp); fp_c.append(fp)

    npos = max(1, len(gts))
    precisions, recalls = [], []
    for t in range(len(tp_c)):
        P = tp_c[t] / max(tp_c[t] + fp_c[t], 1)
        R = tp_c[t] / npos
        precisions.append(P); recalls.append(R)

    # interpolated AP
    if not recalls:
        ap = 0.0
    else:
        R = [0.0] + recalls + [1.0]
        P = [1.0] + precisions + [precisions[-1] if precisions else 0.0]
        for i in range(len(P)-2, -1, -1):
            P[i] = max(P[i], P[i+1])
        ap = 0.0
        for i in range(len(R)-1):
            ap += (R[i+1]-R[i]) * P[i+1]

    # best F1 operating point (koristno za tabelo)
    best = {"F1":0,"precision":0,"recall":0,"idx":None}
    for t in range(len(tp_c)):
        TP = tp_c[t]; FP = fp_c[t]; FN = len(gts) - TP
        P = TP / max(TP+FP, 1); R = TP / max(TP+FN, 1)
        F1 = 2*P*R / max(P+R, 1e-9)
        if F1 > best["F1"]:
            best = {"F1":F1, "precision":P, "recall":R, "idx":t}

    return ap, precisions, recalls, best
