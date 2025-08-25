from __future__ import annotations
import json, os
import cv2

def annotate_mosaic(mosaic_path: str, out_json: str, display_max=(1920, 1080)) -> str:
    """
    Two-click annotator with live preview.
    LMB 1st click = anchor; move mouse to size; LMB 2nd click = finalize box
    RMB = cancel current box; Backspace = undo last; Esc = save & exit.
    Auto-scales the display so big mosaics fit on screen; boxes saved in original coords.
    """
    img = cv2.imread(mosaic_path)
    if img is None:
        raise FileNotFoundError(mosaic_path)

    H, W = img.shape[:2]
    maxW, maxH = display_max
    scale = min(maxW / W, maxH / H, 1.0)

    def to_disp(pt):
        x, y = pt
        return (int(round(x * scale)), int(round(y * scale)))

    def from_disp(x, y):
        return (x / max(scale, 1e-9), y / max(scale, 1e-9))

    # Precompute a display-size image (for speed)
    disp_base = cv2.resize(img, (int(round(W * scale)), int(round(H * scale)))) if scale < 1.0 else img.copy()

    boxes = []              # list of [x1,y1,x2,y2] in ORIGINAL coords
    anchor = None           # first corner in ORIGINAL coords
    mouse_xy = None         # current mouse in ORIGINAL coords

    win = "GT annotate (LMB twice=box, RMB=cancel, Backspace=undo, Esc=save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_base.shape[1], disp_base.shape[0])

    def on_mouse(event, x, y, flags, param):
        nonlocal anchor, mouse_xy
        # current mouse in original coords (even if window is scaled)
        ox, oy = from_disp(x, y)
        mouse_xy = (ox, oy)

        if event == cv2.EVENT_LBUTTONDOWN:
            if anchor is None:
                anchor = (ox, oy)
            else:
                x1, y1 = anchor
                x2, y2 = ox, oy
                # normalize corners
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                anchor = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # cancel current in-progress box
            anchor = None

    cv2.setMouseCallback(win, on_mouse)

    # thickness/radius scaled so it stays visible on huge images
    thick = max(2, int(round(2 * scale)))
    dot_r = max(4, int(round(6 * scale)))

    while True:
        vis = disp_base.copy()

        # draw finalized boxes (green)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(vis, to_disp((x1, y1)), to_disp((x2, y2)), (0, 255, 0), thick)

        # draw live preview (if anchor set)
        if anchor is not None and mouse_xy is not None:
            ax, ay = anchor
            mx, my = mouse_xy
            cv2.rectangle(vis, to_disp((ax, ay)), to_disp((mx, my)), (0, 255, 255), thick)  # yellow preview
            cv2.circle(vis, to_disp((ax, ay)), dot_r, (0, 0, 255), -1)                      # red anchor dot

        # HUD: show count
        hud = f"boxes: {len(boxes)}"
        cv2.putText(vis, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 50), 2, cv2.LINE_AA)

        cv2.imshow(win, vis)
        k = cv2.waitKey(16) & 0xFF

        if k == 27:          # Esc -> save & exit
            break
        elif k == 8:         # Backspace -> undo last
            if boxes: boxes.pop()

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([{"x1":b[0],"y1":b[1],"x2":b[2],"y2":b[3]} for b in boxes], f, indent=2)
    cv2.destroyAllWindows()
    return out_json
