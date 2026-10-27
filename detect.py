import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

# Thread is only required on notes where it is clearly visible in photos
THREAD_REQUIRED = {100, 200, 500}
MIN_CONF = {"security_thread": 0.25, "watermark": 0.40, "serial_number": 0.40}
BLUR_LIMIT = 40  # lower = blurrier; tune this on your fake test images


def detect(image_path, denomination=None):
    img = cv2.imread(image_path)
    if img is None:
        return {"status": "Error", "confidence": 0.0,
                "flagged_features": ["could not read image"]}

    result = model(img, conf=0.15, verbose=False)[0]

    best = {}  # best box per feature
    for box in result.boxes:
        name = model.names[int(box.cls)]
        conf = float(box.conf)
        if name not in best or conf > best[name][0]:
            best[name] = (conf, [int(v) for v in box.xyxy[0]])

    required = ["watermark", "serial_number"]
    if denomination in THREAD_REQUIRED:
        required.append("security_thread")

    flagged = []
    for feat in required:
        if feat not in best:
            flagged.append(f"missing {feat}")
        elif best[feat][0] < MIN_CONF[feat]:
            flagged.append(f"weak {feat}")

    # Blur check on the serial number area (catches degraded serials)
    if "serial_number" in best:
        x1, y1, x2, y2 = best["serial_number"][1]
        crop = cv2.cvtColor(img[max(y1, 0):y2, max(x1, 0):x2], cv2.COLOR_BGR2GRAY)
        if crop.size and cv2.Laplacian(crop, cv2.CV_64F).var() < BLUR_LIMIT:
            flagged.append("blurred serial number")

    found = [best[f][0] for f in required if f in best]
    confidence = round(sum(found) / len(found), 2) if found else 0.0

    return {
        "status": "Fake" if flagged else "Genuine",
        "confidence": confidence,
        "flagged_features": flagged,
    }


if __name__ == "__main__":
    import sys
    print(detect(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None))