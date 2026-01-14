import cv2

# Simulated “database / SAP lookup”
FAKE_DB = {
    "PIPE12345": {"grade": "X52", "heat": "H9001", "length_m": 12.0},
    "PIPE67890": {"grade": "X60", "heat": "H9002", "length_m": 10.0},
}

def fetch_details(pipe_id: str):
    # In real life this would be SAP/ERP API call
    return FAKE_DB.get(pipe_id, {"note": "Not found in demo DB"})

img = cv2.imread("barcode.jpg")
if img is None:
    print("❌ barcode.jpg not found. Put it in the same folder.")
    raise SystemExit

detector = cv2.QRCodeDetector()

data, points, _ = detector.detectAndDecode(img)

if not data:
    print("❌ No QR code detected. Try a clearer QR image.")
    raise SystemExit

print("✅ Scanned ID:", data)
details = fetch_details(data)
print("✅ Fetched details:", details)

# Draw bounding box around QR code if detected
if points is not None:
    pts = points.astype(int).reshape(-1, 2)
    for i in range(len(pts)):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)
    cv2.putText(img, data, (pts[0][0], pts[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imwrite("barcode_result.jpg", img)
print("✅ Saved annotated image: barcode_result.jpg")
