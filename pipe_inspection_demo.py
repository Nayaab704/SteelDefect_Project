import cv2
import numpy as np

# ---------------------------
# Simulated ERP/SAP lookup
# ---------------------------
FAKE_DB = {
    "PIPE12345": {"grade": "X52", "heat": "H9001", "length_m": 12.0},
    "PIPE67890": {"grade": "X60", "heat": "H9002", "length_m": 10.0},
}

def fetch_details(pipe_id: str):
    # In real life: SAP/ERP API call
    return FAKE_DB.get(pipe_id, {"note": "Not found in demo DB"})

# ---------------------------
# QR decode using OpenCV
# ---------------------------
def decode_qr(image_bgr):
    detector = cv2.QRCodeDetector()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    data, points, _ = detector.detectAndDecode(gray)
    return data, points

# ---------------------------
# Defect candidate detection (classical CV baseline)
# Output: bounding boxes (x, y, w, h)
# ---------------------------
def detect_defects(image_bgr):
    resized = cv2.resize(image_bgr, (512, 512))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)

    edges = cv2.Canny(enhanced, 30, 120)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:   # tune this
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))

    return resized, enhanced, bboxes

# ---------------------------
# "Spray instruction" mock
# ---------------------------
def spray_instruction(bboxes):
    # In a real system this would go to PLC/actuator controller
    # Here we just print the areas to mark.
    if not bboxes:
        return "No defects detected → No spray"
    return f"Spray mark on {len(bboxes)} region(s): {bboxes[:3]}{'...' if len(bboxes)>3 else ''}"

def main():
    # For demo: single image.
    # In production: camera stream or folder watcher.
    img = cv2.imread("input.jpg")
    if img is None:
        print("input.jpg not found.")
        return

    # 1) Decode barcode / QR (use barcode.jpg for dedicated QR image)
    # If your input.jpg doesn't include QR, you can also read barcode.jpg here.
    barcode_img = cv2.imread("barcode.jpg")
    if barcode_img is None:
        print("barcode.jpg not found (QR image).")
        return

    pipe_id, points = decode_qr(barcode_img)
    if not pipe_id:
        print("QR not detected in barcode.jpg")
        return

    details = fetch_details(pipe_id)
    print("✅ Pipe ID:", pipe_id)
    print("✅ Pipe details:", details)

    # 2) Defect detection on surface image
    resized, enhanced, bboxes = detect_defects(img)

    # 3) Draw boxes
    output = resized.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 4) Mock spray instruction
    instruction = spray_instruction(bboxes)
    print("🟦 Actuator instruction:", instruction)

    # Save outputs
    cv2.imwrite("pipeline_enhanced.jpg", enhanced)
    cv2.imwrite("pipeline_bboxes.jpg", output)
    print("✅ Saved: pipeline_enhanced.jpg, pipeline_bboxes.jpg")

    cv2.imshow("Enhanced", enhanced)
    cv2.imshow("Defect Regions (BBoxes)", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
