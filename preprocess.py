import cv2
import numpy as np

image = cv2.imread("input.jpg")
if image is None:
    print("❌ input.jpg not found.")
    exit()

resized = cv2.resize(image, (512, 512))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# TUNED VALUES 
blur = cv2.GaussianBlur(gray, (3, 3), 0)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blur)

# Creating a defect/suspicious-area mask
# 1) Detect edges (scratches show up as strong edges)
edges = cv2.Canny(enhanced, 50, 150)

# 2) Thicken edges so defects look more like regions
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# 3) Find contours (connected regions)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4) Draw bounding boxes around large-enough regions
output = resized.copy()
mask = np.zeros_like(gray)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 150:  # filter small noise (tune this)
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Fill mask for this region
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

# Save all outputs
cv2.imwrite("01_resized.jpg", resized)
cv2.imwrite("02_gray.jpg", gray)
cv2.imwrite("03_enhanced.jpg", enhanced)
cv2.imwrite("04_edges.jpg", edges)
cv2.imwrite("05_dilated.jpg", dilated)
cv2.imwrite("06_mask.jpg", mask)
cv2.imwrite("07_bboxes.jpg", output)

print("✅ Saved: 04_edges.jpg, 05_dilated.jpg, 06_mask.jpg, 07_bboxes.jpg")

cv2.imshow("Enhanced", enhanced)
cv2.imshow("Edges", edges)
cv2.imshow("Mask", mask)
cv2.imshow("Bounding Boxes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
