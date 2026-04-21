import cv2
import sys

# Check if image path is provided
if len(sys.argv) < 2:
    print("Usage: python edge_detection.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

# Load image
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image. Check file path.")
    sys.exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection
edges1 = cv2.Canny(gray, 50, 150)
edges2 = cv2.Canny(gray, 100, 200)

# Show results
cv2.imshow("Original", image)
cv2.imshow("Edges (50,150)", edges1)
cv2.imshow("Edges (100,200)", edges2)

cv2.waitKey(0)
cv2.destroyAllWindows()