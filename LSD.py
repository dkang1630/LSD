import cv2
import numpy as np
import os
from pylsd.lsd import lsd

def find_power_line(lines):
    # Your logic to identify the power line among multiple line segments
    # For simplicity, let's assume it's the longest line segment
    lengths = np.sqrt(np.sum((lines[:, :2] - lines[:, 2:4]) ** 2, axis=1))
    index_of_longest_line = np.argmax(lengths)
    return index_of_longest_line

fullName = 'car.jpg'
folder, imgName = os.path.split(fullName)

# Read the image
src = cv2.imread(fullName, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# Detect line segments using LSD
lines = lsd(gray)

# Find the index of the power line
power_line_index = find_power_line(lines)

# Draw the power line
pt1 = (int(lines[power_line_index, 0]), int(lines[power_line_index, 1]))
pt2 = (int(lines[power_line_index, 2]), int(lines[power_line_index, 3]))
width = lines[power_line_index, 4]
cv2.line(src, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))

# Calculate bounding box coordinates
x, y, w, h = cv2.boundingRect(np.array([pt1, pt2]))

# Draw the bounding box around the power line
cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save the result
output_path = os.path.join(folder, 'cv2_power_line_bbox_' + imgName.split('.')[0] + '.jpg')
cv2.imwrite(output_path, src)
print(f"Result saved to: {output_path}")
