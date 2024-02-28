import cv2
import numpy as np
from pathlib import Path
import argparse
import os

# Global variables
window_name = 'Image'
img_original = None
img_display = None
points = []
auto_mark = True
pause = False

# Calculate where the line intersects with image boundaries.
def extend_line(x1, y1, x2, y2, width, height):
    if x2 == x1:  # Vertical line
        return [(x1, 0), (x1, height)]
    elif y2 == y1:  # Horizontal line
        return [(0, y1), (width, y1)]
    else:
        m = (y2 - y1) / (x2 - x1)  # Slope
        b = y1 - m * x1  # Intercept

        # all intersection points
        points = []

        # Intersection with top and bottom
        y_top = 0
        x_top = (y_top - b) / m
        y_bottom = height
        x_bottom = (y_bottom - b) / m

        if 0 <= x_top <= width:
            points.append((int(x_top), y_top))
        if 0 <= x_bottom <= width:
            points.append((int(x_bottom), y_bottom))

        # Intersection with the left and right sides
        y_left = m * 0 + b
        y_right = m * width + b

        if 0 <= y_left <= height:
            points.append((0, int(y_left)))
        if 0 <= y_right <= height:
            points.append((width, int(y_right)))

        if len(points) > 2:
            dist = lambda p1, p2: (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
            p1, p2 = max([(p1, p2) for p1 in points for p2 in points if p1 != p2], key=lambda pair: dist(*pair))
            return [p1, p2]
        return points


# Find intersection of two lines within angle range(87, 92), return [(x1, y1), (x2, [y2)]
def find_intersection(line1, line2, angle = (87, 93)):
    x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
    x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]

    # slopes
    if x2 - x1 == 0:  # Vertical line1
        m1 = float('inf')
    else:
        m1 = (y2 - y1) / (x2 - x1)
    
    if x4 - x3 == 0:  # Vertical line2
        m2 = float('inf')
    else:
        m2 = (y4 - y3) / (x4 - x3)
    
    # Check if lines within angle range
    if m1 * m2 != -1: 
        angle_rad = np.abs(np.arctan((m2 - m1) / (1 + m1 * m2)))
        angle_deg = np.degrees(angle_rad)
        if not (angle[0] <= angle_deg <= angle[1]):
            return None
    
    # find intersection
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None

    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    # Check if intersection point is within line
    if (min(x1, x2) <= intersect_x <= max(x1, x2) and
        min(y1, y2) <= intersect_y <= max(y1, y2) and
        min(x3, x4) <= intersect_x <= max(x3, x4) and
        min(y3, y4) <= intersect_y <= max(y3, y4)):
        return (intersect_x, intersect_y)
    else:
        return None

# Euclidean distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Find point in points that closest to image center
def find_closest_point_to_center(points, center):
    return min(points, key=lambda point: distance(point, center))

def find(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.bilateralFilter(image,9,75,75)
    if image is None:
        print("Error: Image not found")
        return

    height, width = image.shape[:2]
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(image)[0]  # Detect lines
    image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    min_line_length = 125
    extended_lines = []

    # Extend lines and draw on image_with_lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length > min_line_length:
                extended_points = extend_line(x1, y1, x2, y2, width, height)
                if len(extended_points) == 2:
                    extended_lines.append(extended_points)
                    cv2.line(image_with_lines, extended_points[0], extended_points[1], (0, 255, 0), 2)

    # Find intersections of extended lines
    intersections = []
    for i, line1 in enumerate(extended_lines):
        for line2 in extended_lines[i+1:]:
            intersect = find_intersection(line1, line2)
            if intersect:
                intersections.append(intersect)

    # Group and select closest points to the center from each group
    image_center = (width / 2, height / 2)
    grouped_points = []
    visited = set()

    for i, point1 in enumerate(intersections):
        if i in visited:
            continue
        close_points = [point1]
        visited.add(i)
        for j, point2 in enumerate(intersections[i+1:], start=i+1):
            if distance(point1, point2) < 500:  # Grouping threshold
                close_points.append(point2)
                visited.add(j)
        grouped_points.append(close_points)

    closest_points_to_center = [find_closest_point_to_center(group, image_center) for group in grouped_points]

    # # Draw intersection points on image
    # for point in closest_points_to_center:
    #     x, y = int(point[0]), int(point[1])
    #     cv2.circle(image_with_lines, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    return closest_points_to_center

# Resize image to fit on display
def resize_image_to_fit_screen(image, max_height=600):
    height, _ = image.shape[:2]
    if height > max_height:
        scaling_factor = max_height / float(height)
        return cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return image

def redraw_display_image():
    global img_display, img_original, points
    img_display = resize_image_to_fit_screen(img_original.copy())
    for point in points:
        height, width = img_original.shape[:2]
        display_height, display_width = img_display.shape[:2]
        x_display = int(point[0] * (display_width / width))
        y_display = int(point[1] * (display_height / height))
        cv2.circle(img_display, (x_display, y_display), 5, (0, 0, 255), -1)

def remove_closest_point(x, y):
    global points, img_original, img_display
    if not points:
        return

    # Calculate the corresponding point on the original image for comparison
    height, width = img_original.shape[:2]
    display_height, display_width = img_display.shape[:2]
    x_original = int(x * (width / display_width))
    y_original = int(y * (height / display_height))

    # Find the point closest to the clicked location
    closest_point = min(points, key=lambda point: (point[0] - x_original)**2 + (point[1] - y_original)**2)
    points.remove(closest_point)

def draw_polygon_and_blackout_outside(event, x, y, flags, param):
    global points, img_original, img_display

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        # Calculate the corresponding point on the original image
        height, width = img_original.shape[:2]
        display_height, display_width = img_display.shape[:2]
        x_original = int(x * (width / display_width))
        y_original = int(y * (height / display_height))

        # Add point and show it on the display image
        points.append((x_original, y_original))
        redraw_display_image()
        cv2.imshow(window_name, img_display)

        if len(points) == 4:
            # Create mask and blackout everything outside the polygon
            process_and_display_image()

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove closest point and refresh display image when right click
        remove_closest_point(x, y)
        redraw_display_image()
        cv2.imshow(window_name, img_display)

    elif event == cv2.EVENT_RBUTTONDBLCLK:
        # Remove all point and refresh display image when right double click
        points = []
        redraw_display_image()
        cv2.imshow(window_name, img_display)

def order_points(pts):
    centroid = np.mean(pts, axis=0)

    sorted_pts = sorted(pts, key=lambda point: np.arctan2(point[1] - centroid[1], point[0] - centroid[0]))
    top = sorted(sorted_pts[:2], key=lambda x: x[0])
    bottom = sorted(sorted_pts[2:], key=lambda x: x[0], reverse=True)
    
    return [top[0], top[1], bottom[0], bottom[1]]

def process_and_display_image():
    global img_original, points
    points = order_points(points)
    mask = np.zeros(img_original.shape[:2], dtype=np.uint8)
    if len(points) >= 3:
        roi_corners = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, 255)  # Fill the polygon

        mask_inv = cv2.bitwise_not(mask)  # Invert mask to blackout outside
        img_blackout_outside = img_original.copy()
        img_blackout_outside[mask_inv == 255] = (0, 0, 0)

        # Resize the result to fit on screen and display
        img_result_display = resize_image_to_fit_screen(img_blackout_outside)
        cv2.imshow(window_name, img_result_display)
    else:
        redraw_display_image()  # Redraw with remaining points if not enough for a polygon

def crop_and_save_image(image_path):
    global img_original, points
    if len(points) == 4:
        points = order_points(points)

        width = int(max(distance(points[0], points[1]), distance(points[2], points[3])))
        height = int(max(distance(points[0], points[3]), distance(points[1], points[2])))

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')

        M = cv2.getPerspectiveTransform(np.array(points, dtype='float32'), dst)
        result = cv2.warpPerspective(img_original, M, (width, height))

        # Save the result
        img_path = Path(image_path)
        save_path = img_path.parent / f"cropped_{img_path.name}"
        cv2.imwrite(str(save_path), result)
        print(f"Image cropped and saved as '{str(save_path)}'")

def main(image_path):
    global img_original, img_display, points, auto_mark, pause
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"Error: Unable to open {image_path}")
        return

    # Obtain initial points from find(image_path)
    if auto_mark:
        initial_points = find(image_path)
        points = list(initial_points)

    img_display = resize_image_to_fit_screen(img_original)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon_and_blackout_outside)

    # Draw initial points and allow for adjustments
    redraw_display_image()
    cv2.imshow(window_name, img_display)
    
    while(1):
        cv2.imshow(window_name, img_display)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            crop_and_save_image(image_path)
            break
        elif k == ord('m'):
            auto_mark = not auto_mark
            if auto_mark:
                initial_points = find(image_path)
                points = list(initial_points)
            else:
                points = []
            redraw_display_image()
            print(f"  auto mark: {auto_mark}")
        elif k == ord('p'):
            pause = True
            break
        elif k == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\n    Welcome to use this cropping tool.")
    print("    Press left button to mark points, press right button to remove cloest point, double press right button to remove all points.")
    print("    Press 'esc' to skip one image, press 'm' to toggle auto mark function, press 'p' to pause and quit")
    print("    Press 's' to save cropped image\n")
    while True:
        image_path = input('Please enter input folder: ')
        if os.path.isdir(image_path):
            print(f"Valid folder provided: {image_path}")
            break
        else:
            print("The provided path is not a valid folder. Please try again.")
    image_path = str(image_path)
    input_folder_path = Path(image_path)
    image_files = list(input_folder_path.rglob('*.png')) + \
                    list(input_folder_path.rglob('*.jpg')) + \
                    list(input_folder_path.rglob('*.jpeg'))

    for img_path in image_files:
        if pause:
            break
        points = []
        if img_path.stem.startswith("cropped_"):
            continue
        
        cropped_img_path = img_path.parent / f"cropped_{img_path.name}"
        
        if not cropped_img_path.exists():
            print(f"Working on {img_path}")
            main(str(img_path))
        else:
            print(f"Skipped {img_path.name}, as cropped version exists.")
