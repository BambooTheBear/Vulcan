# Imports
import math

import cv2
import numpy as np


def scale(image, max_width, max_height):
    # Get the original dimensions of the image
    original_height, original_width = image.shape[:2]

    # Calculate the scaling factors to fit the image onto the screen
    scale_x = max_width / original_width
    scale_y = max_height / original_height
    scaling_factor = min(scale_x, scale_y)

    # Calculate the new dimensions after scaling
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # Resize the image using the calculated dimensions
    scaled_image = cv2.resize(image, (new_width, new_height))

    return scaled_image


def enhanceImage(image_path):
    # Read the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Noise Reduction using Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (1, 1), 0)

    # Thresholding
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    extremed_image = image

    # Convert gray pixels to black
    for y in range(extremed_image.shape[0]):
        for x in range(extremed_image.shape[1]):
            if 0 < extremed_image[y, x] < 255:
                extremed_image[y, x] = 0

    _, extremed_thresholded_image = cv2.threshold(extremed_image, 0, 255, cv2.THRESH_OTSU)

    return extremed_thresholded_image


def detectRectangles(image):
    # Convert the image to binary using thresholding
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Convert to color image
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    rectangles = []
    # Iterate through the contours and draw rectangles with colored outlines
    for contour in contours:
        # OG: 0.04
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 3 detects arrows? 10 detects plus signs?
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)

            if 30 <= w <= image.shape[1] * 0.9 and 30 <= h <= image.shape[0] * 0.9:
                rectangles.append(((x, y), (x + w, y + h)))
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle

    return color_image, rectangles


def detectArrows(image):
    # Convert the image to binary using thresholding
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Convert to color image
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    triangles = []
    # Iterate through the contours and draw rectangles with colored outlines
    for contour in contours:
        # OG: 0.04
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 3 detects arrows? 10 detects plus signs?
        if len(approx) == 3:
            x, y, w, h = cv2.boundingRect(contour)
            triangles.append(((x, y), (x + w, y + h)))
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw green rectangle

    return color_image, triangles


def detectSimpleArrows(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred_image, 50, 150)

    # Perform Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=5)

    # Extract starting and ending points of detected lines
    line_points = []
    marked_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle_deg = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))  # Calculate angle in degrees

            # Check if the angle falls within the desired ranges
            range_deg = 5
            if not (
                    0 <= angle_deg <= range_deg or 90 - range_deg <= angle_deg <= 90 + range_deg or 180 - range_deg <= angle_deg <= 180 + range_deg or
                    270 - range_deg <= angle_deg <= 290 + range_deg or 360 - range_deg <= angle_deg <= 360):
                line_points.append(((x1, y1), (x2, y2)))
                cv2.line(marked_image, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Draw line in blue
    return marked_image, line_points


def detectLines(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred_image, 50, 150)

    # Perform Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=2)

    # Extract starting and ending points of detected lines
    line_points = []
    marked_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle_deg = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))  # Calculate angle in degrees
            # Check if the angle falls within the desired ranges
            range_deg = 2
            if (
                    0 <= angle_deg <= range_deg or 90 - range_deg <= angle_deg <= 90 + range_deg or 180 - range_deg <= angle_deg <= 180 + range_deg or
                    270 - range_deg <= angle_deg <= 290 + range_deg or 360 - range_deg <= angle_deg <= 360):
                line_points.append(((x1, y1), (x2, y2)))
                cv2.line(marked_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Draw line in blue

    return marked_image, line_points


def filterDoubledArrows(triangles, slopes):
    filteredSlopes = []
    tolerance = 100
    for ((x1, y1), (x2, y2)) in slopes:
        within = False
        for ((mx1, my1), (mx2, my2)) in triangles:
            if (
                    ((mx1 + tolerance < x1 < mx2 - tolerance) or (mx1 - tolerance < x1 < mx2 + tolerance) or (
                            mx1 + tolerance > x1 > mx2 - tolerance) or (mx1 - tolerance > x1 > mx2 + tolerance)) and (
                    (my1 + tolerance < y1 < my2 - tolerance) or (my1 - tolerance < y1 < my2 + tolerance) or (
                    my1 + tolerance > y1 > my2 - tolerance) or (my1 - tolerance > y1 > my2 + tolerance))
            ):
                print(f"({x1} | {y1}) and ({x2} | {y2}) is within ({mx1} | {my1}) and ({mx2} | {my2})")
                within = True
                break
        if not within:
            filteredSlopes.append(((x1, y1), (x2, y2)))
    return filteredSlopes


def paintLines(points, image):
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for ((x1, y1), (x2, y2)) in points:
        cv2.line(colored_image, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Draw line in blue
    return colored_image


def distance_point_to_line_segment(x, y, x1, y1, x2, y2):
    # Calculate the squared length of the line segment
    line_length_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Calculate the parameter t that represents the projection of the point onto the line segment
    t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_squared

    # If t is outside the range [0, 1], the closest point is one of the endpoints
    if t < 0:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    elif t > 1:
        return math.sqrt((x - x2) ** 2 + (y - y2) ** 2)

    # Calculate the coordinates of the closest point on the line
    closest_x = x1 + t * (x2 - x1)
    closest_y = y1 + t * (y2 - y1)

    # Calculate the distance between the point and the closest point on the line
    return math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)


def filterNonArrows(linesA, linesB, tolerance):
    filtered_lines = []

    for lineA in linesA:
        for lineB in linesB:
            dist_start_start = ((lineA[0][0] - lineB[0][0]) ** 2 + (lineA[0][1] - lineB[0][1]) ** 2) ** 0.5
            dist_start_end = ((lineA[0][0] - lineB[1][0]) ** 2 + (lineA[0][1] - lineB[1][1]) ** 2) ** 0.5
            dist_end_start = ((lineA[1][0] - lineB[0][0]) ** 2 + (lineA[1][1] - lineB[0][1]) ** 2) ** 0.5
            dist_end_end = ((lineA[1][0] - lineB[1][0]) ** 2 + (lineA[1][1] - lineB[1][1]) ** 2) ** 0.5

            if (dist_start_start <= tolerance or
                    dist_start_end <= tolerance or
                    dist_end_start <= tolerance or
                    dist_end_end <= tolerance):
                for lineB2 in linesB:
                    if lineB2 != lineB:
                        if distance_point_to_line_segment(lineB[0][0], lineB[0][1], lineB2[0][0], lineB2[0][1],
                                                          lineB2[1][0], lineB2[1][1]) < tolerance and \
                                distance_point_to_line_segment(lineB[1][0], lineB[1][1], lineB2[0][0], lineB2[0][1],
                                                               lineB2[1][0], lineB2[1][1]) < tolerance:
                            filtered_lines.append(lineA)
                            filtered_lines.append(lineB)
                            break  # Once a match is found, no need to check the rest of the linesB

    return filtered_lines


def calculate_distance(rect1, rect2):
    x1_1, y1_1 = rect1[0]
    x2_1, y2_1 = rect1[1]
    x1_2, y1_2 = rect2[0]
    x2_2, y2_2 = rect2[1]

    distance_x = max(0, max(x1_1, x1_2) - min(x2_1, x2_2))
    distance_y = max(0, max(y1_1, y1_2) - min(y2_1, y2_2))

    return distance_x + distance_y


def filterClassSegments(rectangles):
    rectangles.sort(key=lambda rect: rect[1][1],
                    reverse=True)  # Sort rectangles by y2 (bottom y-coordinate) in descending order
    result = []

    while rectangles:
        current_rect = rectangles.pop(0)
        group = [current_rect]

        to_remove = []

        for idx, rect in enumerate(rectangles):
            if calculate_distance(current_rect, rect) <= 100:
                group.append(rect)
                to_remove.append(idx)

        for idx in reversed(to_remove):
            rectangles.pop(idx)

        group.sort(key=lambda rect: rect[1][0],
                   reverse=True)  # Sort group by x2 (right x-coordinate) in descending order
        result.append(group)

    return result


def paintClasses(rectangles, image):
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for rectangleList in rectangles:
        for i, ((x1, y1),(x2, y2)) in enumerate(rectangleList):
            print(i)
            if i == 0:
                cv2.rectangle(colored_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif i == 1:
                cv2.rectangle(colored_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif i==2:
                cv2.rectangle(colored_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return colored_image


# Main Method
if __name__ == '__main__':
    print("Started Vulcan")

    enhancedImage = enhanceImage("./data/custom/simple_2.png")

    # Show Image
    cv2.imshow("Enhanced Image", scale(enhancedImage, 1000, 1000))
    rectangleImage, rectangles = detectRectangles(enhancedImage)
    cv2.imshow("Rectangles", scale(rectangleImage, 1000, 1000))
    triangleImage, triangles = detectArrows(enhancedImage)
    cv2.imshow("Arrows", scale(triangleImage, 1000, 1000))
    slopeLinesImage, slopeLines = detectSimpleArrows(enhancedImage)
    # cv2.imshow("Simple Arrows", scale(slopeLinesImage, 1000, 1000))
    filteredDoubleArrows = filterDoubledArrows(triangles, slopeLines)
    # cv2.imshow("Filtered Arrows",
    #           scale(paintLines(filteredDoubleArrows, enhancedImage), 1000, 1000))
    horizontalVerticalLinesImage, horizontalVerticalLines = detectLines(enhancedImage)
    # cv2.imshow("Lines", scale(horizontalVerticalLinesImage, 1000, 1000))
    filteredActualSimpleArrows = filterNonArrows(filteredDoubleArrows, horizontalVerticalLines, 5)
    cv2.imshow("Actual Simple Arrows",
               scale(paintLines(filteredActualSimpleArrows, enhancedImage), 1000, 1000))
    #print(filterClassSegments(rectangles))
    cv2.imshow("Classes", scale(paintClasses(filterClassSegments(rectangles), enhancedImage), 1000, 1000))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Top Left to bottom right?
