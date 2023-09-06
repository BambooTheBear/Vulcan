# Imports
import math

import cv2
import numpy as np
from PIL import Image
import pytesseract

from generateCode import generateCode


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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=5, maxLineGap=0)

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


def paintLinesColored(points, colored_image):
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
        for i, ((x1, y1), (x2, y2)) in enumerate(rectangleList):
            if i == 0:
                cv2.rectangle(colored_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif i == 1:
                cv2.rectangle(colored_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif i == 2:
                cv2.rectangle(colored_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return colored_image


def paintClassesColored(rectangles, colored_image):
    for j, rectangleList in enumerate(rectangles):
        for i, ((x1, y1), (x2, y2)) in enumerate(rectangleList):
            # cv2.imshow(f"{pytesseract.image_to_string(Image.fromarray(cut(x1,y1,x2,y2,colored_image)))}", cut(x1,y1,x2,y2,colored_image))
            if i == 0:
                cv2.rectangle(colored_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif i == 1:
                cv2.rectangle(colored_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif i == 2:
                cv2.rectangle(colored_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return colored_image


def getClassesText(rectangles, original_image):
    classesJson=[]
    for j, rectangleList in enumerate(rectangles):
        o = {"ClassName": "", "Attributes": [], "Methods": []}
        tmp = False
        for i, ((x1, y1), (x2, y2)) in enumerate(rectangleList):
            t = pytesseract.image_to_string(Image.fromarray(cut(x1, y1, x2, y2, original_image)))
            # cv2.imshow(f"{t}", cut(x1, y1, x2, y2, original_image))
            if tmp or len(rectangleList) == 1:
                t = t.replace("\n", "")
                o["ClassName"] = t
            elif len(rectangleList) == 2:
                if "(" in t:
                    x = t.split("\n")
                    for s in x:
                        if s != "":
                            o["Methods"].append(s)
                    tmp = True
                elif i == 0:
                    x = t.split("\n")
                    for s in x:
                        if s != "":
                            o["Attributes"].append(s)
                    tmp = True
                else:
                    t = t.replace("\n", "")
                    o["ClassName"] = t
            else:
                if i == 0:
                    x = t.split("\n")
                    for s in x:
                        if s != "":
                            o["Methods"].append(s)
                if i == 1:
                    x = t.split("\n")
                    for s in x:
                        if s != "":
                            o["Attributes"].append(s)
                else:
                    t = t.replace("\n", "")
                    o["ClassName"] = t
        print(o)
        classesJson.append(o)

    return original_image, classesJson


def paintRectangles(rectangles, image):
    for ((x1, y1), (x2, y2)) in rectangles:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    return image


def is_line_inside_rect(line, rect):
    x1, y1 = line[0]
    x2, y2 = line[1]
    rx1, ry1 = rect[0]
    rx2, ry2 = rect[1]

    return rx1 <= x1 <= rx2 and ry1 <= y1 <= ry2 and rx1 <= x2 <= rx2 and ry1 <= y2 <= ry2


def filterHVLines(lines, rectangles, threshold):
    filtered_lines = []

    for line in lines:
        is_inside = False

        for rect in rectangles:
            extended_rect = ((rect[0][0] - threshold, rect[0][1] - threshold),
                             (rect[1][0] + threshold, rect[1][1] + threshold))

            if is_line_inside_rect(line, extended_rect):
                is_inside = True
                break

        if not is_inside:
            filtered_lines.append(line)

    return filtered_lines


def drawEverything(inheritanceArows, assosiationArrows, classes, blackWhiteImage):
    image = cv2.cvtColor(blackWhiteImage, cv2.COLOR_GRAY2BGR)
    image = paintRectangles(inheritanceArows, image)
    image = paintLinesColored(assosiationArrows, image)
    image = paintClassesColored(classes, image)
    image = scale(image, 1000, 1000)
    cv2.imshow("Everything", image)


def cut(x1, y1, x2, y2, image):
    return image[y1:y2, x1:x2]


def flatten_list_of_lists(list_of_lists):
    flattened_list = []
    for sublist in list_of_lists:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def mark_adjacent_lines(lines, arrowHeads, marked_lines, proximity_threshold):
    newly_marked = []

    for line in lines:
        if line in marked_lines:
            continue

        for rect in arrowHeads:
            if (distance(line[0], rect[0]) <= proximity_threshold or
                    distance(line[0], rect[1]) <= proximity_threshold or
                    distance(line[1], rect[0]) <= proximity_threshold or
                    distance(line[1], rect[1]) <= proximity_threshold):
                marked_lines.append(line)
                newly_marked.append(line)
                break

        for marked_line in marked_lines:
            if (distance(line[0], marked_line[0]) <= proximity_threshold or
                    distance(line[0], marked_line[1]) <= proximity_threshold or
                    distance(line[1], marked_line[0]) <= proximity_threshold or
                    distance(line[1], marked_line[1]) <= proximity_threshold):
                marked_lines.append(line)
                newly_marked.append(line)
                break

    return newly_marked


def find_connected_lines(lines, proximity_threshold):
    connected_lines = {}
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i != j and (distance(line1[0], line2[0]) <= proximity_threshold or
                           distance(line1[0], line2[1]) <= proximity_threshold or
                           distance(line1[1], line2[0]) <= proximity_threshold or
                           distance(line1[1], line2[1]) <= proximity_threshold):
                connected_lines.setdefault(line1, []).append(line2)

    return connected_lines


def filter_marked_lines(lines, arrowHeads, proximity_threshold):
    marked_lines = []
    new_marked_lines = mark_adjacent_lines(lines, arrowHeads, marked_lines, proximity_threshold)

    line_arrowhead_dict = {}  # Dictionary to store line-arrowhead associations

    while new_marked_lines:
        for line in new_marked_lines:
            for rect in arrowHeads:
                if (distance(line[0], rect[0]) <= proximity_threshold or
                        distance(line[0], rect[1]) <= proximity_threshold or
                        distance(line[1], rect[0]) <= proximity_threshold or
                        distance(line[1], rect[1]) <= proximity_threshold):
                    line_arrowhead_dict.setdefault(line, []).append(rect)

        marked_lines.extend(new_marked_lines)
        new_marked_lines = mark_adjacent_lines(lines, arrowHeads, marked_lines, proximity_threshold)

    connected_lines = find_connected_lines(lines, proximity_threshold)

    return marked_lines, connected_lines


def extractIndirectConnections(graph):
    def findIndirectConnections(node, x):
        indirect_connections = []
        if x < 5:
            if node in graph:
                for neighbor in graph[node]:
                    indirect_connections.extend(findIndirectConnections(neighbor, x + 1))
                    indirect_connections.append(neighbor)
        return indirect_connections

    indirect_graph = {}
    for node in graph:
        indirect_connections = findIndirectConnections(node, 0)
        direct_connections = graph[node]
        all_connections = list(set(direct_connections + indirect_connections))
        indirect_graph[node] = all_connections

    return indirect_graph


def extractShortestConnection(dictionary):
    result = []
    for key, values in dictionary.items():
        valuesSorted = sorted(values, key=lambda p: math.sqrt((key[0][0] - p[1][0]) ** 2 + (key[0][1] - p[1][1]) ** 2))
        result.append((key[0], valuesSorted[-1][1]))
    return result


def filterLinesWithArrowhead(lines, arrowheads, classes):
    filtered_lines = []
    lineRectangles = {}
    tolerance = 20
    for line in lines:
        start_point, end_point = line
        for rect1 in arrowheads:
            for rect2 in classes:
                x1, y1 = start_point
                (mx1, my1), (mx2, my2) = rect1
                if (
                        ((mx1 + tolerance < x1 < mx2 - tolerance) or (mx1 - tolerance < x1 < mx2 + tolerance) or (
                                mx1 + tolerance > x1 > mx2 - tolerance) or (
                                 mx1 - tolerance > x1 > mx2 + tolerance)) and (
                        (my1 + tolerance < y1 < my2 - tolerance) or (my1 - tolerance < y1 < my2 + tolerance) or (
                        my1 + tolerance > y1 > my2 - tolerance) or (my1 - tolerance > y1 > my2 + tolerance))
                ):
                    x1, y1 = end_point
                    (mx1, my1), (mx2, my2) = rect2
                    if (
                            ((mx1 + tolerance < x1 < mx2 - tolerance) or (mx1 - tolerance < x1 < mx2 + tolerance) or (
                                    mx1 + tolerance > x1 > mx2 - tolerance) or (
                                     mx1 - tolerance > x1 > mx2 + tolerance)) and (
                            (my1 + tolerance < y1 < my2 - tolerance) or (my1 - tolerance < y1 < my2 + tolerance) or (
                            my1 + tolerance > y1 > my2 - tolerance) or (my1 - tolerance > y1 > my2 + tolerance))
                    ):
                        lineRectangles.setdefault(line, (rect2, rect1))
                        filtered_lines.append(line)
                else:
                    x1, y1 = end_point
                    if (
                            ((mx1 + tolerance < x1 < mx2 - tolerance) or (mx1 - tolerance < x1 < mx2 + tolerance) or (
                                    mx1 + tolerance > x1 > mx2 - tolerance) or (
                                     mx1 - tolerance > x1 > mx2 + tolerance)) and (
                            (my1 + tolerance < y1 < my2 - tolerance) or (my1 - tolerance < y1 < my2 + tolerance) or (
                            my1 + tolerance > y1 > my2 - tolerance) or (my1 - tolerance > y1 > my2 + tolerance))
                    ):
                        x1, y1 = start_point
                        (mx1, my1), (mx2, my2) = rect2
                        if (
                                ((mx1 + tolerance < x1 < mx2 - tolerance) or (
                                        mx1 - tolerance < x1 < mx2 + tolerance) or (
                                         mx1 + tolerance > x1 > mx2 - tolerance) or (
                                         mx1 - tolerance > x1 > mx2 + tolerance)) and (
                                (my1 + tolerance < y1 < my2 - tolerance) or (
                                my1 - tolerance < y1 < my2 + tolerance) or (
                                        my1 + tolerance > y1 > my2 - tolerance) or (
                                        my1 - tolerance > y1 > my2 + tolerance))
                        ):
                            lineRectangles.setdefault(line, (rect2, rect1))
                            filtered_lines.append(line)
    return filtered_lines, lineRectangles


def detectRelations(connectionDict, classes, original_image):
    connections=[]
    tolerance = 20
    for ((lineX1, lineY1), (lineX2, lineY2)), (
    ((startX1, startY1), (startX2, startY2)), ((endX1, endY1), (endX2, endY2))) in connectionDict.items():
        for clas in classes:
            (mx1, my1), (mx2, my2) = clas[0]
            if (
                    ((mx1 + tolerance < endX1 < mx2 - tolerance) or (
                            mx1 - tolerance < endX1 < mx2 + tolerance) or (
                             mx1 + tolerance > endX1 > mx2 - tolerance) or (
                             mx1 - tolerance > endX1 > mx2 + tolerance)) and (
                    (my1 + tolerance < endY1 < my2 - tolerance) or (
                    my1 - tolerance < endY1 < my2 + tolerance) or (
                            my1 + tolerance > endY1 > my2 - tolerance) or (
                            my1 - tolerance > endY1 > my2 + tolerance))
            ):
                for clas2 in classes:
                    if ((startX1, startY1), (startX2, startY2)) in clas2:
                        (mx1, my1), (mx2, my2) = clas[-1]
                        if ((mx1, my1), (mx2, my2)) != ((startX1, startY1), (startX2, startY2)):
                            t = ""+pytesseract.image_to_string(
                                Image.fromarray(cut(startX1, startY1, startX2, startY2, original_image)))
                            s = ""+pytesseract.image_to_string(
                                Image.fromarray(cut(mx1, my1, mx2, my2, original_image)))
                            print(f"Connection from {t} to {s}")
                            t = t.replace("«abstract»", "").replace("«interface»", "").replace("<<abstract>>", "").replace("<<interface>>", "")
                            s = s.replace("«abstract»", "").replace("«interface»", "").replace("<<abstract>>", "").replace("<<interface>>", "")
                            connections.append((t, s))
    return connections

def printDictNice(dictionary):
    for key, values in dictionary.items():
        print(f"   {key}     ")
        for value in values:
            print(f"         {value} ;")


# Main Method
if __name__ == '__main__':
    print("Started Vulcan")
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Benno\AppData\Local\Tesseract-OCR\tesseract.exe'
    filePath = "./data/custom/semi_complex.png"

    enhancedImage = enhanceImage(filePath)

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
    cv2.imshow("Lines", scale(horizontalVerticalLinesImage, 1000, 1000))
    filteredActualSimpleArrows = filterNonArrows(filteredDoubleArrows, horizontalVerticalLines, 5)
    cv2.imshow("Actual Simple Arrows",
               scale(paintLines(filteredActualSimpleArrows, enhancedImage), 1000, 1000))
    classes = filterClassSegments(rectangles.copy())
    cv2.imshow("Classes", scale(paintClasses(classes, enhancedImage), 1000, 1000))

    filterdLines = filterHVLines(horizontalVerticalLines, flatten_list_of_lists(classes), 10)
    filterdLinesImage = paintLines(filterdLines, enhancedImage)
    cv2.imshow("filtered Lines", scale(filterdLinesImage, 1000, 1000))

    markedLines, markedLinesDict = filter_marked_lines(filterdLines, triangles, 50)
    markedLinesImage = paintLines(markedLines, enhancedImage)
    cv2.imshow("Marked Lines", scale(markedLinesImage, 1000, 1000))

    indirectLines = extractIndirectConnections(markedLinesDict)
    shortestConnections = extractShortestConnection(indirectLines)
    indirectLinesImage = paintLines(indirectLines, enhancedImage)
    cv2.imshow("Indirect Lines", scale(indirectLinesImage, 1000, 1000))
    shortestConnectionImage = paintLines(shortestConnections, enhancedImage)
    cv2.imshow("Shortest Connection Lines", scale(shortestConnectionImage, 1000, 1000))
    actualConnections, connectionDict = filterLinesWithArrowhead(shortestConnections, triangles, rectangles)
    actualConnectionImage = paintLines(actualConnections, enhancedImage)
    cv2.imshow("Actual Connection Lines", scale(actualConnectionImage, 1000, 1000))

    relations = detectRelations(connectionDict, classes, cv2.imread(filePath, cv2.IMREAD_GRAYSCALE))

    drawEverything(triangles, filteredActualSimpleArrows, classes, enhancedImage)

    classesImage, classesJson = getClassesText(classes, cv2.imread(filePath, cv2.IMREAD_GRAYSCALE))

    generateCode(classesJson, relations)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Top Left to bottom right?
