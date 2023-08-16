# Imports
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

    # Iterate through the contours and draw rectangles with colored outlines
    for contour in contours:
        # OG: 0.04
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 3 detects arrows? 10 detects plus signs?
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)

            if 30 <= w <= image.shape[1]*0.9 and 30 <= h <= image.shape[0]*0.9:
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle

    return color_image


def detectArrows(image):

    # Convert the image to binary using thresholding
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Convert to color image
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Iterate through the contours and draw rectangles with colored outlines
    for contour in contours:
        # OG: 0.04
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 3 detects arrows? 10 detects plus signs?
        if len(approx) == 3:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw green rectangle

    return color_image

def detectSimpleArrows(image):
    # Convert the image to binary using thresholding
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Convert to color image
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Iterate through the contours and draw rectangles with colored outlines
    for i, contour in enumerate(contours):
        #print(contour)
        # OG: 0.04
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 3 detects arrows? 10 detects plus signs?
        if True or len(approx) == 2:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(color_image, (x, y), (x + w, y + h), (i*100, 0, 255), 2)  # Draw green rectangle

    return color_image

def detectLines(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred_image, 50, 150)

    # Perform Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=5)

    # Extract starting and ending points of detected lines
    line_points = []
    marked_image=cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle_deg = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))  # Calculate angle in degrees
            print(f"from ({x1} | {y1}) to ({x2} | {y2}) with slope {angle_deg}")
            # Check if the angle falls within the desired ranges
            range_deg = 5
            if not (0 <= angle_deg <= range_deg or 90-range_deg <= angle_deg <= 90+range_deg or 180-range_deg <= angle_deg <= 180+range_deg or
                    270-range_deg <= angle_deg <= 290+range_deg or 360-range_deg <= angle_deg <= 360):
                line_points.append(((x1, y1), (x2, y2)))
                cv2.line(marked_image, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Draw line in blue
            else:
                range_deg = 2
                if (0 <= angle_deg <= range_deg or 90 - range_deg <= angle_deg <= 90 + range_deg or 180 - range_deg <= angle_deg <= 180 + range_deg or
                        270 - range_deg <= angle_deg <= 290 + range_deg or 360 - range_deg <= angle_deg <= 360):
                    line_points.append(((x1, y1), (x2, y2)))
                    cv2.line(marked_image, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Draw line in blue
                cv2.line(marked_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Draw line in blue

    return marked_image


# Main Method
if __name__ == '__main__':
    print("Started Vulcan")

    enhancedImage = enhanceImage("./data/custom/simple_2.png")

    # Show Image
    cv2.imshow("Enhanced Image", scale(enhancedImage, 1000, 1000))
    cv2.imshow("Rectangles", scale(detectRectangles(enhancedImage), 1000, 1000))
    cv2.imshow("Arrows", scale(detectArrows(enhancedImage), 1000, 1000))
    #cv2.imshow("Simple Arrows", scale(detectSimpleArrows(enhancedImage), 1000, 1000))
    cv2.imshow("Lines", scale(detectLines(enhancedImage), 1000, 1000))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
