# Imports
import cv2
import numpy as np


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
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 10:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle

    return color_image


# Main Method
if __name__ == '__main__':
    print("Started Vulcan")

    enhancedImage = enhanceImage("./data/custom/simple_1.png")

    # Show Image
    cv2.imshow("Enhanced Image", enhancedImage)
    cv2.imshow("Rectangles", detectRectangles(enhancedImage))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
