import cv2
import numpy as np

# Function to display multiple images using OpenCV

def display_images(images, window_name="Robot Images"):
    # Get dimensions of the first image
    height, width = images[0].shape[:2]

    # Resize all images to the same dimensions
    resized_images = [cv2.resize(image, (width, height)) for image in images]

    # Concatenate images horizontally
    concatenated_image = np.concatenate(resized_images, axis=1)

    # Display the image
    cv2.imshow(window_name, concatenated_image)
    while True:
    # Check if the window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed!")
            break

        # Press any key to exit manually
        key = cv2.waitKey(100)  # Wait for a key press (100ms delay)
        if key != -1:  # Any key except -1 (no key press)
            print("Quit key pressed!")
            break

    cv2.destroyAllWindows()