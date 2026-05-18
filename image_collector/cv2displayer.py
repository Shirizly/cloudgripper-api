import cv2
import numpy as np

# Function to display multiple images using OpenCV

def display_images(images, window_name="Robot Images"):
    # Get dimensions of the first image
    height, width = images[1].shape[:2]

    # Resize all images to the same dimensions
    resized_images = [cv2.resize(image, (width, height)) for image in images]

    # Concatenate images horizontally
    concatenated_image = np.concatenate(resized_images, axis=0)

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

def update_images(images, window_name="Robot Images"):
    # Get dimensions of the first image
    # height0, width0 = images[0].shape[:2]
    # # print(f"Image 0 dimensions: {width}x{height}")
    # height1, width1 = images[1].shape[:2]
    # print(f"Image 1 dimensions: {width}x{height}")

    # Resize all images to the same height, maintaining aspect ratio
    # target_height = min(height0, height1)
    target_height = images[0].shape[0] 
    target_height = 500
    resized_images = []
    for image in images:
        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = int(target_height * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, target_height))
        resized_images.append(resized_image)
    

    # Concatenate images horizontally
    concatenated_image = np.concatenate(resized_images, axis=1)

    # Display the image
    cv2.imshow(window_name, concatenated_image)
    cv2.waitKey(10)  # Small delay to allow image to render