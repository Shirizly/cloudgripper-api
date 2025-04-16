import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from ImageDataset import ImageDataset as ID
from background_subtraction import process_image, load_image_dataset
import time

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)

def process_and_save_images(sequence_dirs, background_clean, output_root,mode="with objects"):
    for sequence_dir in sequence_dirs:
        start_time = time.time()
        sequence_name = os.path.basename(sequence_dir)
        output_dir_robot = os.path.join(output_root, sequence_name, "robot")
        output_dir_robot_objects = os.path.join(output_root, sequence_name, "robot_objects")
        
        os.makedirs(output_dir_robot, exist_ok=True)
        os.makedirs(output_dir_robot_objects, exist_ok=True)
        
        image_files = sorted([f for f in os.listdir(os.path.join(sequence_dir,"top_images")) if f.endswith(".png")])
        dataset = ID([sequence_dir], transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        image_tensors = load_image_dataset(loader, num_batches=2)
        background_tensor = torch.median(image_tensors, dim=0).values
        background = transforms.ToPILImage()(background_tensor)
        background.save(os.path.join(output_dir_robot,"median_image.png"))

        mask_count_dict = {}
        mask_ob_count_dict = {}
        for image_file in image_files:
            image_path = os.path.join(sequence_dir,"top_images", image_file)
            image = Image.open(image_path)
            
            
            image_filtered_robot, mask_robot, filtered_masks = process_image(image, background)
            # Save the processed image and mask
            save_image(image_filtered_robot, os.path.join(output_dir_robot, image_file))
            save_image(mask_robot, os.path.join(output_dir_robot, "mask_" + image_file))
            # add the number of masks to a dictionary
            mask_count = len(filtered_masks)
            mask_count_dict[image_file] = mask_count

            if mode == "with objects":
                image_filtered_robot_objects, mask_robot_objects, filtered_masks = process_image(image, background_clean)
                mask_ob_count = len(filtered_masks)
                mask_ob_count_dict[image_file] = mask_count
                save_image(image_filtered_robot_objects, os.path.join(output_dir_robot_objects, image_file))
                save_image(mask_robot_objects, os.path.join(output_dir_robot_objects, "mask_" + image_file))

        # Write the mask count dictionary to a file
        mask_count_file_path = os.path.join(output_dir_robot, "mask_count.txt")
        with open(mask_count_file_path, "w") as mask_count_file:
            for key, value in mask_count_dict.items():
                mask_count_file.write(f"{key} {value}\n")
        mask_ob_count_file_path = os.path.join(output_dir_robot, "mask_ob_count.txt")
        with open(mask_ob_count_file_path, "w") as mask_count_file:
            for key, value in mask_ob_count_dict.items():
                mask_count_file.write(f"{key} {value}\n")
        print(f"Processed sequence {sequence_name}. Time taken to process the sequence: {time.time()-start_time}")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    recordings_dir = os.path.join(project_root, "recordings")
    output_root = os.path.join(project_root, "processed_recordings")

    sequence_dirs = []
    if os.path.exists(recordings_dir):
        dataset_dirs = [d for d in os.listdir(recordings_dir) if d.startswith("dataset_0_sequence_")]
        for sequence in dataset_dirs:
            sequence_dirs.append(os.path.join(recordings_dir, sequence))
    
    dataset = ID(sequence_dirs, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    image_tensors = load_image_dataset(loader, num_batches=4)
    
    background_tensor = torch.median(image_tensors, dim=0).values
    background = transforms.ToPILImage()(background_tensor)
    background.save("median_image.png")
    
    process_and_save_images(sequence_dirs, background, output_root)
    
if __name__ == "__main__":
    main()
