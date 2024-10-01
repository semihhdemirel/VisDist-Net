import os
import shutil
import random
import argparse

def split_dataset_into_folds(class_path, output_path, num_folds=5):
            # Shuffle images within each class
        images = os.listdir(class_path)
        random.shuffle(images)
        
        # Divide images into equal parts
        num_images = len(images)
        images_per_fold = num_images // num_folds
        remainder = num_images % num_folds

        start_index = 0
        for fold in range(1, num_folds+1):
            fold_path = os.path.join(output_path, f"Fold{fold}")
            train_path = os.path.join(fold_path, "Train", class_name)
            test_path = os.path.join(fold_path, "Test", class_name)
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)
            
            # Determine the number of images for this fold
            fold_images = images_per_fold + (1 if fold <= remainder else 0)
            end_index = start_index + fold_images
            
            # Assign images to train and test sets
            test_images = images[start_index:end_index]
            train_images = [img for img in images if img not in test_images]
            
            # Copy images to respective folders
            for image in test_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(test_path, image))
            for image in train_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(train_path, image))
            
            # Update start index for the next fold
            start_index = end_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into 5 folds")
    parser.add_argument("--input-dir", type=str, default="./fruit_dataset", help="Path to the original dataset")
    parser.add_argument("--output-dir", type=str, default="./fold_dataset", help="Path to save the folds")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds to create")
    args = parser.parse_args()

    # Create folds directories
    for fold in range(1, args.num_folds + 1):
        fold_path = os.path.join(args.output_dir, f"Fold{fold}")
        os.makedirs(os.path.join(fold_path, "Train"), exist_ok=True)
        os.makedirs(os.path.join(fold_path, "Test"), exist_ok=True)

    # Iterate over classes
    for class_name in os.listdir(args.input_dir):
        class_path = os.path.join(args.input_dir, class_name)
        if os.path.isdir(class_path):
            # Shuffle images within each class
            split_dataset_into_folds(class_path, args.output_dir, args.num_folds)
    
    print("Dataset split into 5 folds successfully.")
