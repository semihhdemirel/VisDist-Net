import os
import shutil
import random
import argparse

# Function to create 'Val' folder for each fold and move %10 of images from training classNames to val classNames
def create_val_folders(dataset_path, fold_prefix='Fold'):
    for i in range(1, 6):  # Iterate over each fold
        fold_dir = os.path.join(dataset_path, f"{fold_prefix}{i}")
        train_dir = os.path.join(fold_dir, 'Train')
        val_dir = os.path.join(fold_dir, 'Val')
        
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        class_names = [name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))]
        print(class_names)
        for class_name in class_names:
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            
            if not os.path.exists(val_class_dir):
                os.makedirs(val_class_dir)

            # Get list of images in the training class directory
            images = os.listdir(train_class_dir)
            # Calculate 10% of the images
            val_count = int(len(images) * 0.1)
            # Randomly select images to move to val directory
            val_images = random.sample(images, val_count)

            # Move selected images to val directory
            for image in val_images:
                src = os.path.join(train_class_dir, image)
                dst = os.path.join(val_class_dir, image)
                shutil.move(src, dst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create validation folders for each fold.')
    parser.add_argument('--fold-dir', type=str, default='./fold_dataset', help='Path to dataset folds.')
    args = parser.parse_args()
    
    create_val_folders(args.fold_dir)

