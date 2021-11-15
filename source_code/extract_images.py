#Randomly select and copy unlabeled images from Hyperkvasir dataset
import os, random, shutil

random.seed(42)

source_dir = '/Users/andreastoras/Documents/XAI_NorthPole/kvasirSeg_mask/Kvasir-SEG/images'
target_dir = '/Users/andreastoras/Documents/XAI_NorthPole/kvasir_clustering/external_tile_validation/original_images'
mask_dir = '/Users/andreastoras/Documents/XAI_NorthPole/kvasirSeg_mask/Kvasir-SEG/masks'
mask_target = '/Users/andreastoras/Documents/XAI_NorthPole/kvasir_clustering/external_tile_validation/original_masks'
#List of files already in target folder
#Don't want to select same image twice
selected_files = os.listdir(target_dir)


def extract_images( source_dir = source_dir, target_dir = target_dir, selected_files = selected_files, n_more_images = 0):
    print('Copying 100 kvasir-SEG images to unlabeled folder...')
    while len(selected_files)< 101 + n_more_images:
        random_file = random.choice(os.listdir(source_dir))
        #Checks that the file is not already selected
        if (random_file not in selected_files):
            selected_files.append(random_file)
            source_path = os.path.join(source_dir, random_file)
            target_path = os.path.join(target_dir, random_file)
            shutil.copy(source_path, target_path)
    print('Successfully copied images to unlabeled folder!')

#print(selected_files)
selected_files.remove('.DS_Store')
#print(selected_files)
def extract_masks(mask_dir=mask_dir, target_dir=mask_target, selected_files=selected_files):
    for _file in selected_files[1:]:
        #Remove the jpg extension
        image_name = str(_file)
        #Add another extension since .png in mask file: 
        #image_name = image + '.png'
        print('Image name:', image_name)
        #Copy from mask folder into target mask folder
        source_path = os.path.join(mask_dir, image_name)
        target_path = os.path.join(target_dir, image_name)
        shutil.copy(source_path, target_path)
    
#extract_images(source_dir, target_dir, selected_files)
extract_masks(mask_dir = mask_dir, target_dir = mask_target, selected_files=selected_files)