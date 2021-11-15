from PIL import Image
import os
import cv2

whole_image_path = '/Users/andreastoras/Downloads/MediaEvalMedico_21TestDataset/images'
tile_mask_path = '/Users/andreastoras/Documents/XAI_NorthPole/michael_efficientnet_gradcam/segmentation_masks_testdata'
target_folder = '/Users/andreastoras/Documents/XAI_NorthPole/michael_efficientnet_gradcam/whole_segmentation_masks_testdata'
selected_files = os.listdir(whole_image_path)
tile_files = os.listdir(tile_mask_path)

for _img in selected_files:
    _img = _img.strip('.jpg')
    tile_parts = []
    #Selects the four tiles belonging to same whole image
    for tile in tile_files:
        if _img in tile:
            tile_parts.append(tile)
    #Marks the tiles as upper left/right and lower right/left
    for _tile in tile_parts:
        if '-0-0.jpg' in _tile[-8:]:
            upper_left = _tile
        elif '-0-1.jpg' in _tile[-8:]:
            upper_right = _tile
        elif '-1-0.jpg' in _tile[-8:]:
            lower_left = _tile
        elif '-1-1.jpg' in _tile[-8:]:
            lower_right = _tile
    upper_left_img = cv2.imread(os.path.join(tile_mask_path, upper_left))
    upper_right_img = cv2.imread(os.path.join(tile_mask_path, upper_right))
    lower_left_img = cv2.imread(os.path.join(tile_mask_path, lower_left))
    lower_right_img = cv2.imread(os.path.join(tile_mask_path, lower_right))
    im_h_upper = cv2.hconcat([upper_left_img, upper_right_img])
    im_h_lower = cv2.hconcat([lower_left_img, lower_right_img])
    whole_image = cv2.vconcat([im_h_upper, im_h_lower])
    cv2.imwrite(target_folder + '/' + _img + '.jpg', whole_image)
