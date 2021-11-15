from PIL import Image
import os


def imgcrop(image_path, xPieces, yPieces):
    pieces = image_path.split('/')
    #The last part equals the image:
    image_name = pieces[-1]
    image_name, extension = image_name.split('.')
    #print('Image name:', image_name)
    #print('Extension:', extension)
    im = Image.open(image_path)
    imgwidth, imgheight = im.size
    height = imgheight // yPieces
    width = imgwidth // xPieces
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            a = im.crop(box)
            a.save(out_path + '/'+ image_name + "-" + str(i) + "-" + str(j) + '.'+ extension)


image_path = '/Users/andreastoras/Downloads/MediaEvalMedico_21TestDataset/images'
selected_files = os.listdir(image_path)
#selected_files.remove('.DS_Store')
#img = Image.open(image_path)
#h, w = img.size
out_path = '/Users/andreastoras/Downloads/MediaEvalMedico_21TestDataset/tile_images'

for img_file in selected_files:
    source_path = os.path.join(image_path, img_file)
    imgcrop(source_path, 2, 2)