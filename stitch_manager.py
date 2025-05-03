from image_stitcher import stitch
import os

def add_together(image1, image2, outputFilePath, name = 'panorama'):
    imageFileName1 = image1.split('/')[-1]
    imageFileName2 = image2.split('/')[-1]
    os.rename(image1, "cache_overflow/" + imageFileName1)
    os.rename(image2, "cache_overflow/" + imageFileName2)
    stitch("cache_overflow", 0, range = None, panorama_name= name, type='.jpg', path = outputFilePath)

def add_image_array(images, outputFilePath, name = 'panorama'):
    for image in images:
        imageFileName = image.split('/')[-1]
        os.rename(image, "cache_overflow/" + imageFileName)
    stitch("cache_overflow", 0, range = None, panorama_name= name, type='.jpg', path = outputFilePath)

def move_image_array_to_folder(images, folder):
    for image in images:
        imageFileName = image.split('/')[-1]
        os.rename(image, folder + imageFileName)

def dyna_stitch_folder(folder, outputFilePath, name = 'panorama'):
    all_images = []
    for filename in os.listdir(folder):
        #Create full file path 
        f = os.path.join(folder, filename)
        #Append to images
        all_images.append(f)
    # print(all_images)
    for i in range(0, len(all_images) // 5):
        pass

move_image_array_to_folder(['ImgSampleA1_2\DJI_0001.JPG', 'ImgSampleA1_2\DJI_0002.JPG', 'ImgSampleB\googlecity_001.png'], 'cache_overflow\\')
        





# images = [
#     'ImgSampleA1_10/DJI_0001.JPG',
#     'ImgSampleA1_10/DJI_0002.JPG',
#     'ImgSampleA1_10/DJI_0003.JPG',
#     'ImgSampleA1_10/DJI_0004.JPG',
#     'ImgSampleA1_10/DJI_0005.JPG',
#     'ImgSampleA1_10/DJI_0006.JPG',
#     'ImgSampleA1_10/DJI_0007.JPG',
#     'ImgSampleA1_10/DJI_0008.JPG',
#     'ImgSampleA1_10/DJI_0009.JPG',
#     'ImgSampleA1_10/DJI_00010.JPG',
#     ]
# add_image_array(images, 'cache_stitched')

dyna_stitch_folder('ImgSampleE', 'cache_stitched')

