from PIL import Image
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Input: Image object
# Resize the image to (224,224, 3) array
# Output: numpy array
def image2array(image):
	newImage = image.resize((224, 224))
	try:
		data = np.asarray(newImage, dtype = 'uint8')
	except SystemError:
		data = np.asarray(newImage.getdata(), dtype = 'uint8')
	if data.shape != (224, 224, 3):
		print ("data array not match!")
	return data


# Example
if __name__ == "__main__":
	image = Image.open("./broc/273.jpg")
	array = image2array(image)
	print (array)