import glob
from PIL import Image
import os
import sys
import numpy as np
import pickle
from PIL import ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

label_map = {'/toufu': 1, '/rib': 2, '/fish': 3, '/tm_eggs': 4, '/broc': 5, '/duck': 6}
food = glob.glob('./*')
tr_dataset = []
tr_labels = []
v_dataset = []
v_labels = []
te_dataset = []
te_labels = []

for f in food:
	tokens = f.split('.')
	if tokens[-1] == 'py':
		continue
	images = glob.glob('.' + tokens[1] + '/*')
	label = label_map[tokens[1]]
	counter = 0
	partition_2 = int(len(images) * 0.9)
	partition_1 = int(len(images) * 0.1)
	random.shuffle(images)
	for image in images:
		i = Image.open(image)
		newI = i.resize((224, 224))
		try:
			data = np.asarray(newI, dtype = 'uint8')
		except SystemError:
			data = np.asarray(newI.getdata(), dtype = 'uint8')
		if data.shape != (224, 224, 3):
			print (image)
		if counter < partition_1:
			v_dataset.append(data)
			v_labels.append(label)
		elif counter < partition_2:
			tr_dataset.append(data)
			tr_labels.append(label)
		else:
			te_dataset.append(data)
			te_labels.append(label)
		counter += 1

tr_dataset = np.asarray(tr_dataset)
tr_labels = np.asarray(tr_labels)
te_dataset = np.asarray(te_dataset)
te_labels = np.asarray(te_labels)
v_dataset = np.asarray(v_dataset)
v_labels = np.asarray(v_labels)

f = open('tr_dataset.pickle', 'wb')
pickle.dump(tr_dataset, f)
f.close()
f = open('tr_labels.pickle', 'wb')
pickle.dump(tr_labels, f)
f.close()
f = open('te_dataset.pickle', 'wb')
pickle.dump(te_dataset, f)
f.close()
f = open('te_labels.pickle', 'wb')
pickle.dump(te_labels, f)
f.close()
f = open('v_dataset.pickle', 'wb')
pickle.dump(v_dataset, f)
f.close()
f = open('v_labels.pickle', 'wb')
pickle.dump(v_labels, f)
f.close()

print (v_labels)


#images = glob.glob('./' + food + '/*')

#label = label_map[food]
#labels = np.asarray([[label]]*len(images))
#counter = 1
#dataset = []

#for image in images:
#	i = Image.open(image)
#	newI = i.resize((224, 224))
#	try:
#		data = np.asarray(newI, dtype = 'uint8')
#	except SystemError:
#		data = np.asarray(newI.getdata(), dtype = 'uint8')
#	dataset.append(data)
#	counter += 1
#dataset = np.asarray(dataset)
#print (dataset.shape)
#f = open(food + '.pickle', 'wb')
#pickle.dump(dataset, f)
#f.close()

