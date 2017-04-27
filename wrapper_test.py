# -*- coding: utf-8 -*-

import socket
import pickle

import numpy as np
import matplotlib.pyplot as plt

from smallcnn import SmallCNN

sc = SmallCNN()
sc.load_data()

x = sc.X_test[0]

port = 8888
ip = 'localhost'

s = socket.socket()
s.connect((ip, port))

xp = pickle.dumps(x)

s.send(xp)

result = s.recv(1024)

pred = pickle.loads(result)
print(pred)

s.close()





