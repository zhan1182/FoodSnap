#! /usr/bin/env python

import socket
import pickle

import numpy as np

from vgg16 import load_saved_model

model = load_saved_model()

ip = '0.0.0.0'
port = 8888

cnn_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cnn_server.bind((ip, port))
cnn_server.listen(1)

print('CNN server created, listening port {0}'.format(port))


while True:
    connection, client_addr = cnn_server.accept()

    try:
        message = connection.recv(160000)
        print('message')
    except socket.error:
        connection.close()

    x = pickle.loads(message)
    xl = x.reshape((1, 224, 224, 3))

    result = model.predict(xl)
    preds = np.argmax(result[0])

    pp = pickle.dumps(preds)

    connection.send(pp)




