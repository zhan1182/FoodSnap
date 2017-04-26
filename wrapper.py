#! /usr/bin/env python
# -*- coding: utf-8 -*-

import socket

ip = '0.0.0.0'
port = 8888

cnn_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cnn_server.bind((ip, port))
cnn_server.listen(1)

print('CNN server created, listening port {0}'.format(port))


while True:
    connection, client_addr = cnn_server.accept()

    try:
        message = connection.recv(256)
        print('message = {0}'.format(message))
    except socket.error:
        connection.close()

    query = message.decode('utf-8').strip()

    if not query:
        js = json.dumps({u'text': 'No data received.'})
        connection.send(js.encode('utf-8'))

    # classify_result = nlp_search.search(query)

    js = json.dumps(classify_result)

    connection.send(js.encode('utf-8'))

