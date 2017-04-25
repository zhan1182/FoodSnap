
import os

model_dir = './models'
log_dir = './logs'

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

if not os.path.isdir(log_dir):
    os.mkdir(log_dir)