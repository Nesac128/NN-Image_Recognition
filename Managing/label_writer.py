import os
import string
from itertools import product
import json

bspath = '/home/planetgazer8360/PycharmProjects/NN-Image_recognition/training_images/'

folders = os.listdir(bspath)
print(folders)

keys = []

for k in product(string.ascii_uppercase, repeat=2):
    keys.append(k[0]+k[1])

ukeys = []

for kn in range(3):
    ukeys.append(keys[kn])


obj_to_lb = {}

with open('datasets/fruits_ALP/labels2.txt', 'w') as labels:
    i = 0
    for folder in folders:
        for image in os.listdir(bspath+folder):
            labels.write(ukeys[i]+'\n')
        obj_to_lb[folder.split('-')[0]] = ukeys[i]
        i += 1

with open('datasets/fruits_ALP/obj_labels.json', 'w') as txt:
    json.dump(obj_to_lb, txt, indent=3, sort_keys=True)
