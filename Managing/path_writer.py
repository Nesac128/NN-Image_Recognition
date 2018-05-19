import os

bspath = '/media/planetgazer8360/Elements/Object recognition projects/Datasets/StanfordDogs/Resized_images/'

folders = os.listdir(bspath)

with open('/home/planetgazer8360/PycharmProjects/NN-Image_recognition/StanfordDogs/paths.txt', 'w') as paths:
    for folder in folders:
        for image in os.listdir(bspath+folder):
            paths.write(bspath+folder+'/'+image+'\n')
