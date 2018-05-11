import csv
from PIL import Image
import os


class PathManager:
    def __init__(self, im_paths, id):
        self.im_paths = im_paths
        self.id = id

        if not os.path.exists(im_paths):
            with open(im_paths, 'w') as pathfile:
                pathfile.close()

    def copy_paths(self):
        paths = self.clean_read_paths()
        sess_id = int(self.id)
        with open('metadata/cpaths.csv', 'a') as pathfile:
            writer = csv.writer(pathfile, delimiter=',')
            writer.writerow([sess_id])
            for path in paths:
                writer.writerow([path])

        return True

    def read_paths(self):
        with open(self.im_paths, 'r') as imgfile:
            paths = imgfile.readlines()
        return paths

    def clean_read_paths(self):
        paths = self.read_paths()
        for n in range(len(paths)):
            paths[n] = paths[n].split('\n')[0]
        return paths


class ImageLoader:
    def __init__(self, im_paths):
        self.rgb_vals = []
        self.imgs = []
        self.n_images = len(im_paths)

        self.im_paths = im_paths
        self.open_images()

        self.pixels = []

        self.sess_id()

    def sess_id(self):
        with open('im_session.txt', 'r') as sessfile:
            i = int(sessfile.readline())

        with open('im_session.txt', 'w') as sessfile:
            sessfile.write(str(i + 1))

    def get_sess_id(self):
        with open('im_session.txt', 'r') as sessfile:
            return sessfile.readline()

    def open_images(self):
        for img_path in self.clean_read_paths():
            img = Image.open(img_path)
            self.imgs.append(img)

        return True

    def read_paths(self):
        with open(self.im_paths, 'r') as imgfile:
            paths = imgfile.readlines()
        return paths

    def clean_read_paths(self):
        paths = self.read_paths()
        for n in range(len(paths)):
            paths[n] = paths[n].split('\n')[0]
        return paths

    def load_pixels(self):
        raw_pixels = []
        for img in self.imgs:
            pixels = img.load()
            raw_pixels.append(pixels)
        return raw_pixels

    def mean_pixels(self):
        for pixels in self.load_pixels():
            im_pixels = []
            f = []
            for n_image in range(self.n_images):
                f.append(n_image)
                if n_image > f[0]:
                    break
                else:
                    for x in range(self.get_dims()[n_image][0]):
                        for y in range(self.get_dims()[n_image][1]):
                            rgb_sum = pixels[x, y][0] + pixels[x, y][1] + pixels[x, y][2]
                            rgb_avr = rgb_sum / 3
                            im_pixels.append(rgb_avr)
            self.pixels.append(im_pixels)
        return self.pixels

    def get_dims(self):
        sizes = []
        for img in self.imgs:
            sizes.append(img.size)
        return sizes

    def main(self):
        #####################################################
        pman = PathManager(self.im_paths, self.get_sess_id())
        pman.copy_paths()
        #####################################################

        return self.getRGB()

    def getRGB(self):
        rgb_vals = []
        self.load_pixels()
        for im_pixels in self.mean_pixels():
            rgb_vals.append(im_pixels)
            self.rgb_vals.append(im_pixels)
        return self.rgb_vals


class ImageDataWriter:
    def __init__(self, data, fname):
        self.raw_data = data
        self.fname = fname

    def main(self):
        for image_data in self.raw_data:
            self.writeCSV(image_data)

    def writeCSV(self, img):
        with open(self.fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)


class ImageTrainDataWriter:
    def __init__(self, data, fname, labels_path):
        # Store parameters in variables
        self.input_data = data
        self.fname = fname
        self.labels_path = labels_path

        self.labels = []

    def read_labels(self):
        with open(self.labels_path, 'r') as txtfile:
            raw_data = txtfile.readlines()
            txtfile.close()
        return raw_data

    def clean_raw_data(self):
        raw = self.read_labels()
        clean = []
        for line in range(len(raw)):
            clean.append(raw[line].split('\n')[0])
        self.labels = clean

    def main(self):
        self.clean_raw_data()
        for imn in range(len(self.input_data)):
            self.input_data[imn].append(self.labels[imn])
            self.writeCSV(self.input_data[imn])

    def writeCSV(self, img):
        with open(self.fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)


i = ImageLoader('metadata/paths/images3.txt')
data = i.main()
print(len(data))
itdw = ImageDataWriter(data, 'data/unclassified2.csv')
itdw.main()


# i = ImageLoader('metadata/paths/images2.txt')
# pixel_data = i.main()
# itdw = ImageDataWriter(pixel_data, 'data/unclassified_data4.csv')
# itdw.main()
