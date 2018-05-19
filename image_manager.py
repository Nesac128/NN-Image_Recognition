from PIL import Image
from pcontrol import *
import time


class ImageLoader:
    def __init__(self, im_paths):
        self.sess = Sess()
        self.sess.add()

        self.rgb_vals = []
        self.imgs = []
        self.n_images = len(im_paths)

        self.pfile = im_paths
        self.open_images()

        self.pixels = []

        if not os.path.exists('metadata/sess/'+str(self.sess.read()+'/')):
            os.mkdir('metadata/sess/'+str(self.sess.read()+'/'))

        self.Meta = MetaData(self.sess.read())

    def open_images(self):
        reader = Reader(self.pfile)
        dat = reader.clean_read()
        for img_path in dat:
            print("Reading image: ", img_path)
            img = Image.open(img_path)
            self.imgs.append(img)
        return True

    def load_pixels(self):
        raw_pixels = []
        for img in self.imgs:
            print("Loading image: ", img)
            pixels = img.load()
            raw_pixels.append(pixels)
        return raw_pixels

    def mean_pixels(self):
        print("Began mean_pixels ...")
        time.sleep(5)
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
        print("Finished mean_pixels ...")
        time.sleep(3)
        return self.pixels

    def get_dims(self):
        sizes = []
        for img in self.imgs:
            sizes.append(img.size)
        return sizes

    def main(self):
        data = self.getRGB()

        self.Meta.write(path_file=self.pfile)
        self.Meta.write(n_columns=str(len(data[0])))

        pman = PathManager()
        pman.cpaths()

        return data

    def getRGB(self):
        rgb_vals = []
        n = 0
        mean_pixels = self.mean_pixels()
        for im_pixels in mean_pixels:
            print("Reading image ", n, " out of ", len(mean_pixels))
            rgb_vals.append(im_pixels)
            self.rgb_vals.append(im_pixels)
        return self.rgb_vals


class ImageDataWriter:
    def __init__(self, data, fname):
        self.raw_data = data
        self.fname = fname
        self.Meta = MetaData(Sess().read())

    def main(self):
        for image_data in self.raw_data:
            self.writeCSV(image_data)
        print(self.fname)
        self.Meta.write(data_path=os.getcwd()+'/'+self.fname)
        self.Meta.write(n_classes='0')
        self.Meta.write(trainable='False')

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

        self.Meta = MetaData(Sess().read())
        self.Reader = Reader(self.labels_path)

        self.labels = self.Reader.clean_read()

    def clabels(self):
        unique_labels = []
        c = 0
        for label_n in range(len(self.labels)):
            print(unique_labels)
            if label_n == 0:
                unique_labels.append(self.labels[label_n])
                c += 1
            else:
                unique_labels.append(self.labels[label_n])
                if unique_labels[c-1] == unique_labels[c]:
                    del unique_labels[c]
                else:
                    c += 1
                    continue

        return str(len(unique_labels))

    def main(self):
        print("Began with TrainingDataWriting...")
        print(len(self.input_data), "Input data length")
        for imn in range(len(self.input_data)):
            print(imn, "Image-n")
            self.input_data[imn].append(self.labels[imn])
            self.writeCSV(self.input_data[imn])
        self.Meta.write(data_path=os.getcwd()+self.fname)
        self.Meta.write(n_classes=self.clabels())
        self.Meta.write(trainable='True')

    def writeCSV(self, img):
        with open(self.fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)


i = ImageLoader('datasets/fruits_ALP/paths.txt')
data = i.main()
itdw = ImageDataWriter(data, 'data/unclassified/fruits_ALP/data3.csv')
itdw.main()
