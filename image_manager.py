from PIL import Image
from pcontrol import *


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
        for img_path in reader.clean_read():
            img = Image.open(img_path)
            self.imgs.append(img)
        return True

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
        self.Meta.write(path_file=self.pfile)

        pman = PathManager()
        pman.cpaths()

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
        self.Meta = MetaData(Sess().read())

    def main(self):
        for image_data in self.raw_data:
            self.writeCSV(image_data)
        print(self.fname)
        self.Meta.write(data_path=os.getcwd()+'/'+self.fname)

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

        self.Meta = MetaData(Sess().read())

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
        self.Meta.write(data_path=os.getcwd()+self.fname)

    def writeCSV(self, img):
        with open(self.fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)


i = ImageLoader('test.txt')
data = i.main()
itdw = ImageDataWriter(data, 'data/unclassified5.csv')
itdw.main()
