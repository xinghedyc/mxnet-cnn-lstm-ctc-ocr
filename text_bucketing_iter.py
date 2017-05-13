import sys
#sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx
import time
import cv2
import random


def default_read_content(path):
    with open(path) as ins:
        content = ins.read()
        content = content.split('\n')
        content.pop()
        return content


def get_path_from_content(content, num_label):
    paths = []
    for line in content:
        path = line.split('\t')[-1]
        paths.append(path)
    return paths


def get_label_from_content(content, num_label):
    labels = []
    for line in content:
        label = line.split('\t')[1:-1]
        labels.append(label)
    #print (labels)
    return labels


def get_image_batch(paths, data_root):

    data = []
    base_hight = 32
    max_ratio = 10
    for path in paths:
        img = cv2.imread(data_root + '/' + path)
        shape = img.shape
        hight = shape[0]
        width = shape[1]
        ratio = (1.0 * width / hight)
        if ratio > max_ratio:
            ratio = max_ratio
        if ratio < 1:
            ratio = 1
        img = cv2.resize(img, (int(32 * ratio), 32))
        hight = 32
        width = int(32 * ratio)
        assert hight == base_hight

        img = np.transpose(img, (2, 0, 1))
        if width % hight != 0:

            padding_ratio = (min(int(ratio + 1), max_ratio))
            new_img = np.zeros((3, base_hight, base_hight * padding_ratio))
            for i in range(3):
                padding_value = int(np.mean(img[i][:][-1]))
                z = np.ones((base_hight, base_hight *
                             padding_ratio - width)) * padding_value
                new_img[i] = np.hstack((img[i], z))
            data.append(new_img)
        else:
            data.append(img)
    return np.array(data)


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class TextIter(mx.io.DataIter):
    def __init__(self, path, data_root, batch_size,
                 init_states, num_label, data_name='data', label_name='label',
                 get_image_function=None, read_content=None, data_shape=[32, 320], buckets=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40]):
        super(TextIter, self).__init__()
        if get_image_function == None:
            self.get_image_function = get_image_batch
        if read_content == None:
            self.read_content = default_read_content
        self.data_root = data_root
        self.content = self.read_content(path)
        print (path + 'records number : ', len(self.content))
        self.num_label = num_label
        self.imagepaths = get_path_from_content(self.content, num_label)
        self.imagelabels = get_label_from_content(self.content, num_label)
        self.default_bucket_key = max(buckets)
        self.factor = 4
        self.imagepaths = np.array(self.imagepaths)
        self.imagelabels = np.array(self.imagelabels)
        self.bucket_images, self.bucket_labels, self.data_plan = self.make_buckets(
            buckets, data_root, batch_size)
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [
            ('data', (batch_size, 3, data_shape[0], data_shape[1]))] + init_states
        self.provide_label = [('label', (batch_size, self.num_label))]
        self.all_idx = range(len(self.content))
        self.current = 0
        self.batch_size = batch_size
        self.size = len(self.data_plan)
        random.shuffle(self.data_plan)

    def make_buckets(self, buckets, data_root, batch_size):
        print ("making buckets")
        buckets_len = len(buckets)
        bucket_images = []
        bucket_labels = []
        for i in range(buckets_len):
            bucket_images.append([])
            bucket_labels.append([])
        data_plan = []
        max_ratio = 10
        for label_idx, img in enumerate(self.imagepaths):
            image = cv2.imread(data_root + '/' + img)
            shape = image.shape
            hight = shape[0]
            width = shape[1]
            ratio = (1.0 * width / hight)
            if ratio > max_ratio:
                ratio = max_ratio
            if ratio < 1:
                ratio = 1
            hight = 32
            width = int(32 * ratio)
            if width % hight != 0:
                ratio = min(int(ratio + 1), max_ratio)
            else:
                ratio = int(ratio)
            bucket_images[ratio - 1].append(img)
            bucket_labels[ratio - 1].append(self.imagelabels[label_idx])
        #print (len(bucket_images))
        for bucket_idx, i in enumerate(bucket_images):
            length_bucket = len(i)
            print ("bucket " + " length :", length_bucket)
            for idx in range(length_bucket / batch_size):
                data_plan.append([bucket_idx, idx])

        return bucket_images, bucket_labels, data_plan

    def iter_next(self):
        return self.current < self.size

    def next(self):
        if self.iter_next():
            start = time.time()
            i = self.current
            init_state_names = [x[0] for x in self.init_states]
            # idx=self.all_idx[int(i*self.batch_size):int(i*self.batch_size+self.batch_size)]
            current_batch = self.data_plan[i]
            buck_idx = current_batch[0]
            img_idx = current_batch[1]
            data = self.get_image_function(
                self.bucket_images[buck_idx][img_idx * self.batch_size:img_idx * self.batch_size + self.batch_size], self.data_root)
            label = self.bucket_labels[buck_idx][img_idx *
                                                 self.batch_size:img_idx * self.batch_size + self.batch_size]
            data_all = [mx.nd.array(data)] + self.init_state_arrays
            # print label
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']
            data_batch = SimpleBatch(
                data_names, data_all, label_names, label_all, (buck_idx + 1) * self.factor)
            self.current += 1
            end = time.time()

            return data_batch

        else:
            raise StopIteration

    def reset(self):
        self.current = 0
        random.shuffle(self.data_plan)
