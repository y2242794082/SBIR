"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

import os
import os.path as osp

# add for print log
import datetime
import sys
from tee_print import Tee
try:
    from tabulate import tabulate
except:
    print ("tabulate lib not installed")


pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def get_image(image_path, image_height, image_width, resize_h, resize_w, is_crop=False, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_height, image_width, resize_h, resize_w, is_crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def center_crop(x, crop_h, crop_w, resize_h, resize_w):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_h, resize_w])


def transform(image, image_height, image_width, resize_h, resize_w, is_crop=False):
    # npx : # of pixels width/height of image
    # pdb.set_trace()
    if is_crop:
        cropped_image = center_crop(image, image_height, image_width, resize_h, resize_w)
    else:
        cropped_image = scipy.misc.imresize(image, [image_height, image_width])
    # return np.array(cropped_image)/127.5 - 1.
    return np.array(cropped_image) / 255


def inverse_transform(images):
    return (images + 1.) / 2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append(
                        {"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2 ** (int(layer_idx) + 2), 2 ** (int(layer_idx) + 2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def visualize(sess, dcgan, config, option):
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    elif option == 1:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in range(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
    elif option == 2:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in [random.randint(0, 99) for _ in range(100)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            # z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 3:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in range(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1. / config.batch_size)

        for idx in range(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

            image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
                         for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def mkdir_if_missing(d):
    if not osp.isdir(d):
        os.makedirs(d)


def mkdir_for_models(out_dir, sv_dir, cr_dirs):
    output_dir = os.path.join(out_dir, sv_dir)
    for cr_dir in cr_dirs:
        mkdir_if_missing(os.path.join(output_dir, cr_dir))


def mkdir_for_models_train(out_dir='./', sv_dir = './', chk_dir = 'checkpoints', log_dir = 'logs', ts_dir = 'train_samples'):
    output_dir = os.path.join(out_dir, sv_dir)
    cr_dirs = [chk_dir, log_dir, ts_dir]
    for cr_dir in cr_dirs:
        mkdir_if_missing(os.path.join(output_dir, cr_dir))


def print_log(log_prefix, FLAGS):
    # log direcory and file
    log_dir = './logs/'
    net_name = 'GoogleNet_Person_Reid'
    log_name = log_prefix + '_' + os.uname()[1].split('.')[0] + '_' + datetime.datetime.now().isoformat().split('.')[0].replace('-','_').replace(':', '_') + '_log.txt'
    log_file = os.path.join(log_dir, log_name)
    f = open(log_file, 'w')
    orig_out = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    print ("logging to ", log_file, "...")

    # print header
    print ("===============================================")
    print ("Trainning ", net_name, " in this framework")
    print ("===============================================")

    print ("Tensorflow flags:")

    flag_table = {}
    flag_table['FLAG_NAME'] = []
    flag_table['Value'] = []
    flag_lists = FLAGS.saved_flags.split()
    # print self.FLAGS.__flags
    for attr in flag_lists:
        if attr not in ['saved_flags', 'net_name', 'data_dir']:
            flag_table['FLAG_NAME'].append(attr.upper())
            flag_table['Value'].append(getattr(FLAGS, attr))
    flag_table['FLAG_NAME'].append('NET_NAME')
    flag_table['Value'].append(net_name)
    flag_table['FLAG_NAME'].append('HOST_NAME')
    flag_table['Value'].append(os.uname()[1].split('.')[0])
    try:
        print (tabulate(flag_table, headers="keys", tablefmt="fancy_grid").encode('utf-8'))
    except:
        for attr in flag_lists:
            print ("attr name, ", attr.upper())
            print ("attr value, ", getattr(FLAGS, attr))