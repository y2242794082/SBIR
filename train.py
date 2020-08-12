import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.layers import BatchNormalization
import numpy as np
from scipy.io import loadmat, savemat
import scipy.spatial.distance as ssd
from sbir_sampling import triplet_sampler_asy
from tensorflow.contrib.slim import nets
from sbir_util import *
from ops import spatial_softmax, reshape_feats
import os, errno
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

from tensorflow.compat.v1 import ConfigProto

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

NET_ID = 0  # 0 for step3 pre-trained model, 1 for step2 pre-trained model


def attentionNet(inputs, pool_method='sigmoid'):
    assert (pool_method in ['sigmoid', 'softmax'])
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=True):
        net = slim.conv2d(inputs, 256, [1, 1], padding='SAME', scope='conv1_0')
        if pool_method == 'sigmoid':
            net = slim.conv2d(net, 1, [1, 1], activation_fn=tf.nn.sigmoid, scope='conv2_0')
        else:
            net = slim.conv2d(net, 1, [1, 1], activation_fn=None, scope='conv2_0')
            net = spatial_softmax(net)
    return net


def sketch_a_net_sbir(inputs, trainable):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=False):
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            # x = tf.reshape(inputs, shape=[-1, 225, 225, 1])
            conv1 = slim.conv2d(inputs, 64, [15, 15], 3, scope='conv1_s1')
            conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], scope='conv2_s1')
            conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
            conv3 = slim.conv2d(conv2, 256, [3, 3], padding='SAME', scope='conv3_s1')
            conv4 = slim.conv2d(conv3, 256, [3, 3], padding='SAME', scope='conv4_s1')
            conv5 = slim.conv2d(conv4, 256, [3, 3], padding='SAME', scope='conv5_s1')  # trainable=trainable
            conv5 = slim.max_pool2d(conv5, [3, 3], scope='pool3')
            conv5 = slim.flatten(conv5)
            fc6 = slim.fully_connected(conv5, 512, trainable=trainable, scope='fc6_s1')
            fc7 = slim.fully_connected(fc6, 256, activation_fn=None, trainable=trainable, scope='fc7_sketch')
            fc7 = tf.nn.l2_normalize(fc7, dim=1)
    return fc7


def sketch_a_net_dssa(inputs, trainable):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=False):  # when test 'trainable=True', don't forget to change it
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            # x = tf.reshape(inputs, shape=[-1, 225, 225, 1])
            conv1 = slim.conv2d(inputs, 64, [15, 15], 3, scope='conv1_s1')
            conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], scope='conv2_s1')
            conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
            conv3 = slim.conv2d(conv2, 256, [3, 3], padding='SAME', scope='conv3_s1')
            conv4 = slim.conv2d(conv3, 256, [3, 3], padding='SAME', scope='conv4_s1')
            conv5 = slim.conv2d(conv4, 256, [3, 3], padding='SAME', trainable=trainable, scope='conv5_s1')
            conv5 = slim.max_pool2d(conv5, [3, 3], scope='pool3')
            # residual attention
            att_mask = attentionNet(conv5, 'softmax')
            att_map = tf.multiply(conv5, att_mask)
            att_f = tf.add(conv5, att_map)
            attended_map = tf.reduce_sum(att_f, reduction_indices=[1, 2])
            attended_map = tf.nn.l2_normalize(attended_map, dim=1)
            att_f = slim.flatten(att_f)
            fc6 = slim.fully_connected(att_f, 512, trainable=True, scope='fc6_s1')
            f6 = tf.nn.l2_normalize(fc6, dim=1)
            fc7 = slim.fully_connected(fc6, 256, activation_fn=None, trainable=True, scope='fc7_sketch')
            fc7 = tf.nn.l2_normalize(fc7, dim=1)
            final_feature_map = tf.concat([fc7, attended_map], 1)
    return final_feature_map

def handle_color_net(inputs, trainable):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=trainable,
                        reuse=tf.AUTO_REUSE):  # when test 'trainable=True', don't forget to change it
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            conv1 = slim.conv2d(inputs, 128, [5, 5], 3, scope='conv1_c1')
            # conv1 = slim.batch_norm(conv1, reuse=tf.AUTO_REUSE, scope='bn1')
            conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
            # conv2 = slim.conv2d(conv1, 256, [3, 3], scope='conv2_c2')
            # conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
            conv2 = slim.flatten(conv1)
            fc = slim.fully_connected(conv2, 1024, scope='fc_final')
            fc_final = tf.nn.l2_normalize(fc, dim=1)
    return fc_final

# def handle_color_net(inputs, trainable):
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                         activation_fn=tf.nn.relu,
#                         weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
#                         weights_regularizer=slim.l2_regularizer(0.0005),
#                         trainable=trainable,
#                         reuse=tf.AUTO_REUSE):  # when test 'trainable=True', don't forget to change it
#         with slim.arg_scope([slim.conv2d], padding='VALID'):
#             conv1 = slim.conv2d(inputs, 128, [3, 3], 3, scope='conv1_c1')
#             conv1 = BatchNormalization()(conv1)
#             conv2 = slim.conv2d(inputs, 128, [3, 3], scope='conv2_c2')
#             conv2 = BatchNormalization()(conv2)
#             conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool1')
#             # conv2 = slim.conv2d(conv1, 256, [3, 3], scope='conv2_c2')
#             # conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
#             conv2 = slim.flatten(conv2)
#             fc = slim.fully_connected(conv2, 512, scope='fc_final')
#             fc_final = tf.nn.l2_normalize(fc, dim=1)
#     return fc_final
# def handle_color_net(inputs, trainable):
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                         activation_fn=tf.nn.relu,
#                         weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
#                         weights_regularizer=slim.l2_regularizer(0.0005),
#                         trainable=trainable,
#                         reuse=tf.AUTO_REUSE):  # when test 'trainable=True', don't forget to change it
#         with slim.arg_scope([slim.conv2d], padding='SAME'):
#             conv1 = slim.conv2d(inputs, 128, [3, 3], 3, scope='conv1_c1')
#             conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
#             conv1 = slim.flatten(conv1)
#             fc1 = slim.fully_connected(conv1, 512, scope='fc2')
#             fc1 = tf.nn.l2_normalize(fc1, dim=1)
#             conv2 = slim.conv2d(inputs, 128, [5, 5], 3, scope='conv1_c2')
#             conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
#             conv2 = slim.flatten(conv2)
#             fc2 = slim.fully_connected(conv2, 512, scope='fc2')
#             fc2 = tf.nn.l2_normalize(fc2, dim=1)
#             # conv3 = slim.conv2d(inputs, 128, [7, 7], 3, scope='conv1_c3')
#             # conv3 = slim.max_pool2d(conv3, [3, 3], scope='pool3')
#             # conv3 = slim.flatten(conv3)
#             # fc3 = slim.fully_connected(conv3, 256, scope='fc3')
#             # fc3 = tf.nn.l2_normalize(fc3, dim=1)
#             fc_final = tf.concat([fc1, fc2], 1)
#     return fc_final

def init_variables(model_file='/home/yang/PycharmProjects/MySBIR/model/SBIR-shoe/sketchnet_init.npy'):
    if NET_ID==0:
        pretrained_paras = ['conv1_s1', 'conv2_s1', 'conv3_s1', 'conv4_s1', 'conv5_s1', 'fc6_s1', 'fc7_sketch']
    else:
        pretrained_paras = ['conv1_s1', 'conv2_s1', 'conv3_s1', 'conv4_s1', 'conv5_s1', 'fc6_s1']
    d = np.load(model_file,encoding="latin1").item()
    init_ops = []  # a list of operations
    for var in tf.global_variables():
        for w_name in pretrained_paras:
            if w_name in var.name:
                print('Initialise var %s with weight %s' % (var.name, w_name))
                try:
                    if 'weights' in var.name:
                        # using assign(src, dst) to assign the weights of pre-trained model to current network
                        # init_ops.append(var.assign(d[w_name+'/weights:0']))
                        init_ops.append(var.assign(d[w_name]['weights']))
                    elif 'biases' in var.name:
                        # init_ops.append(var.assign(d[w_name+'/biases:0']))
                        init_ops.append(var.assign(d[w_name]['biases']))
                except KeyError:
                     if 'weights' in var.name:
                        # using assign(src, dst) to assign the weights of pre-trained model to current network
                        init_ops.append(var.assign(d[w_name+'/weights:0']))
                        # init_ops.append(var.assign(d[w_name]['weights']))
                     elif 'biases' in var.name:
                        init_ops.append(var.assign(d[w_name+'/biases:0']))
                        # init_ops.append(var.assign(d[w_name]['biases']))
                except:
                     if 'weights' in var.name:
                        # using assign(src, dst) to assign the weights of pre-trained model to current network
                        init_ops.append(var.assign(d[w_name][0]))
                        # init_ops.append(var.assign(d[w_name]['weights']))
                     elif 'biases' in var.name:
                        init_ops.append(var.assign(d[w_name][1]))
                        # init_ops.append(var.assign(d[w_name]['biases']))
    return init_ops


# def init_variables(model_file='/home/yang/PycharmProjects/MY_FG_SBIR/SBIR/model/shoes/DSSA/model-iter0.npy'):
#     # pretrained_paras = ['conv1_s1', 'conv2_s1', 'conv3_s1', 'conv4_s1', 'conv5_s1', 'fc6_s1', 'fc7_sketch',
#     # 'att_conv1', 'att_conv2']
#     d = np.load(model_file).item()
#     pretrained_paras = d.keys()
#     # print(pretrained_paras)
#     init_ops = []  # a list of operations
#     # for var in tf.global_variables():
#     for var in tf.global_variables():
#         for w_name in pretrained_paras:
#             if w_name in var.name:
#                 # print('Initialise var %s with weight %s' % (var.name, w_name))
#                 init_ops.append(var.assign(d[w_name]))
#     return init_ops


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    d = tf.square(tf.sub(x, y))
    d = tf.sqrt(tf.reduce_sum(d))  # What about the axis ???
    return d


def square_distance(x, y):
    return tf.reduce_sum(tf.square(x - y), axis=1)


def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):
    with tf.name_scope("triplet_loss"):
        d_p_squared = square_distance(anchor_feature, positive_feature)
        d_n_squared = square_distance(anchor_feature, negative_feature)
        loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)


def triple(inputs, outputs):
    for i in range(3):
        outputs[::, i] = inputs
    return outputs


def main(subset, sketch_dir, image_dir, black_dir, stroke_LPF_dir, triplet_path, triplet_path_color, mean, hard_ratio, batch_size, phase,
         net_model):
    ITERATIONS = 42000
    VALIDATION_TEST = 200
    perc_train = 0.9
    MARGIN = 0.7
    SAVE_STEP = 200
    # model_path = "./model/%s/%s/" % (subset, net_model)
    model_path = "/home/yang/PycharmProjects/MY_FG_SBIR/SBIR/model/shoes/DSSA/"
    resnet_model_path = 'resnet_v2_50_2017_04_14/resnet_v2_50.ckpt'  # Path to the pretrained model
    # resnet_model_path = 'resnet_v2_50_2017_04_14/vgg_16.ckpt'  # test
    pre_trained_model = './model/sketchnet_init.npy'
    pre_step = 0
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Siamease place holders
    train_anchor_data = tf.placeholder(tf.float32, shape=(None, 225, 225, 1), name="anchor")
    train_positive_data = tf.placeholder(tf.float32, shape=(None, 225, 225, 1), name="positive")
    train_negative_data = tf.placeholder(tf.float32, shape=(None, 225, 225, 1), name="negative")
    train_positive_black_data = tf.placeholder(tf.float32, shape=(None, 225, 225, 1), name="positive")
    train_negative_black_data = tf.placeholder(tf.float32, shape=(None, 225, 225, 1), name="negative")
    train_stroke_data = tf.placeholder(tf.float32, shape=(None, 225, 225, 1), name="negative")

    # Creating the architecturek
    if net_model == 'deep_sbir':
        train_anchor = sketch_a_net_sbir(tf.cast(train_anchor_data, tf.float32) - mean, True)
        tf.get_variable_scope().reuse_variables()
        train_positive = sketch_a_net_sbir(tf.cast(train_positive_data, tf.float32) - mean, True)
        train_negative = sketch_a_net_sbir(tf.cast(train_negative_data, tf.float32) - mean, True)
    elif net_model == 'DSSA':
        train_anchor_original = sketch_a_net_dssa(tf.cast(train_anchor_data, tf.float32) - mean, True)
        tf.get_variable_scope().reuse_variables()
        train_positive_original = sketch_a_net_dssa(tf.cast(train_positive_data, tf.float32) - mean, True)
        train_negative_original = sketch_a_net_dssa(tf.cast(train_negative_data, tf.float32) - mean, True)
        train_positive_black = handle_color_net(tf.cast(train_positive_black_data, tf.float32) - mean, True)
        train_negative_black = handle_color_net(tf.cast(train_negative_black_data, tf.float32) - mean, True)
        train_stroke = handle_color_net(tf.cast(train_stroke_data, tf.float32) - mean, True)
        # train_positive_black = sketch_a_net_dssa(tf.cast(train_positive_black_data, tf.float32) - mean, True)
        # train_negative_black = sketch_a_net_dssa(tf.cast(train_negative_black_data, tf.float32) - mean, True)
        # train_stroke = sketch_a_net_dssa(tf.cast(train_stroke_data, tf.float32) - mean, True)
        # train_anchor = my_net(train_anchor_original, train_stroke, True)
        # print(train_anchor_original.shape, train_stroke.shape)
        # train_positive = my_net(train_positive_original, train_positive_black, True)
        # train_negative = my_net(train_negative_original, train_negative_black, True)
    else:
        print('Please define the net_model')

    # init_ops = init_variables('/home/yang/PycharmProjects/MY_FG_SBIR/SBIR/model/shoes/DSSA/model-iter6400.npy')
    init_ops = init_variables()
    loss1, positives1, negatives1 = compute_triplet_loss(train_anchor_original, train_positive_original,
                                                         train_negative_original, MARGIN)
    loss2, positives2, negatives2 = compute_triplet_loss(train_stroke, train_positive_black, train_negative_black,
                                                         MARGIN)
    loss = 0*loss1 + loss2

    # Defining training parameters
    batch = tf.Variable(0)
    learning_rate = 0.001
    data_sampler = triplet_sampler_asy.TripletSamplingLayer()
    data_sampler.setup(sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase)
    data_sampler_stroke = triplet_sampler_asy.TripletSamplingLayer()
    data_sampler_stroke.setup(stroke_LPF_dir, black_dir, triplet_path_color, mean, hard_ratio, batch_size, phase)
    # optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate).minimize(loss,
    #                                                                                            global_step=batch)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=batch)
    # validation_prediction = tf.nn.softmax(lenet_validation)
    # saver = tf.train.Saver(max_to_keep=5)
    dst_path = '/home/yang/PycharmProjects/MySBIR/log'
    model_id = '%s_%s_log.txt' % (subset, net_model)
    filename = dst_path + '/' + model_id

    # use resnet
    checkpoint_exclude_scopes = 'Logits'
    exclusions = None
    if checkpoint_exclude_scopes:
        exclusions = [
            scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)

    saver_restore = tf.train.Saver(var_list=variables_to_restore)
    saver = tf.train.Saver(tf.global_variables())

    # f = open(filename, 'a')
    # Training
    with tf.Session() as session:

        session.run(tf.global_variables_initializer())
        session.run(init_ops)

        for step in range(ITERATIONS):
            f = open(filename, 'a')
            batch_anchor, batch_positive, batch_negative = data_sampler.get_next_batch()
            batch_stroke, batch_positive_black, batch_negative_black = data_sampler_stroke.get_next_batch()

            feed_dict = {train_anchor_data: batch_anchor,
                         train_positive_data: batch_positive,
                         train_negative_data: batch_negative,
                         train_stroke_data: batch_stroke,
                         train_positive_black_data: batch_positive_black,
                         train_negative_black_data: batch_negative_black,
                         }
            # _, l1 = session.run([optimizer1, loss1], feed_dict=feed_dict)
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            # save_path = saver.save(session, model_path, global_step=step)
            print("Iter %d: Loss Train %f" % (step + pre_step, l))
            f.write("Iter " + str(step + pre_step) + ": Loss Train: " + str(l))
            f.write("\n")
            # train_writer.add_summary(summary, step)

            if step % SAVE_STEP == 0:
                str_temp = '%smodel-iter%d.npy' % (model_path, step + pre_step)
                save_dict = {var.name: var.eval(session) for var in tf.global_variables()}
                np.save(str_temp, save_dict)

            # if step % VALIDATION_TEST == 0:
            #     batch_anchor, batch_positive, batch_negative = data_sampler_te.get_next_batch()
            #
            #     feed_dict = {train_anchor_data: batch_anchor,
            #                  train_positive_data: batch_positive,
            #                  train_negative_data: batch_negative
            #                  }
            #
            #     lv = session.run([loss], feed_dict=feed_dict)
            #     # test_writer.add_summary(summary, step)
            #     print("Loss Validation {0}".format(lv))
            #     f.write("Loss Validation: " + str(lv))
            #     f.write("\n")
            # f.close()


if __name__ == '__main__':
    # 'deep_sbir'(the model of cvpr16) or 'DSSA'(the model of iccv17)
    net_model = 'DSSA'
    subset = 'shoes'
    mean = 250.42
    hard_ratio = 0.75
    batch_size = 64
    phase = 'TRAIN'
    phase_te = 'TEST'
    base_path = './data'

    # #cloth
    # sketch_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes/sketch0_train.mat'
    # image_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes/edge_train.mat'
    # triplet_path = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes/clothes_annotation.json'
    # triplet_path_color = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes/clothes_annotation_color.json'
    # stroke_LPF_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes/stroke_LPF_train.mat'
    # black_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes/black_train.mat'

    # cloth
    # sketch_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/sketch0_train.mat'
    # image_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/edge_train.mat'
    # triplet_path = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/clothes_annotation.json'
    # triplet_path_color = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/clothes_annotation.json'
    # stroke_LPF_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/stroke_LPF_train.mat'
    # black_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/black_train.mat'

    # #shoe
    # sketch_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/shoes/sketch0_train.mat'
    # image_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/shoes/edge_train.mat'
    # triplet_path = '/home/yang/PycharmProjects/MY_FG_SBIR/data/shoes/shoes_annotation.json'
    # triplet_path_color = '/home/yang/PycharmProjects/MY_FG_SBIR/data/shoes/shoes_annotation_color.json'
    # stroke_LPF_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/shoes/stroke_LPF_train.mat'
    # black_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/shoes/black_train.mat'

    # ablation/home/yang/MatLabProjects/cloth_test.mat
    sketch_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/sketch_train.mat'
    image_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/edge_train.mat'
    triplet_path = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/clothes_annotation.json'
    triplet_path_color = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/clothes_annotation.json'
    stroke_LPF_dir = '/home/yang/MatLabProjects/cloth_train.mat'
    black_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/black_train.mat'
    main(subset, sketch_dir, image_dir, black_dir, stroke_LPF_dir, triplet_path, triplet_path_color, mean, hard_ratio, batch_size, phase,
         net_model)
