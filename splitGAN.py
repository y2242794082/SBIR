from __future__ import print_function, division
from keras import backend as K
from scipy.io import loadmat
from functools import partial
import tensorflow as tf
from scipy import misc
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.merge import _Merge
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import bitwise
import h5py

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

sketch_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/disentangle/v2/sketch.mat'
edge_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/disentangle/v2/edge.mat'
stroke_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/disentangle/v2/stroke.mat'


sketch_test_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/sketch_test.mat'
edge_test_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/sketch0_test.mat'
stroke_test_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/clothes_v3/stroke_test.mat'
# sketch_test_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/shoes/sketch_test.mat'
# edge_test_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/shoes/sketch0_test.mat'
# stroke_test_dir = '/home/yang/PycharmProjects/MY_FG_SBIR/data/shoes/stroke_test.mat'

train_sketch = loadmat(sketch_dir)["data"]
train_edge = loadmat(edge_dir)["data"]
train_stroke = loadmat(stroke_dir)["data"]
test_sketch = loadmat(sketch_test_dir)["data"]
test_edge = loadmat(edge_test_dir)["data"]
test_stroke = loadmat(stroke_test_dir)["data"]


def add_data(data):
    return np.vstack([data, data[:27, :, :]])


train_sketch = add_data(train_sketch)
train_edge = add_data(train_edge)
train_stroke = add_data(train_stroke)


# train_sketch = h5py.File(sketch_dir)["data"][:]
# train_edge = h5py.File(edge_dir)["data"][:]
# train_sketch = np.reshape(train_sketch, (5180, 256, 256))
# train_edge = np.reshape(train_edge, (5180, 256, 256))

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((16, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class Minusimg(_Merge):
    def _merge_function(self, inputs):
        return K.clip(inputs[0] - inputs[1], 0, 1)

class myGAN():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.gf_e = 32
        self.gf_s = 32
        self.df_e = 64
        self.df_s = 64

        self.generator_edge = self.build_generator_edge()
        self.generator_stroke = self.build_generator_stroke()
        self.discriminator_edge = self.build_discriminator_edge()
        self.discriminator_stroke = self.build_discriminator_stroke()

        optimizer_g = Adam(0.00005)
        optimizer_d = Adam(0.0001)
        self.n_critic = 5

        # 训练判别器时,冻结生成器
        self.generator_edge.trainable = False
        self.generator_stroke.trainable = False
        self.discriminator_stroke.trainable = False
        # self.discriminator_edge.trainable = False

        # 真实数据输入
        sketch_img = Input(shape=self.img_shape)
        edge_img = Input(shape=self.img_shape)
        stroke_img = Input(shape=self.img_shape)

        # 噪声输入
        # z_disc = Input(shape=(self.latent_dim,))
        fake_img_edge = self.generator_edge(sketch_img)
        fake_img_stroke = self.generator_stroke(sketch_img)

        # 判断样本是真是假
        fake_edge = self.discriminator_edge(fake_img_edge)
        valid_edge = self.discriminator_edge(edge_img)
        fake_stroke = self.discriminator_stroke(fake_img_stroke)
        valid_stroke = self.discriminator_stroke(stroke_img)

        # 构建权重平均图像
        interpolated_img_edge = RandomWeightedAverage()([edge_img, fake_img_edge])
        interpolated_img_stroke = RandomWeightedAverage()([stroke_img, fake_img_stroke])
        # 得到验证结果
        validity_interpolated_edge = self.discriminator_edge(interpolated_img_edge)
        validity_interpolated_stroke = self.discriminator_stroke(interpolated_img_stroke)

        # 构建惩罚项损失函数
        partial_gp_loss_edge = partial(self.gradient_penalty_loss,
                                       averaged_samples=interpolated_img_edge)
        partial_gp_loss_stroke = partial(self.gradient_penalty_loss,
                                         averaged_samples=interpolated_img_stroke)
        partial_gp_loss_edge.__name__ = 'gradient_penalty'  # Keras requires function names
        partial_gp_loss_stroke.__name__ = 'gradient_penalty'  # Keras requires function names

        self.discriminator_model = Model(inputs=[sketch_img, edge_img, stroke_img],
                                         outputs=[valid_edge, fake_edge, validity_interpolated_edge, valid_stroke,
                                                  fake_stroke, validity_interpolated_stroke])
        self.discriminator_model.compile(loss=[self.wasserstein_loss,
                                               self.wasserstein_loss,
                                               partial_gp_loss_edge,
                                               self.wasserstein_loss,
                                               self.wasserstein_loss,
                                               partial_gp_loss_stroke, ],
                                         optimizer=optimizer_d,
                                         loss_weights=[1, 1, 10, 0, 0, 0])

        # 训练生成器，固定判别器
        self.generator_edge.trainable = True
        self.generator_stroke.trainable = False
        self.discriminator_edge.trainable = False
        self.discriminator_stroke.trainable = False

        # 噪声
        sketch_img = Input(shape=self.img_shape)

        img_edge = self.generator_edge(sketch_img)
        img_stroke = self.generator_stroke(sketch_img)

        fake_edge = self.discriminator_edge(img_edge)
        fake_stroke = self.discriminator_stroke(img_stroke)

        self.generator_model = Model(inputs=[sketch_img],
                                     outputs=[img_edge, fake_edge, img_stroke, fake_stroke])
        self.generator_model.compile(
            loss=[self.dice_loss, self.wasserstein_loss, self.dice_loss, self.wasserstein_loss],
            optimizer=optimizer_g, loss_weights=[10, 1, 0, 0])

        # 测试
        self.discriminator_edge.trainable = False
        self.generator_edge.trainable = False
        self.discriminator_stroke.trainable = False
        self.generator_stroke.trainable = False

        sketch_img = Input(shape=self.img_shape)

        predict_edge = self.generator_edge(sketch_img)
        predict_stroke = self.generator_stroke(sketch_img)

        edge_pred = Minusimg()([predict_edge, sketch_img])
        stroke_pred = Minusimg()([predict_stroke, sketch_img])

        self.test_model = Model(inputs=[sketch_img], outputs=[predict_edge, predict_stroke])
        self.test_model.compile(loss=[self.dice_loss, self.dice_loss], optimizer=optimizer_g, loss_weights=[1, 0])

    def test_mAP(self, y_true, y_pred, smooth=0.001,alpha=10):
        intersection = alpha*K.sum((y_true) * (y_pred), axis=[1, 2, 3])
        union = alpha*K.sum((y_true), axis=[1, 2, 3]) + K.sum((y_pred), axis=[1, 2, 3])
        return K.mean((intersection + smooth) / (union + smooth), axis=0)

    def dice_loss(self, y_true, y_pred, smooth=0.0001):
        intersection = K.sum((1 - y_true) * (1 - y_pred), axis=[1, 2, 3])
        union = K.sum((1 - y_true), axis=[1, 2, 3]) + K.sum((1 - y_pred), axis=[1, 2, 3])
        return 1 - K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator_edge(self):

        def conv2d(layer_input, filters, f_size=3, padding='same', strides=2):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding=padding)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = BatchNormalization()(d)
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding=padding)(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = BatchNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0, padding='same', strides=2):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=strides, padding=padding, activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization()(u)
            u = UpSampling2D(size=2)(u)
            u = Concatenate()([u, skip_input])
            return u

            # Image input

        d0 = Input(shape=self.img_shape)

        # Downsampling
        # d1 = conv2d(d0, self.gf_e)
        # d2 = conv2d(d1, self.gf_e * 2)
        # d3 = conv2d(d2, self.gf_e * 4)
        # d4 = conv2d(d3, self.gf_e * 8)
        # d5 = conv2d(d4, self.gf_e * 16)
        # d6 = conv2d(d5, self.gf_e * 32)
        # d7 = conv2d(d6, self.gf_e * 64)
        #
        # u1 = deconv2d(d7, d6, self.gf_e * 32)
        # u2 = deconv2d(u1, d5, self.gf_e * 16)
        # u3 = deconv2d(u2, d4, self.gf_e * 8)
        # u4 = deconv2d(u3, d3, self.gf_e * 4)
        # u5 = deconv2d(u4, d2, self.gf_e * 2)
        # u6 = deconv2d(u5, d1, self.gf_e * 1)
        d1 = conv2d(d0, self.gf_e)
        d2 = conv2d(d1, self.gf_e * 2)
        d3 = conv2d(d2, self.gf_e * 4)
        d4 = conv2d(d3, self.gf_e * 8)
        d5 = conv2d(d4, self.gf_e * 16)
        d6 = conv2d(d5, self.gf_e * 32)

        u1 = deconv2d(d6, d5, self.gf_e * 16)
        u2 = deconv2d(u1, d4, self.gf_e * 8)
        u3 = deconv2d(u2, d3, self.gf_e * 4)
        u4 = deconv2d(u3, d2, self.gf_e * 2)
        u5 = deconv2d(u4, d1, self.gf_e * 1)

        u7 = UpSampling2D(size=2)(u5)
        u7 = BatchNormalization()(u7)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u7)

        return Model(d0, output_img)

    def build_generator_stroke(self):

        def conv2d(layer_input, filters, f_size=3, padding='same', strides=2):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding=padding)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = BatchNormalization()(d)
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding=padding)(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = BatchNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0, padding='same', strides=2):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=strides, padding=padding, activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization()(u)
            u = UpSampling2D(size=2)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        # d1 = conv2d(d0, self.gf_s)
        # d2 = conv2d(d1, self.gf_s * 2)
        # d3 = conv2d(d2, self.gf_s * 4)
        # d4 = conv2d(d3, self.gf_s * 8)
        # d5 = conv2d(d4, self.gf_s * 16)
        #
        # u1 = deconv2d(d5, d4, self.gf_s * 8)
        # u2 = deconv2d(u1, d3, self.gf_s * 4)
        # u3 = deconv2d(u2, d2, self.gf_s * 2)
        # u4 = deconv2d(u3, d1, self.gf_s * 1)
        d1 = conv2d(d0, self.gf_e)
        d2 = conv2d(d1, self.gf_e * 2)
        d3 = conv2d(d2, self.gf_e * 4)
        d4 = conv2d(d3, self.gf_e * 8)
        d5 = conv2d(d4, self.gf_e * 16)
        d6 = conv2d(d5, self.gf_e * 32)

        u1 = deconv2d(d6, d5, self.gf_e * 16)
        u2 = deconv2d(u1, d4, self.gf_e * 8)
        u3 = deconv2d(u2, d3, self.gf_e * 4)
        u4 = deconv2d(u3, d2, self.gf_e * 2)
        u5 = deconv2d(u4, d1, self.gf_e * 1)

        u7 = UpSampling2D(size=2)(u5)
        u7 = BatchNormalization()(u7)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u7)

        return Model(d0, output_img)

    def build_discriminator_edge(self):

        def d_layer(layer_input, filters, f_size=5, normalization=True, strides=2):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = BatchNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df_e, normalization=False)
        d2 = d_layer(d1, self.df_e * 2)
        d3 = d_layer(d2, self.df_e * 4)
        d4 = d_layer(d3, self.df_e * 8)
        d5 = d_layer(d4, self.df_e * 16)

        validity = Conv2D(1, kernel_size=3, strides=1, padding='same')(d5)
        validity = BatchNormalization()(validity)
        validity = Flatten()(validity)
        validity = Dense(1, activation='tanh')(validity)

        return Model(img, validity)

    def build_discriminator_stroke(self):

        def d_layer(layer_input, filters, f_size=5, normalization=True, strides=2):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = BatchNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df_s, normalization=False)
        d2 = d_layer(d1, self.df_s * 2)
        d3 = d_layer(d2, self.df_s * 4)
        d4 = d_layer(d3, self.df_s * 8)
        d5 = d_layer(d4, self.df_s * 16)
        # d6 = d_layer(d5, self.df_s * 16)

        validity = Conv2D(1, kernel_size=3, strides=1, padding='same')(d5)
        validity = BatchNormalization()(validity)
        validity = Flatten()(validity)
        validity = Dense(1, activation='tanh')(validity)

        return Model(img, validity)

    def test(self, epochs, batch_size):
        sketch_test = test_sketch
        edge_test = test_edge
        stroke_test = test_stroke
        sketch_test = sketch_test.astype(np.float32) / 255
        sketch_test = np.expand_dims(sketch_test, axis=3)
        edge_test = edge_test.astype(np.float32) / 255
        edge_test = np.expand_dims(edge_test, axis=3)
        stroke_test = stroke_test.astype(np.float32) / 255
        stroke_test = np.expand_dims(stroke_test, axis=3)

        # #选择最好的权重
        # for i in range(158,214+1):
        #     self.generator_edge.load_weights('/home/yang/PycharmProjects/MySBIR/model/DISENTANGLE/generator/g_e_%d.h5'%(100*i))
        #     # self.generator_stroke.load_weights('/home/yang/PycharmProjects/MySBIR/model/DISENTANGLE/generator/g_s_%d.h5'%(100*i))
        #
        #     g_loss_total = 0
        #     for epoch in range(epochs):
        #         idx = range(epoch * batch_size, (epoch + 1) * batch_size)
        #         sketch_imgs = sketch_test[idx]
        #         edge_imgs = edge_test[idx]
        #         stroke_imgs = stroke_test[idx]
        #         edge_gt = K.clip(edge_imgs - sketch_imgs, 0, 1)
        #         stroke_gt = K.clip(stroke_imgs - sketch_imgs, 0, 1)
        #         g_loss = self.test_model.train_on_batch([sketch_imgs], [edge_imgs, stroke_imgs])
        #         g_loss_total = g_loss[0]+g_loss_total
        #
        #     print("%d [G loss: %f]" % (i, g_loss_total / epochs))

        # 输出测试图片
        # self.generator_edge.load_weights('/home/yang/PycharmProjects/MY_FG_SBIR/DISENTANGLE/g_e_18000.h5')
        # self.generator_stroke.load_weights('/home/yang/PycharmProjects/MY_FG_SBIR/DISENTANGLE/g_s_18000.h5')
        self.generator_edge.load_weights(
            '/home/yang/PycharmProjects/MY_FG_SBIR/DISENTANGLE/g_e_21300.h5')
        self.generator_stroke.load_weights('/home/yang/PycharmProjects/MY_FG_SBIR/DISENTANGLE/g_s_23600.h5')
        g_loss_total = 0
        for epoch in range(epochs):
            idx = range(epoch * batch_size, (epoch + 1) * batch_size)
            sketch_imgs = sketch_test[idx]
            edge_imgs = edge_test[idx]
            stroke_imgs = stroke_test[idx]
            edge_gt = edge_imgs - sketch_imgs
            stroke_gt = stroke_imgs - sketch_imgs
            g_loss = self.test_model.train_on_batch([sketch_imgs], [edge_imgs, stroke_imgs])
            g_loss_total = g_loss[0] + g_loss_total
            if epoch % 1 == 0:
                self.sample_images(epoch, sketch_imgs, batch_size, 'test')

        print("[G loss: %f]" % (g_loss_total / epochs))

    def train(self, epochs, batch_size, sample_interval=50, save_interval=100):
        self.generator_edge.load_weights('/home/yang/PycharmProjects/MY_FG_SBIR/DISENTANGLE/g_e_6500.h5')
        # self.generator_stroke.load_weights('/home/yang/PycharmProjects/MY_FG_SBIR/DISENTANGLE/g_s_1500.h5')
        self.discriminator_edge.load_weights(
            '/home/yang/PycharmProjects/MY_FG_SBIR/DISENTANGLE/d_e_6500.h5')
        # self.discriminator_stroke.load_weights(
        #     '/home/yang/PycharmProjects/MY_FG_SBIR/DISENTANGLE/d_s_1500.h5')

        # Load the dataset
        sketch_train = train_sketch
        edge_train = train_edge
        stroke_train = train_stroke

        # Rescale -1 to 1
        sketch_train = sketch_train.astype(np.float32) / 255
        edge_train = edge_train.astype(np.float32) / 255
        stroke_train = stroke_train.astype(np.float32) / 255
        sketch_train = np.expand_dims(sketch_train, axis=3)
        edge_train = np.expand_dims(edge_train, axis=3)
        stroke_train = np.expand_dims(stroke_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for i in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, sketch_train.shape[0], batch_size)
                sketch_imgs = sketch_train[idx]
                edge_imgs = edge_train[idx]
                stroke_imgs = stroke_train[idx]
                # Train the critic
                d_loss = self.discriminator_model.train_on_batch([sketch_imgs, edge_imgs, stroke_imgs],
                                                                 [valid, fake, dummy, valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            idx = np.random.randint(0, sketch_train.shape[0], batch_size)
            sketch_imgs = sketch_train[idx]
            edge_imgs = edge_train[idx]
            stroke_imgs = stroke_train[idx]
            g_loss = self.generator_model.train_on_batch([sketch_imgs], [edge_imgs, valid, stroke_imgs, valid])

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, sketch_imgs, batch_size, 'train')

            if epoch % save_interval == 0:
                self.generator_edge.save(
                    '/home/yang/PycharmProjects/MySBIR/model/DISENTANGLE/generator/g_e_%d.h5' % (epoch+6500))
                # self.generator_stroke.save(
                #     '/home/yang/PycharmProjects/MySBIR/model/DISENTANGLE/generator/g_s_%d.h5' % (epoch))
                self.discriminator_edge.save(
                    '/home/yang/PycharmProjects/MySBIR/model/DISENTANGLE/discriminator/d_e_%d.h5' % (epoch+6500))
                # self.discriminator_stroke.save(
                #     '/home/yang/PycharmProjects/MySBIR/model/DISENTANGLE/discriminator/d_s_%d.h5' % (epoch))

    def sample_images(self, epoch, input, batch_size, flag):
        r, c = 4, 4
        gen_imgs_edge = self.generator_edge.predict(input)
        # print(gen_imgs_edge[0][120])
        gen_imgs_stroke = self.generator_stroke.predict(input)

        if flag == 'test':
            for i in range(batch_size):
                test_imgs_edge = gen_imgs_edge[i]
                test_imgs_edge = np.reshape(test_imgs_edge, (256, 256))
                test_imgs_stroke = gen_imgs_stroke[i]
                test_imgs_stroke = np.reshape(test_imgs_stroke, (256, 256))
                misc.imsave("/home/yang/PycharmProjects/MySBIR/data/results/test/edge/disentangle_edge_%03d.png" % (
                        epoch * batch_size + i), test_imgs_edge)
                misc.imsave("/home/yang/PycharmProjects/MySBIR/data/results/test/stroke/disentangle_stroke_%03d.png" % (
                        epoch * batch_size + i), test_imgs_stroke)

        else:
            fig1, axs1 = plt.subplots(r, c)
            fig2, axs2 = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs1[i, j].imshow(gen_imgs_edge[cnt, :, :, 0])
                    axs1[i, j].axis('off')
                    axs2[i, j].imshow(gen_imgs_stroke[cnt, :, :, 0])
                    axs2[i, j].axis('off')
                    cnt += 1
            fig1.savefig("/home/yang/PycharmProjects/MySBIR/data/results/edge/disentangle_edge_%d.png" % (
                    epoch+6500))
            # fig2.savefig(
            #     "/home/yang/PycharmProjects/MySBIR/data/results/stroke/disentangle_stroke_%d.png" % (
            #                 epoch))
        plt.close()


if __name__ == '__main__':
    gan = myGAN()
    # gan.train(epochs=200000, batch_size=16, sample_interval=20)
    # gan.test(epochs=6, batch_size=42)
    gan.test(epochs=25, batch_size=5)