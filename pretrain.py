import tensorflow as tf
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import *
from load_data import *
from build_model import *
from loss_function import *

class GANimation_pretrain:
    def __init__(self):
        self.encoder = build_encoder()
        self.att_decoder = build_att_decoder()
        self.img_decoder = build_img_decoder()
        self.discriminator = build_discriminator()
        self.g_opt = Adam(1e-4)
        self.d_opt = Adam(1e-4)
        self.natural_roots_train, self.expression_roots_train = load_ck(train=True)
        self.natural_roots_test, self.expression_roots_test = load_ck(train=False)
        self.total_roots_train, self.label_train = build_pretrain_data(self.natural_roots_train, self.expression_roots_train)
        self.total_roots_test, self.label_test = build_pretrain_data(self.natural_roots_test, self.expression_roots_test)

    def gen_pretrain_step(self, source, label, train=True):
        label = tf.one_hot(label, depth=2)
        source = tf.cast(source, dtype='float32')
        label = tf.cast(label, dtype='float32')
        with tf.GradientTape() as tape:
            feat = self.encoder.call([source, label])
            A = self.att_decoder.call(feat)
            C = self.img_decoder.call(feat)
            gen_imgs = (1 - A) * C + A * source
            #gen_imgs = C
            v_gen, c_gen = self.discriminator.call(gen_imgs)
            loss_adv_g = adversarial_loss(v_gen, True)
            loss_att1, loss_reg1 = att_loss(A)
            loss_cls = classify_loss(label, c_gen)
            loss_img = img_loss(source, gen_imgs)
            loss_g = loss_adv_g + 0.005 * loss_att1 + 0.005 * loss_reg1 + 10 * loss_cls + 10 * loss_img
            #loss_g = loss_adv_g + 10 * loss_cls + 10 * loss_img
        if train:
            trainable_variables = self.encoder.trainable_variables + \
                                  self.img_decoder.trainable_variables + \
                                  self.att_decoder.trainable_variables
            grads_enc = tape.gradient(loss_g, trainable_variables)
            self.g_opt.apply_gradients(zip(grads_enc, trainable_variables))
            return loss_g, loss_adv_g, 0.005 * loss_att1, 0.005 * loss_reg1, 10 * loss_cls, 10 * loss_img
            #return loss_g, loss_adv_g, 10 * loss_cls, 10 * loss_img
        else:
            return loss_g

    def dis_pretrain_step(self, source, label, train=True):
        label = tf.one_hot(label, depth=2)
        source = tf.cast(source, dtype='float32')
        label = tf.cast(label, dtype='float32')
        with tf.GradientTape() as tape:
            feat = self.encoder.call([source, label])
            A = self.att_decoder.call(feat)
            C = self.img_decoder.call(feat)
            gen_imgs = (1 - A) * C + A * source
            v_gen, c_gen = self.discriminator.call(gen_imgs)
            v_real, c_real = self.discriminator.call(tf.cast(source, dtype='float32'))
            loss_adv_d = 0.5 * (adversarial_loss(v_gen, False) + adversarial_loss(v_real, True))
            loss_cls = classify_loss(label, c_real)
            loss_d = loss_adv_d + 10 * loss_cls
        if train == True:
            grads = tape.gradient(loss_d, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
            return loss_d, loss_adv_d, 10 * loss_cls
        else:
            return loss_d

    def pretrain(self, epochs=30, interval=1, batch_size=8, batch_num=104):
        tr_L_G_avg = []
        tr_L_G_adv_avg = []
        tr_L_G_att1_avg = []
        tr_L_G_reg_avg = []
        tr_L_G_cls_avg = []
        tr_L_G_img_avg = []
        tr_L_D_avg = []
        tr_L_D_adv_avg = []
        tr_L_D_cls_avg = []
        te_L_G_avg = []
        te_L_D_avg = []
        start = time.time()
        for epoch in range(epochs):
            ep_start = time.time()
            tr_L_G = []
            tr_L_G_adv = []
            tr_L_G_att1 = []
            tr_L_G_reg = []
            tr_L_G_cls = []
            tr_L_G_img = []
            tr_L_D = []
            tr_L_D_adv = []
            tr_L_D_cls = []
            te_L_G = []
            te_L_D = []

            for b in range(batch_num):
                source = load_image(get_batch_data(self.total_roots_train, b, batch_size))
                label = get_batch_data(self.label_train, b, batch_size)
                b_test = random.randint(0, 14)
                source_test = load_image(get_batch_data(self.total_roots_test, b_test, batch_size))
                label_test = get_batch_data(self.label_test, b_test, batch_size)

                loss_g, loss_adv_g, loss_att1, loss_reg, loss_cls, loss_img = self.gen_pretrain_step(source, label)
                #loss_g, loss_adv_g, loss_cls, loss_img = self.gen_pretrain_step(source, label)
                loss_g_test = self.gen_pretrain_step(source_test, label_test, train=False)
                tr_L_G.append(loss_g)
                tr_L_G_adv.append(loss_adv_g)
                tr_L_G_att1.append(loss_att1)
                tr_L_G_reg.append(loss_reg)
                tr_L_G_cls.append(loss_cls)
                tr_L_G_img.append(loss_img)
                te_L_G.append(loss_g_test)
                loss_d, loss_adv_d, loss_cls_d = self.dis_pretrain_step(source, label)
                loss_d_test = self.dis_pretrain_step(source_test, label_test, train=False)
                tr_L_D.append(loss_d)
                tr_L_D_adv.append(loss_adv_d)
                tr_L_D_cls.append(loss_cls_d)
                te_L_D.append(loss_d_test)
            tr_L_G_avg.append(np.mean(tr_L_G))
            tr_L_G_adv_avg.append(np.mean(tr_L_G_adv))
            tr_L_G_att1_avg.append(np.mean(tr_L_G_att1))
            tr_L_G_reg_avg.append(np.mean(tr_L_G_reg))
            tr_L_G_cls_avg.append(np.mean(tr_L_G_cls))
            tr_L_G_img_avg.append(np.mean(tr_L_G_img))
            tr_L_D_avg.append(np.mean(tr_L_D))
            tr_L_D_adv_avg.append(np.mean(tr_L_D_adv))
            tr_L_D_cls_avg.append(np.mean(tr_L_D_cls))
            te_L_G_avg.append(np.mean(te_L_G))
            te_L_D_avg.append(np.mean(te_L_D))

            t_pass = time.time() - start
            m_pass, s_pass = divmod(t_pass, 60)
            h_pass, m_pass = divmod(m_pass, 60)
            print('\nTime for pass  {:<4d}: {:<2d} hour {:<3d} min {:<4.3f} sec'.format(epoch + 1, int(h_pass),
                                                                                        int(m_pass), s_pass))
            print('Time for epoch {:<4d}: {:6.3f} sec'.format(epoch + 1, time.time() - ep_start))
            print('Train Loss Generator     :  {:8.5f}'.format(tr_L_G_avg[-1]))
            print('Train Loss Gen_adv       :  {:8.5f}'.format(tr_L_G_adv_avg[-1]))
            print('Train Loss Gen_att1      :  {:8.5f}'.format(tr_L_G_att1_avg[-1]))
            print('Train Loss Gen_reg       :  {:8.5f}'.format(tr_L_G_reg_avg[-1]))
            print('Train Loss Gen_cls       :  {:8.5f}'.format(tr_L_G_cls_avg[-1]))
            print('Train Loss Gen_img       :  {:8.5f}'.format(tr_L_G_img_avg[-1]))
            print('Train Loss Discriminator :  {:8.5f}'.format(tr_L_D_avg[-1]))
            print('Train Loss Dis_adv       :  {:8.5f}'.format(tr_L_D_adv_avg[-1]))
            print('Train Loss Dis class     :  {:8.5f}'.format(tr_L_D_cls_avg[-1]))
            print('Test Loss Generator      :  {:8.5f}'.format(te_L_G_avg[-1]))
            print('Test Loss Discriminator  :  {:8.5f}'.format(te_L_D_avg[-1]))
            self.sample_images_pretrain(epoch)
            if (epoch % interval == 0 or epoch + 1 == epochs) and (te_L_G_avg[-1] <= np.min(te_L_G_avg)):
                self.encoder.save_weights('pretrain_weight/2_encoder_pretrained_weights_{}'.format(epoch + 1))
                self.img_decoder.save_weights('pretrain_weight/2_img_decoder_pretrained_weights_{}'.format(epoch + 1))
                self.att_decoder.save_weights('pretrain_weight/2_att_decoder_pretrained_weights_{}'.format(epoch + 1))
                self.discriminator.save_weights('pretrain_weight/2_discriminator_pretrained_weights_{}'.format(epoch + 1))
        return [tr_L_G_avg, tr_L_G_adv_avg, tr_L_G_att1_avg, tr_L_G_reg_avg, tr_L_G_cls_avg, tr_L_G_img_avg], \
               [tr_L_D_avg, tr_L_D_adv_avg, tr_L_D_cls_avg], [te_L_G_avg, te_L_D_avg]
        #return [tr_L_G_avg, tr_L_G_adv_avg, tr_L_G_cls_avg, tr_L_G_img_avg], \
        #       [tr_L_D_avg, tr_L_D_adv_avg, tr_L_D_cls_avg], [te_L_G_avg, te_L_D_avg]

    def sample_images_pretrain(self, epoch, path='/home/pomelo96/Desktop/Python/GANimation/pretrain_picture/2_pretrain'):
        source_train = load_image(get_batch_data(self.total_roots_train, 0, 5))
        source_test = load_image(get_batch_data(self.total_roots_test, 0, 5))
        source_sampling = tf.concat([source_train, source_test], axis=0)

        label_train = get_batch_data(self.label_train, 0, 5)
        label_test = get_batch_data(self.label_test, 0, 5)
        label_sampling = []
        [[label_sampling.append(i) for i in j] for j in [label_train, label_test]]
        label_sampling = tf.one_hot(label_sampling, depth=2)
        feat = self.encoder.call([source_sampling, label_sampling])
        A = self.att_decoder.call(feat)
        C = self.img_decoder.call(feat)
        gen_img = (1 - A) * C + A * source_sampling
        #gen_img = C
        source_sampling = 0.5 * (source_sampling + 1)
        C = 0.5 * (C + 1)
        gen_img = 0.5 * (gen_img + 1)

        r ,c = 4, 10
        fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
        plt.subplots_adjust(hspace=0.2)
        cnt = 0
        for j in range(c):
            axs[0, j].imshow(source_sampling[cnt], cmap='gray')
            axs[0, j].axis('off')
            axs[1, j].imshow(A[cnt], cmap='gray')
            axs[1, j].axis('off')
            axs[2, j].imshow(C[cnt], cmap='gray')
            axs[2, j].axis('off')
            axs[3, j].imshow(gen_img[cnt], cmap='gray')
            axs[3, j].axis('off')
            cnt += 1
        fig.savefig(path + '_{}.png'.format(epoch + 1))
        plt.close()


if __name__ == '__main__':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    import os
    print(tf.__version__)
    print(tf.test.is_gpu_available())
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    config = ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    ganimation = GANimation_pretrain()
    ganimation.encoder.load_weights('pretrain_weight/encoder_pretrained_weights_28')
    ganimation.img_decoder.load_weights('pretrain_weight/img_decoder_pretrained_weights_28')
    ganimation.discriminator.load_weights('pretrain_weight/discriminator_pretrained_weights_28')
    [tr_L_G_avg, tr_L_G_adv_avg, tr_L_G_att1_avg, tr_L_G_reg_avg, tr_L_G_cls_avg, tr_L_G_img_avg], \
    [tr_L_D_avg, tr_L_D_adv_avg, tr_L_D_cls_avg], [te_L_G_avg, te_L_D_avg] \
        = ganimation.pretrain(epochs=30, interval=1)

    plt.plot(tr_L_G_avg)
    plt.title('pretrain_Generator total loss')
    plt.savefig('pretrain_picture/2_Generator loss.jpg')
    plt.close()

    plt.plot(tr_L_G_adv_avg)
    plt.plot(tr_L_D_adv_avg)
    plt.title('pretrain_Adversarial total loss')
    plt.legend(['Generator', 'Discriminator'])
    plt.savefig('pretrain_picture/2_Adversarial loss.jpg')
    plt.close()

    plt.plot(tr_L_G_att1_avg)
    plt.title('pretrain_Generator attention loss')
    plt.savefig('pretrain_picture/2_Generator attention loss.jpg')
    plt.close()

    plt.plot(tr_L_G_reg_avg)
    plt.title('pretrain_Generator attention regression loss')
    plt.savefig('pretrain_picture/2_Generator attention regression loss.jpg')
    plt.close()

    plt.plot(tr_L_G_cls_avg)
    plt.title('pretrain_Generator classify loss')
    plt.savefig('pretrain_picture/2_Generator classify loss.jpg')
    plt.close()

    plt.plot(tr_L_G_img_avg)
    plt.title('pretrain_Generator image loss')
    plt.savefig('pretrain_picture/2_Generator image loss.jpg')
    plt.close()

    plt.plot(tr_L_D_avg)
    plt.title('pretrain_Discriminator total loss')
    plt.savefig('pretrain_picture/2_Discriminator loss.jpg')
    plt.close()

    plt.plot(tr_L_D_cls_avg)
    plt.title('pretrain_Discriminator classify total loss')
    plt.savefig('pretrain_picture/2_Discriminator classify loss.jpg')
    plt.close()

    plt.plot(te_L_G_avg)
    plt.title('pretrain_Generator Test loss')
    plt.savefig('pretrain_picture/2_Generator test loss.jpg')
    plt.close()

    plt.plot(te_L_D_avg)
    plt.title('pretrain_Discriminator Test loss')
    plt.savefig('pretrain_picture/2_Discriminator test loss.jpg')
    plt.close()
