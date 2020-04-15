# -*- coding: utf-8 -*-
"""
Color Transfer Neural Network (Colornet).
Created on Wed Oct 10 17:00:00 2018
Author: Prasun Roy | https://prasunroy.github.io
GitHub: https://github.com/prasunroy/stefann

"""


# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import glob
import numpy
import os
import tensorflow

from keras.callbacks import Callback
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
from PIL import Image

from utils import TelegramIM


# configurations
# -----------------------------------------------------------------------------
TIMESTAMP        = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

RANDOM_SEED      = None

ARCHITECTURE     = 'colornet'

TRAIN_IMAGES_DIR = 'datasets/colornet/train/'
VALID_IMAGES_DIR = 'datasets/colornet/valid/'
TEST_IMAGES_DIR  = 'datasets/colornet/test/'
OUTPUT_DIR       = 'output/{}/{}/'.format(ARCHITECTURE, TIMESTAMP)

IMAGE_FILE_EXT   = '.jpg'

FUNCTION_OPTIM   = 'adam'
FUNCTION_LOSS    = 'mae'

INPUT_IMAGE_SIZE = (64, 64)

SCALE_COEFF_IMG  = 1.
BATCH_SIZE       = 1
NUM_EPOCHS       = 1
NOTIFY_EVERY     = 1

AUTH_TOKEN       = None
CHAT_ID          = None
# -----------------------------------------------------------------------------


# DataGenerator class
class DataGenerator(object):
    def __init__(self, image_dir_input1, image_dir_input2, image_dir_output,
                 image_ext='.jpg', target_shape=(64, 64), rescale=1.,
                 batch_size=1, shuffle=True, seed=None):
        numpy.random.seed(seed)
        self._imdir_in1 = image_dir_input1
        self._imdir_in2 = image_dir_input2
        self._imdir_out = image_dir_output
        self._imext = image_ext
        self._shape = target_shape
        self._scale = rescale
        self._batch = batch_size
        self._shake = shuffle
        self._files = sorted([os.path.split(path)[-1] for path in \
                              glob.glob('{}/*{}'.format(image_dir_input1, image_ext))])
        self._steps = int(len(self._files) / self._batch + 0.5)
        self._index = 0
        if shuffle:
            numpy.random.shuffle(self._files)
        return
    
    def flow(self):
        while True:
            x1 = []
            x2 = []
            y1 = []
            endidx = self._index + self._batch
            subset = self._files[self._index:endidx]
            self._index = endidx if endidx < len(self._files) else 0
            if self._shake:
                numpy.random.shuffle(subset)
            for file in subset:
                file_input1 = os.path.join(self._imdir_in1, file)
                file_input2 = os.path.join(self._imdir_in2, file)
                file_output = os.path.join(self._imdir_out, file)
                try:
                    input1 = Image.open(file_input1).convert('RGB').resize(self._shape)
                    input1 = numpy.asarray(input1, dtype=numpy.uint8)
                    input1 = numpy.atleast_3d(input1)
                    input2 = Image.open(file_input2).convert('L').resize(self._shape)
                    input2 = numpy.asarray(input2, dtype=numpy.uint8)
                    input2 = numpy.atleast_3d(input2)
                    output = Image.open(file_output).convert('RGB').resize(self._shape)
                    output = numpy.asarray(output, dtype=numpy.uint8)
                    output = numpy.atleast_3d(output)
                except:
                    continue
                x1.append(input1)
                x2.append(input2)
                y1.append(output)
            x1 = numpy.asarray(x1, dtype=numpy.float32) * self._scale
            x2 = numpy.asarray(x2, dtype=numpy.float32) * self._scale
            y1 = numpy.asarray(y1, dtype=numpy.float32) * self._scale
            yield [[x1, x2], y1]


# Colornet class
class Colornet(object):
    def __new__(self, input_shapes, optimizer, loss, weights=None):
        # build network
        x1 = Input(input_shapes[0])
        x2 = Input(input_shapes[1])
        
        y1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x1)
        y1 = LeakyReLU(alpha=0.2)(y1)
        y1 = BatchNormalization()(y1)
        
        y2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x2)
        y2 = LeakyReLU(alpha=0.2)(y2)
        y2 = BatchNormalization()(y2)
        
        y = Concatenate()([y1, y2])
        
        y = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        y = MaxPooling2D(pool_size=(2, 2))(y)
        
        y = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        y = MaxPooling2D(pool_size=(2, 2))(y)
        
        y = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        y = UpSampling2D(size=(2, 2))(y)
        y = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        y = UpSampling2D(size=(2, 2))(y)
        y = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        y = Conv2D(filters=3, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        # compile network
        model = Model(inputs=[x1, x2], outputs=y)
        model.compile(optimizer=optimizer, loss=loss)
        
        # optionally load existing weights into network
        try:
            if not weights is None:
                model.load_weights(weights)
        except:
            pass
        
        return model


# ProgressMonitor class
class ProgressMonitor(Callback):
    def __init__(self, image_dir_input1, image_dir_input2, image_dir_output,
                 save_to_dir, image_ext='.jpg', rescale=1.,
                 thumbnail_size=(64, 64), messenger=None, notify_every=1,
                 network_id='network'):
        self._imdir_in1 = image_dir_input1
        self._imdir_in2 = image_dir_input2
        self._imdir_out = image_dir_output
        self._imdir_dmp = save_to_dir
        self._img_ext = image_ext
        self._rescale = rescale
        self._tn_size = thumbnail_size
        self._msg_app = messenger
        self._msg_frq = max(1, notify_every)
        self._network = network_id
        return
    
    def on_train_begin(self, logs):
        try:
            if not self._msg_app is None:
                self._msg_app.send_message('`A new {} training has been started.\nI will post an update every {} epoch{}.`'\
                                           .format(self._network, self._msg_frq, 's' if self._msg_frq > 1 else ''))
        except:
            pass
        return
    
    def on_train_end(self, logs):
        try:
            if not self._msg_app is None:
                self._msg_app.send_message('`An ongoing {} training is finished.`'\
                                           .format(self._network))
        except:
            pass
        return
    
    def on_epoch_end(self, epoch, logs):
        images_in1 = sorted(glob.glob(self._imdir_in1 + '/*' + self._img_ext))
        images_in2 = sorted(glob.glob(self._imdir_in2 + '/*' + self._img_ext))
        images_out = sorted(glob.glob(self._imdir_out + '/*' + self._img_ext))
        for image_in1, image_in2, image_out in zip(images_in1, images_in2, images_out):
            try:
                img_in1 = Image.open(image_in1).convert('RGB')
                img_in2 = Image.open(image_in2).convert('L')
                img_out = Image.open(image_out).convert('RGB')
                x1 = img_in1.resize(self.model.input_shape[0][1:3])
                x1 = numpy.asarray(x1, dtype=numpy.float32) * self._rescale
                x1 = numpy.atleast_3d(x1)
                x1 = numpy.expand_dims(x1, axis=0)
                x2 = img_in2.resize(self.model.input_shape[1][1:3])
                x2 = numpy.asarray(x2, dtype=numpy.float32) * self._rescale
                x2 = numpy.atleast_3d(x2)
                x2 = numpy.expand_dims(x2, axis=0)
                y1 = self.model.predict([x1, x2])
                y1 = numpy.squeeze(y1)
                y1 = numpy.asarray(y1 / self._rescale, dtype=numpy.uint8)
                img_gen = Image.fromarray(y1)
                img_gen = self._postprocess_image(img_gen, img_in2)
                img_res = self._combine_images([img_in1, img_in2, img_out, img_gen], self._tn_size)
                dmp_res = self._imdir_dmp + '/epoch_{}/{}.jpg'.format(epoch + 1, os.path.splitext(os.path.basename(image_out))[0])
                self._save_image(dmp_res, img_res)
            except:
                continue
        if (epoch + 1) % self._msg_frq == 0:
            try:
                self._msg_app.send_message('`Epoch {}\nloss: {:.4f} - val_loss: {:.4f}`'\
                                           .format(epoch + 1, logs.get('loss'), logs.get('val_loss')))
                img_dir = self._imdir_dmp + '/epoch_{}/'.format(epoch + 1)
                caption = '`Epoch {}`'.format(epoch + 1)
                self._msg_app.send_photos(img_dir, caption)
            except:
                pass
        return
    
    def _postprocess_image(self, image, mask):
        image_pped = Image.new(image.mode, image.size)
        image_mask = mask.convert('L').resize(image.size)
        image_pped.paste(image, (0, 0), image_mask)
        return image_pped
    
    def _combine_images(self, images=[], size=(64, 64),
                        border_colors=[], border_width=0):
        n = len(images)
        w = n * (size[0] + 2 * border_width)
        h = size[1] + 2 * border_width
        surf = Image.new('RGB', (w, h))
        flag = True if len(border_colors) == n and border_width > 0 else False
        for index, image in enumerate(images):
            x = index * w // n
            y = 0
            if flag:
                back = Image.new('RGB', (w // n, h), border_colors[index])
                surf.paste(back, (x, y))
            x += border_width
            y += border_width
            surf.paste(image.convert('RGB').resize(size), (x, y))
        return surf
    
    def _save_image(self, filepath, image):
        directory = os.path.dirname(filepath)
        if not os.path.isdir(directory) and directory != '':
            os.makedirs(directory)
        image.save(filepath)
        return


# get tensorflow version number
def tensorflow_version():
    return int(tensorflow.__version__.split('.')[0])


# train
def train():
    # setup seed for random number generators for reproducibility
    numpy.random.seed(RANDOM_SEED)
    
    if tensorflow_version() == 2:
        tensorflow.random.set_seed(RANDOM_SEED)
    else:
        tensorflow.set_random_seed(RANDOM_SEED)
    
    # setup paths
    mdl_dir = os.path.join(OUTPUT_DIR, 'models')
    log_dir = os.path.join(OUTPUT_DIR, 'logs')
    cpt_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    pro_dir = os.path.join(OUTPUT_DIR, 'progress')
    
    setup_flag = True
    for directory in [TRAIN_IMAGES_DIR, VALID_IMAGES_DIR]:
        if not os.path.isdir(directory):
            print('[INFO] Data directory not found at {}'.format(directory))
            setup_flag = False
    if not os.path.isdir(TEST_IMAGES_DIR):
        print('[INFO] Data directory not found at {}'.format(directory))
    for directory in [OUTPUT_DIR, mdl_dir, log_dir, cpt_dir, pro_dir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        elif len(glob.glob(os.path.join(directory, '*.*'))) > 0:
            print('[INFO] Output directory {} must be empty'.format(directory))
            setup_flag = False
    if not setup_flag:
        return
    
    mdl_file = os.path.join(mdl_dir, '{}.json'.format(ARCHITECTURE))
    log_file = os.path.join(log_dir, '{}_training.csv'.format(ARCHITECTURE))
    cpt_file_best = os.path.join(cpt_dir, '{}_weights.h5'.format(ARCHITECTURE))
    cpt_file_last = os.path.join(cpt_dir, '{}_weights_{{}}.h5'.format(ARCHITECTURE))
    
    # initialize messenger
    messenger = TelegramIM(auth_token=AUTH_TOKEN, chat_id=CHAT_ID)
    
    # initialize train data generator
    train_datagen = DataGenerator(image_dir_input1=TRAIN_IMAGES_DIR + '/input_color/',
                                  image_dir_input2=TRAIN_IMAGES_DIR + '/input_mask/',
                                  image_dir_output=TRAIN_IMAGES_DIR + '/output_color/',
                                  image_ext=IMAGE_FILE_EXT,
                                  target_shape=INPUT_IMAGE_SIZE,
                                  rescale=SCALE_COEFF_IMG,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  seed=RANDOM_SEED)
    
    # initialize valid data generator
    valid_datagen = DataGenerator(image_dir_input1=VALID_IMAGES_DIR + '/input_color/',
                                  image_dir_input2=VALID_IMAGES_DIR + '/input_mask/',
                                  image_dir_output=VALID_IMAGES_DIR + '/output_color/',
                                  image_ext=IMAGE_FILE_EXT,
                                  target_shape=INPUT_IMAGE_SIZE,
                                  rescale=SCALE_COEFF_IMG,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  seed=RANDOM_SEED)
    
    # build and serialize network
    print('[INFO] Building network... ', end='')
    colornet = Colornet(input_shapes=[INPUT_IMAGE_SIZE + (3,), INPUT_IMAGE_SIZE + (1,)],
                        optimizer=FUNCTION_OPTIM,
                        loss=FUNCTION_LOSS,
                        weights=None)
    print('done')
    colornet.summary()
    
    with open(mdl_file, 'w') as file:
        file.write(colornet.to_json())
    
    # create callbacks
    csv_logs = CSVLogger(filename=log_file, append=True)
    cpt_best = ModelCheckpoint(filepath=cpt_file_best,
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True)
    cpt_last = ModelCheckpoint(filepath=cpt_file_last.format('{epoch:d}'),
                               monitor='val_loss',
                               verbose=0,
                               save_best_only=False,
                               save_weights_only=True)
    progress = ProgressMonitor(image_dir_input1=TEST_IMAGES_DIR + '/input_color/',
                               image_dir_input2=TEST_IMAGES_DIR + '/input_mask/',
                               image_dir_output=TEST_IMAGES_DIR + '/output_color/',
                               save_to_dir=pro_dir,
                               image_ext=IMAGE_FILE_EXT,
                               rescale=SCALE_COEFF_IMG,
                               thumbnail_size=(64, 64),
                               messenger=messenger,
                               notify_every=NOTIFY_EVERY,
                               network_id=ARCHITECTURE.upper())
    
    # train network
    colornet.fit_generator(generator=train_datagen.flow(),
                           steps_per_epoch=train_datagen._steps,
                           epochs=NUM_EPOCHS,
                           callbacks=[csv_logs, cpt_best, cpt_last, progress],
                           validation_data=valid_datagen.flow(),
                           validation_steps=valid_datagen._steps)
    
    return


# main
if __name__ == '__main__':
    train()
