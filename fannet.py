# -*- coding: utf-8 -*-
"""
Font Adaptive Neural Network (FANnet).
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
import itertools
import numpy
import os
import tensorflow

from keras.callbacks import Callback
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
from PIL import Image

from utils import TelegramIM


# configurations
# -----------------------------------------------------------------------------
TIMESTAMP        = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

RANDOM_SEED      = None

ARCHITECTURE     = 'fannet'

SOURCE_CHARS     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
TARGET_CHARS     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

TRAIN_IMAGES_DIR = 'datasets/fannet/train/'
VALID_IMAGES_DIR = 'datasets/fannet/valid/'
PAIRS_IMAGES_DIR = 'datasets/fannet/pairs/'
OUTPUT_DIR       = 'output/{}/{}/'.format(ARCHITECTURE, TIMESTAMP)

IMAGE_FILE_EXT   = '.jpg'
IMAGE_READ_MODE  = 'L'

FUNCTION_OPTIM   = 'adam'
FUNCTION_LOSS    = 'mae'

INPUT_SHAPE_IMG  = (64, 64, 1)
INPUT_SHAPE_HOT  = (len(SOURCE_CHARS), 1)

SCALE_COEFF_IMG  = 1.
BATCH_SIZE       = 1
NUM_EPOCHS       = 1
NOTIFY_EVERY     = 1

AUTH_TOKEN       = None
CHAT_ID          = None
# -----------------------------------------------------------------------------


# DataGenerator class
class DataGenerator(object):
    def __init__(self, source_chars, target_chars, image_dir, image_ext='.jpg',
                 mode='RGB', target_shape=(64, 64), rescale=1.0, batch_size=1,
                 seed=None):
        self._chars = source_chars
        self._perms = list(itertools.product(list(source_chars),
                                             list(target_chars),
                                             os.listdir(image_dir)))
        self._imdir = image_dir
        self._imext = image_ext
        self._imtyp = mode
        self._shape = target_shape
        self._scale = rescale
        self._batch = batch_size
        self._steps = int(len(self._perms) / self._batch + 0.5)
        self._index = 0
        numpy.random.seed(seed)
        numpy.random.shuffle(self._perms)
        return
    
    def flow(self):
        while True:
            x = []
            y = []
            onehot = []
            endidx = self._index + self._batch
            subset = self._perms[self._index:endidx]
            numpy.random.shuffle(subset)
            self._index = endidx if endidx < len(self._perms) else 0
            for perm in subset:
                ch_src = str(ord(perm[0]))
                ch_dst = str(ord(perm[1]))
                ch_fnt = perm[2]
                im_src = os.path.join(self._imdir, ch_fnt, ch_src + self._imext)
                im_dst = os.path.join(self._imdir, ch_fnt, ch_dst + self._imext)
                try:
                    img_x0 = Image.open(im_src).convert(self._imtyp).resize(self._shape)
                    img_x0 = numpy.asarray(img_x0, dtype=numpy.uint8)
                    img_x0 = numpy.atleast_3d(img_x0)
                    img_y0 = Image.open(im_dst).convert(self._imtyp).resize(self._shape)
                    img_y0 = numpy.asarray(img_y0, dtype=numpy.uint8)
                    img_y0 = numpy.atleast_3d(img_y0)
                except:
                    continue
                x.append(img_x0)
                y.append(img_y0)
                idx = self._chars.find(perm[1])
                hot = [0] * len(self._chars)
                hot[idx] = 1
                onehot.append(numpy.asarray(hot, numpy.uint8).reshape(-1, 1))
            x = numpy.asarray(x, numpy.float32) * self._scale
            y = numpy.asarray(y, numpy.float32) * self._scale
            onehot = numpy.asarray(onehot, numpy.float32)
            yield [[x, onehot], y]


# FANnet class
class FANnet(object):
    def __new__(self, input_shapes, optimizer, loss, weights=None):
        # build network
        x1 = Input(input_shapes[0])
        x2 = Input(input_shapes[1])
        
        y1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x1)
        y1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(y1)
        y1 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(y1)
        y1 = Flatten()(y1)
        y1 = Dense(units=512, activation='relu')(y1)
        
        y2 = Flatten()(x2)
        y2 = Dense(units=512, activation='relu')(y2)
        
        y = Concatenate()([y1, y2])
        y = Dense(units=1024, activation='relu')(y)
        y = Dropout(0.5)(y)
        y = Dense(units=1024, activation='relu')(y)
        y = Reshape(target_shape=(8, 8, 16))(y)
        y = UpSampling2D(size=(2, 2))(y)
        y = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(y)
        y = UpSampling2D(size=(2, 2))(y)
        y = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(y)
        y = UpSampling2D(size=(2, 2))(y)
        y = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(y)
        
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
    def __init__(self, out_dir, charset, img_dir, img_ext='.jpg', mode='RGB',
                 rescale=1.0, thumbnail_size=(64, 64), messenger=None,
                 notify_every=1, network_id='network'):
        self._out_dir = out_dir
        self._charset = charset
        self._img_dir = img_dir
        self._img_ext = img_ext
        self._img_typ = mode
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
        images = glob.glob(self._img_dir + '/**/*' + self._img_ext, recursive=True)
        for image in images:
            try:
                im_org = Image.open(image).convert(self._img_typ)
                im_src = im_org.crop((0, 0, im_org.width // 2, im_org.height))
                im_dst = im_org.crop((im_org.width // 2, 0, im_org.width, im_org.height))
                img_x0 = im_src.resize(self.model.input_shape[0][1:3])
                img_x0 = numpy.asarray(img_x0, numpy.float32) * self._rescale
                img_x0 = numpy.atleast_3d(img_x0)
                img_x0 = numpy.expand_dims(img_x0, 0)
                dst_ch = os.path.splitext(os.path.basename(image))[0].split('_')[-1]
                idx_ch = self._charset.find(chr(int(dst_ch)))
                onehot = [0] * len(self._charset)
                onehot[idx_ch] = 1
                onehot = numpy.asarray(onehot, numpy.uint8).reshape(1, -1, 1)
                img_y0 = self.model.predict([img_x0, onehot])
                img_y0 = numpy.squeeze(img_y0)
                img_y0 = numpy.asarray(img_y0 / self._rescale, numpy.uint8)
                im_prd = Image.fromarray(img_y0)
                result = self._combine_images([im_src, im_dst, im_prd], self._tn_size)
                impath = self._out_dir + '/epoch_{}/{}.jpg'.format(epoch + 1, os.path.splitext(os.path.basename(image))[0])
                self._save_image(impath, result)
            except:
                continue
        if (epoch + 1) % self._msg_frq == 0:
            try:
                self._msg_app.send_message('`Epoch {}\nloss: {:.4f} - val_loss: {:.4f}`'\
                                           .format(epoch + 1, logs.get('loss'), logs.get('val_loss')))
                img_dir = self._out_dir + '/epoch_{}/'.format(epoch + 1)
                caption = '`Epoch {}`'.format(epoch + 1)
                self._msg_app.send_photos(img_dir, caption)
            except:
                pass
        return
    
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
    if not os.path.isdir(PAIRS_IMAGES_DIR):
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
    train_datagen = DataGenerator(source_chars=SOURCE_CHARS,
                                  target_chars=TARGET_CHARS,
                                  image_dir=TRAIN_IMAGES_DIR,
                                  image_ext=IMAGE_FILE_EXT,
                                  mode=IMAGE_READ_MODE,
                                  target_shape=INPUT_SHAPE_IMG[:2],
                                  rescale=SCALE_COEFF_IMG,
                                  batch_size=BATCH_SIZE,
                                  seed=RANDOM_SEED)
    
    # initialize valid data generator
    valid_datagen = DataGenerator(source_chars=SOURCE_CHARS,
                                  target_chars=TARGET_CHARS,
                                  image_dir=VALID_IMAGES_DIR,
                                  image_ext=IMAGE_FILE_EXT,
                                  mode=IMAGE_READ_MODE,
                                  target_shape=INPUT_SHAPE_IMG[:2],
                                  rescale=SCALE_COEFF_IMG,
                                  batch_size=BATCH_SIZE,
                                  seed=RANDOM_SEED)
    
    # build and serialize network
    print('[INFO] Building network... ', end='')
    fannet = FANnet(input_shapes=[INPUT_SHAPE_IMG, INPUT_SHAPE_HOT],
                    optimizer=FUNCTION_OPTIM, loss=FUNCTION_LOSS, weights=None)
    print('done')
    fannet.summary()
    
    with open(mdl_file, 'w') as file:
        file.write(fannet.to_json())
    
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
    progress = ProgressMonitor(out_dir=pro_dir,
                               charset=SOURCE_CHARS,
                               img_dir=PAIRS_IMAGES_DIR,
                               img_ext=IMAGE_FILE_EXT,
                               mode=IMAGE_READ_MODE,
                               rescale=SCALE_COEFF_IMG,
                               thumbnail_size=(64, 64),
                               messenger=messenger,
                               notify_every=NOTIFY_EVERY,
                               network_id=ARCHITECTURE.upper())
    
    # train network
    fannet.fit_generator(generator=train_datagen.flow(),
                         steps_per_epoch=train_datagen._steps,
                         epochs=NUM_EPOCHS,
                         callbacks=[csv_logs, cpt_best, cpt_last, progress],
                         validation_data=valid_datagen.flow(),
                         validation_steps=valid_datagen._steps)
    
    return


# main
if __name__ == '__main__':
    train()
