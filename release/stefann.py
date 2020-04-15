# -*- coding: utf-8 -*-
"""
STEFANN | Scene Text Editor using Font Adaptive Neural Network.
Created on Mon Apr  1 11:00:00 2019
Author: Prasun Roy | https://prasunroy.github.io
GitHub: https://github.com/prasunroy/stefann

Copyright (c) 2019 Prasun Roy, Saumik Bhattacharya and Subhankar Ghosh.
Copyright (c) 2019 Indian Statistical Institute.
All Rights Reserved.

"""
# -----------------------------------------------------------------------------

# imports
import base64
import datetime
import io
import json
import os
import time
import webbrowser

import colorama
import cv2
import numpy

from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import Qt, QByteArray
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QDesktopWidget
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtWidgets import QFileDialog

# ensure keras using tensorflow backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# import modules from keras
from keras.models import model_from_json

# -----------------------------------------------------------------------------

APP_INFO = \
"""
   _____ _______ ______ ______      _   _ _   _ 
  / ____|__   __|  ____|  ____/\   | \ | | \ | |
 | (___    | |  | |__  | |__ /  \  |  \| |  \| |
  \___ \   | |  |  __| |  __/ /\ \ | . ` | . ` |
  ____) |  | |  | |____| | / ____ \| |\  | |\  |
 |_____/   |_|  |______|_|/_/    \_\_| \_|_| \_| v.0.1.0


 Scene Text Editor using Font Adaptive Neural Network.
 Copyright (c) 2019 Prasun Roy, Saumik Bhattacharya and Subhankar Ghosh.
 Copyright (c) 2019 Indian Statistical Institute.
 All Rights Reserved.
...............................................................................

"""
APP_NAME = 'STEFANN'

with open('appdata.json', 'r') as fp:
    APP_DATA = json.load(fp)

APP_ICON = APP_DATA['APP_ICON'].encode(encoding='utf-8')
APP_FONT = APP_DATA['APP_FONT'].encode(encoding='utf-8')
APP_DATA = None

RUN_FLAG = True

try:
    with open('models/fannet.json', 'r') as fp:
        NET_F = model_from_json(fp.read())
    with open('models/colornet.json', 'r') as fp:
        NET_C = model_from_json(fp.read())
    
    NET_F.load_weights('models/fannet_weights.h5')
    NET_C.load_weights('models/colornet_weights.h5')
except:
    RUN_FLAG = False

DELTA_FSCALE = 0.2
DELTA_THRESH = 5
DELTA_CNTMIN = 5

# -----------------------------------------------------------------------------

# GUI class
class GUI(QMainWindow):
    
    def __init__(self, root=None):
        super(GUI, self).__init__()
        self.init_ui()
        self.imdir = root
        self.image = None
    
    def init_ui(self):
        # set properties
        icon = QPixmap()
        icon.loadFromData(QByteArray.fromBase64(APP_ICON))
        self.setGeometry(0, 0, 500, 250)
        self.setWindowIcon(QIcon(icon))
        self.setWindowTitle(APP_NAME)
        
        # create widgets
        header = QLabel('STEFANN')
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet('QLabel {font-size: 80px; font-weight: bold;}')
        button_image = QPushButton('Open Image')
        button_paper = QPushButton('About Project')
        button_about = QPushButton('README')
        for button in [button_image, button_paper, button_about]:
            button.setMinimumSize(150, 30)
            button.setStyleSheet('QPushButton {font-size: 12px;}')
        
        # create layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        layout_1 = QHBoxLayout()
        layout_2 = QHBoxLayout()
        layout_1.addWidget(header)
        layout_2.addWidget(button_image)
        layout_2.addWidget(button_paper)
        layout_2.addWidget(button_about)
        main_layout.addLayout(layout_1)
        main_layout.addLayout(layout_2)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.setFixedSize(self.size())
        self.move_window_to_center()
        
        # set actions
        button_image.clicked.connect(self.open_image)
        button_paper.clicked.connect(lambda: webbrowser.open('https://prasunroy.github.io/stefann'))
        button_about.clicked.connect(lambda: webbrowser.open('readme.html'))
        return
    
    def move_window_to_center(self):
        window_rect = self.frameGeometry()
        screen_cent = QDesktopWidget().availableGeometry().center()
        window_rect.moveCenter(screen_cent)
        self.move(window_rect.topLeft())
        return
    
    def open_image(self):
        if self.imdir is None or not os.path.isdir(str(self.imdir)):
            self.imdir = os.path.expanduser('~')
        self.image, _ = QFileDialog.getOpenFileName(self, 'Open Image', self.imdir, 'Image Files (*.bmp *.dib *.jpeg *.jpg *.jpe *.jp2 *.png *.webp *.pbm *.pgm *.ppm *.pxm *.pnm *.sr *.ras *.tiff *.tif *.exr *.hdr *.pic)')
        if os.path.isfile(self.image):
            self.close()
        else:
            self.image = None
        return

# -----------------------------------------------------------------------------

# launcher
def launcher(root=None):
    app = QApplication([])
    gui = GUI(root)
    gui.show()
    app.exec()
    return gui.image

# -----------------------------------------------------------------------------

# get opencv version number
def opencv_version():
    return int(cv2.__version__.split('.')[0])

# -----------------------------------------------------------------------------

# draw grid on image
def draw_grid(image, line_space=10, line_color=(0, 255, 0), line_thickness=1):
    # get image size
    output = image.copy()
    nr, nc = output.shape[:2]
    
    # draw horizontal lines
    for r in range(0, nr, line_space):
        cv2.line(output, (0, r), (nc-1, r), line_color, line_thickness)
    
    # draw vertical lines
    for c in range(0, nc, line_space):
        cv2.line(output, (c, 0), (c, nr-1), line_color, line_thickness)
    
    return output

# -----------------------------------------------------------------------------

# select region of interest
def select_region(event, x, y, flags, points):
    # no reference to points list is passed
    if points is None or not type(points) is list:
        return
    
    # handle events
    # - insert : left click
    # - remove : right click
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) > 4:
            points.clear()
        elif len(points) == 4:
            points.pop(0)
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            points.pop()
    return

# -----------------------------------------------------------------------------

# sort points in top-left, top-right, bottom-right, bottom-left order
def sort_points(points):
    points = sorted(points, key=lambda x: x[1])
    points = sorted(points[:2], key=lambda x: x[0]) + \
             sorted(points[2:], key=lambda x: x[0], reverse=True)
    return points

# -----------------------------------------------------------------------------

# scale points
def scale_points(points, fscale=1.0):
    scaled_points = points.copy()
    for point in scaled_points:
        point[0] = round(point[0] * fscale)
        point[1] = round(point[1] * fscale)
    return scaled_points

# -----------------------------------------------------------------------------

# draw region
def draw_region(image, points):
    output = image.copy()
    points = sort_points(points)
    npoint = len(points)
    for i in range(npoint):
        cv2.line(output, points[i], points[(i + 1) % npoint], (0, 0, 255), 1, cv2.LINE_AA)
    for i in range(npoint):
        cv2.circle(output, points[i], 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, points[i], 4, (0, 255, 0), -1, cv2.LINE_AA)
    return output

# -----------------------------------------------------------------------------

# binarize image
def binarize(image, points=None, thresh=128, maxval=255, thresh_type=0):
    # convert image to grayscale
    image = image.copy()
    
    # remove everything except the region bounded by given points
    if not points is None and type(points) is list and len(points) > 2:
        points = sort_points(points)
        points = numpy.array(points, numpy.int64)
        mask = numpy.zeros_like(image, numpy.uint8)
        cv2.fillConvexPoly(mask, points, (255, 255, 255), cv2.LINE_AA)
        image = cv2.bitwise_and(image, mask)
    
    # estimate mask 1 from MSER
    msers = cv2.MSER_create().detectRegions(image)[0]
    setyx = set()
    for region in msers:
        for point in region:
            setyx.add((point[1], point[0]))
    setyx = tuple(numpy.transpose(list(setyx)))
    mask1 = numpy.zeros(image.shape, numpy.uint8)
    mask1[setyx] = maxval
    
    # estimate mask 2 from thresholding
    mask2 = cv2.threshold(image, thresh, maxval, thresh_type)[1]
    
    # get binary image from estimated masks
    image = cv2.bitwise_and(mask1, mask2)
    
    return image

# -----------------------------------------------------------------------------

# find contours
def find_contours(image, min_area=0, sort=True):
    # convert image to grayscale
    image = image.copy()
    
    # find contours
    if opencv_version() == 3:
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    else:
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    
    # filter contours by area
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    
    if len(contours) < 1:
        return ([], [])
    
    # sort contours from left to right using respective bounding boxes
    if sort:
        bndboxes = [cv2.boundingRect(contour) for contour in contours]
        contours, bndboxes = zip(*sorted(zip(contours, bndboxes), key=lambda x: x[1][0]))
    
    return contours, bndboxes

# -----------------------------------------------------------------------------

# draw contours
def draw_contours(image, contours, index, color=(0, 255, 0), color_mode=None):
    image = cv2.cvtColor(image, color_mode) if color_mode else image.copy()
    drawn = numpy.zeros_like(image, numpy.uint8)
    for i in range(len(contours)):
        drawn = cv2.drawContours(drawn, contours, i, (255, 255, 255), -1, cv2.LINE_AA)
    if len(contours) > 0 and index >= 0:
        drawn = cv2.drawContours(drawn, contours, index, color, -1, cv2.LINE_AA)
    image = cv2.bitwise_and(drawn, image)
    return image

# -----------------------------------------------------------------------------

# grab region of interest
def grab_region(image, bwmask, contours, bndboxes, index):
    region = numpy.zeros_like(bwmask, numpy.uint8)
    if len(contours) > 0 and len(bndboxes) > 0 and index >= 0:
        x, y, w, h = bndboxes[index]
        region = cv2.drawContours(region, contours, index, (255, 255, 255), -1, cv2.LINE_AA)
        region = region[y:y+h, x:x+w]
        bwmask = bwmask[y:y+h, x:x+w]
        bwmask = cv2.bitwise_and(region, region, mask=bwmask)
        region = image[y:y+h, x:x+w]
        region = cv2.bitwise_and(region, region, mask=bwmask)
    return region

# -----------------------------------------------------------------------------

# grab all regions of interest
def grab_regions(image, image_mask, contours, bndboxes):
    regions = []
    for index in range(len(bndboxes)):
        regions.append(grab_region(image, image_mask, contours, bndboxes, index))
    return regions

# -----------------------------------------------------------------------------

# convert image to tensor
def image2tensor(image, shape, padding=0.0, rescale=1.0, color_mode=None):
    output = cv2.cvtColor(image, color_mode) if color_mode else image.copy()
    output = numpy.atleast_3d(output)
    rect_w = output.shape[1]
    rect_h = output.shape[0]
    sqrlen = int(numpy.ceil((1.0 + padding) * max(rect_w, rect_h)))
    sqrbox = numpy.zeros((sqrlen, sqrlen, output.shape[2]), numpy.uint8)
    rect_x = (sqrlen - rect_w) // 2
    rect_y = (sqrlen - rect_h) // 2
    sqrbox[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = output
    output = cv2.resize(sqrbox, shape[:2])
    output = numpy.atleast_3d(output)
    output = numpy.asarray(output, numpy.float32) * rescale
    output = output.reshape((1,) + output.shape)
    return output

# -----------------------------------------------------------------------------

# convert character to one-hot encoding
def char2onehot(character, alphabet):
    onehot = [0.] * len(alphabet)
    onehot[alphabet.index(character)] = 1.
    onehot = numpy.asarray(onehot, numpy.float32).reshape(1, len(alphabet), 1)
    return onehot

# -----------------------------------------------------------------------------

# resize image
def resize(image, w=-1, h=-1, bbox=False):
    image = Image.fromarray(image)
    bnbox = image.getbbox() if bbox else None
    image = image.crop(bnbox) if bnbox else image
    if w <= 0 and h <= 0:
        w = image.width
        h = image.height
    elif w <= 0 and h > 0:
        w = int(image.width / image.height * h)
    elif w > 0 and h <= 0:
        h = int(image.height / image.width * w)
    else:
        pass
    image = image.resize((w, h))
    image = numpy.asarray(image, numpy.uint8)
    return image

# -----------------------------------------------------------------------------

# update bounding boxes
def update_bndboxes(bndboxes, index, image):
    change_x = (image.shape[1] - bndboxes[index][2]) // 2
    bndboxes = list(bndboxes)
    for i in range(0, index + 1):
        x, y, w, h = bndboxes[i]
        bndboxes[i] = (x - change_x, y, w, h)
    for i in range(index + 1, len(bndboxes)):
        x, y, w, h = bndboxes[i]
        bndboxes[i] = (x + change_x, y, w, h)
    bndboxes = tuple(bndboxes)
    return bndboxes

# -----------------------------------------------------------------------------

# paste images
def paste_images(image, patches, bndboxes):
    image = Image.fromarray(image)
    for patch, bndbox in zip(patches, bndboxes):
        patch = Image.fromarray(patch)
        image.paste(patch, bndbox[:2])
    image = numpy.asarray(image, numpy.uint8)
    return image

# -----------------------------------------------------------------------------

# inpaint image
def inpaint(image, mask):
    k = numpy.ones((5, 5), numpy.uint8)
    m = cv2.dilate(mask, k, iterations=1)
    i = cv2.inpaint(image, m, 10, cv2.INPAINT_TELEA)
    return i

# -----------------------------------------------------------------------------

# transfer color having maximum occurence
def transfer_color_max(source, target):
    colors = source.convert('RGB').getcolors(256*256*256)
    colors = sorted(colors, key=lambda x: x[0], reverse=True)
    maxcol = colors[0][1] if len(colors) == 1 else \
             colors[0][1] if colors[0][1] != (0, 0, 0) else \
             colors[1][1]
    output = Image.new('RGB', target.size)
    colors = Image.new('RGB', target.size, maxcol)
    output.paste(colors, (0, 0), target.convert('L'))
    return output

# -----------------------------------------------------------------------------

# transfer color using approximate pallet
def transfer_color_pal(source, target):
    source = source.convert('RGB')
    src_bb = source.getbbox()
    src_bb = source.crop(src_bb) if src_bb else source.copy()
    colors = Image.new('RGB', src_bb.size)
    src_np = numpy.asarray(src_bb, numpy.uint8)
    for i in range(src_np.shape[0]):
        row_np = src_np[i].reshape(1, -1, 3)
        col_id = numpy.where(row_np == 0)[1]
        row_np = numpy.delete(row_np, col_id, axis=1)
        row_im = Image.fromarray(row_np).resize((colors.width, 1))
        colors.paste(row_im, (0, i))
    target = target.convert('L')
    colors = colors.resize(target.size)
    output = Image.new('RGB', target.size)
    output.paste(colors, (0, 0), target)
    return output

# -----------------------------------------------------------------------------

# edit character
def edit_char(image, image_mask, contours, bndboxes, index, char, alphabet, fannet, colornet):
    # validate parameters
    if len(contours) <= 0 or len(bndboxes) <= 0 or len(contours) != len(bndboxes) or index < 0:
        return
    
    # generate character
    region_f = grab_region(image_mask, image_mask, contours, bndboxes, index)
    tensor_f = image2tensor(region_f, fannet.input_shape[0][1:3], .1, 1.)
    onehot_f = char2onehot(char, alphabet)
    output_f = fannet.predict([tensor_f, onehot_f])
    output_f = numpy.squeeze(output_f)
    output_f = numpy.asarray(output_f, numpy.uint8)
    
    # transfer color
    region_c = grab_region(image, image_mask, contours, bndboxes, index)
    source_c = Image.fromarray(region_c)
    target_f = Image.fromarray(output_f)
    output_c = transfer_color_max(source_c, target_f)
    output_c = numpy.asarray(output_c, numpy.uint8)
    # output_c = transfer_color_pal(source_c, target_f)
    # output_c = numpy.asarray(output_c, numpy.uint8)
    # input1_c = image2tensor(region_c, colornet.input_shape[0][1:3], .1, 1.)
    # input2_f = image2tensor(output_f, colornet.input_shape[0][1:3], .1, 1.)
    # output_c = colornet.predict([input1_c, input2_f])
    # output_c = numpy.squeeze(output_c)
    # output_c = numpy.asarray(output_c, numpy.uint8)
    # output_c = cv2.bitwise_and(output_c, output_c, mask=output_f)
    
    output_f = resize(output_f, -1, region_f.shape[0], True)
    output_c = resize(output_c, -1, region_c.shape[0], True)
    
    # inpaint old layout
    mpatches = grab_regions(image_mask, image_mask, contours, bndboxes)
    o_layout = numpy.zeros_like(image_mask, numpy.uint8)
    o_layout = paste_images(o_layout, mpatches, bndboxes)
    inpainted_image = inpaint(image, o_layout)
    
    # create new layout
    bpatches = grab_regions(image, image_mask, contours, bndboxes)
    bndboxes = update_bndboxes(bndboxes, index, output_f)
    bpatches[index] = output_c
    n_layout = numpy.zeros_like(image, numpy.uint8)
    n_layout = paste_images(n_layout, bpatches, bndboxes)
    mpatches[index] = output_f
    m_layout = numpy.zeros_like(image_mask, numpy.uint8)
    m_layout = paste_images(m_layout, mpatches, bndboxes)
    
    # generate final result
    n_layout = Image.fromarray(n_layout)
    m_layout = Image.fromarray(m_layout)
    inpainted_image = Image.fromarray(inpainted_image)
    inpainted_image.paste(n_layout, (0, 0), m_layout)
    
    layout = numpy.asarray(m_layout, numpy.uint8)
    edited = numpy.asarray(inpainted_image, numpy.uint8)
    
    return (layout, edited)

# -----------------------------------------------------------------------------

# apply watermark
def watermark(image, text, size=None, color=None, alpha=1.0, position=0):
    back = Image.fromarray(image).convert('RGBA') if type(image) is numpy.ndarray else image.convert('RGBA')
    fore = Image.new(back.mode, back.size, (0, 0, 0, 0))
    size = min(back.width, back.height) // 20 if size is None else max(20, size)
    font = ImageFont.truetype(io.BytesIO(base64.b64decode(APP_FONT)), size)
    w, h = font.getsize(text)
    rgba = (255, 255, 255) if color is None else color
    rgba = rgba + (int(255 * alpha),)
    if position == 0:
        x, y = (back.width - w) // 2, (back.height - h) // 2
    elif position == 1:
        x, y = 8, 4
    elif position == 2:
        x, y = back.width - w - 8, 4
    elif position == 3:
        x, y = back.width - w - 8, back.height - h - 8
    elif position == 4:
        x, y = 8, back.height - h - 8
    draw = ImageDraw.Draw(fore)
    draw.text((x, y), text, rgba, font)
    output = Image.alpha_composite(back, fore).convert('RGB')
    return numpy.uint8(output) if type(image) is numpy.ndarray else output

# -----------------------------------------------------------------------------

# get timestamp
def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

# -----------------------------------------------------------------------------

# main
if __name__ == '__main__':
    
    # clear screen
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
    
    # initialize colorama
    colorama.init(autoreset=True)
    
    # initialize variables
    imdir = None
    
    # application info
    print(APP_INFO)
    print(colorama.Fore.LIGHTBLACK_EX + '[DEBUG] Loading application... ', end='')
    
    # integrity check
    if RUN_FLAG:
        print(colorama.Fore.LIGHTGREEN_EX + 'done')
    else:
        print(colorama.Fore.LIGHTRED_EX + 'integrity failure')
    
    
    # ---------- Main Loop ----------
    while RUN_FLAG:
        # open launcher
        image = launcher(imdir)
        
        # check image
        if image is None:
            print(colorama.Fore.LIGHTBLACK_EX + '[DEBUG] Exiting application...')
            break
        elif not os.path.isfile(image):
            print(colorama.Fore.LIGHTRED_EX + '[ERROR] File does not exist')
            continue
        
        # read image
        try:
            image_orig = cv2.imread(image, cv2.IMREAD_COLOR)
        except:
            image_orig = None
        if image_orig is None:
            print(colorama.Fore.LIGHTRED_EX + '[ERROR] Failed to read image')
            continue
        else:
            print(colorama.Fore.LIGHTGREEN_EX + f'[DEBUG] Loaded new image from: {image}')
            imdir = os.path.dirname(image)
        
        # create a named window
        cv2.namedWindow('STEFANN')
        
        # initialize variables
        grid = False
        fscale = 1.0
        points = []
        thresh = 150
        invert = 0
        cntmin = 0
        cntidx = 0
        edited = False
        
        step = 1
        
        
        # ---------- Image Scaling and Region Selection ----------
        # setup mouse callback
        cv2.setMouseCallback('STEFANN', select_region, points)
        
        # create a scaled copy of the original image
        image_scaled = image_orig.copy()
        
        # begin loop
        while step == 1:
            # handle keyboard events
            # - exit loop   : `ESC`       >> ASCII: 27
            # - toggle grid : `G` or `g`  >> ASCII: 71, 103
            # - reset scale : `R` or `r`  >> ASCII: 82, 114
            # - scale up    : `+`         >> ASCII: 43
            # - scale down  : `-`         >> ASCII: 45
            # - next step   : `ENTER`     >> ASCII: 13
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                print(colorama.Fore.LIGHTRED_EX + '[DEBUG] Operation canceled')
                break
            elif key == 71 or key == 103:
                grid = not grid
                print(colorama.Fore.LIGHTCYAN_EX + f'[DEBUG] Toggled grids -> {grid}')
            elif key == 82 or key == 114:
                fscale = 1.0
                print(colorama.Fore.LIGHTBLACK_EX + f'[DEBUG] Updated scale -> {fscale}x')
                image_scaled = image_orig.copy()
                points.clear()
            elif key == 43:
                fscale = round(min(fscale + DELTA_FSCALE, 5.0), 1)
                print(colorama.Fore.LIGHTBLACK_EX + f'[DEBUG] Updated scale -> {fscale}x')
                image_scaled = cv2.resize(image_orig, None, fx=fscale, fy=fscale)
                points.clear()
            elif key == 45:
                fscale = round(max(fscale - DELTA_FSCALE, 0.2), 1)
                print(colorama.Fore.LIGHTBLACK_EX + f'[DEBUG] Updated scale -> {fscale}x')
                image_scaled = cv2.resize(image_orig, None, fx=fscale, fy=fscale)
                points.clear()
            elif key == 13:
                step += 1
            
            # perform operations on a working copy of scaled image
            image_work = draw_grid(image_scaled) if grid else image_scaled.copy()
            image_work = draw_region(image_work, points)
            
            # display the working copy of image
            cv2.imshow('STEFANN', image_work)
        
        
        # ---------- Image Binarization and Editing ----------
        # setup mouse callback
        cv2.setMouseCallback('STEFANN', select_region, None)
        
        # perform binarization and contour detection
        image_edit = image_scaled.copy()
        image_gray = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2GRAY)
        image_mask = binarize(image_gray, points, thresh, 255, invert)
        contours, bndboxes = find_contours(image_mask, cntmin)
        
        # begin loop
        while step == 2:
            # handle keyboard events
            # - exit loop                     : `ESC`        >> ASCII: 27
            # - increase threshold            : `+`          >> ASCII: 43
            # - decrease threshold            : `-`          >> ASCII: 45
            # - inverse thresholding          : `TAB`        >> ASCII: 9
            # - increase allowed contour area : `*`          >> ASCII: 42
            # - decrease allowed contour area : `/`          >> ASCII: 47
            # - select contour                : `SPACE`      >> ASCII: 32
            # - insert character              : `A-Z`        >> ASCII: 65-90
            # - undo all insertions           : `BACKSPACE`  >> ASCII: 8
            # - next step                     : `ENTER`      >> ASCII: 13
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                print(colorama.Fore.LIGHTRED_EX + '[DEBUG] Operation canceled')
                break
            elif key == 43:
                thresh = min(thresh + DELTA_THRESH, 255)
                print(colorama.Fore.LIGHTBLACK_EX + f'[DEBUG] Increased threshold -> {thresh}')
                image_mask = binarize(image_gray, points, thresh, 255, invert)
                contours, bndboxes = find_contours(image_mask, cntmin)
            elif key == 45:
                thresh = max(thresh - DELTA_THRESH, 0)
                print(colorama.Fore.LIGHTBLACK_EX + f'[DEBUG] Decreased threshold -> {thresh}')
                image_mask = binarize(image_gray, points, thresh, 255, invert)
                contours, bndboxes = find_contours(image_mask, cntmin)
            elif key == 9:
                invert = int(not invert)
                print(colorama.Fore.LIGHTCYAN_EX + f'[DEBUG] Invert thresholding -> {invert}')
                image_mask = binarize(image_gray, points, thresh, 255, invert)
                contours, bndboxes = find_contours(image_mask, cntmin)
            elif key == 42:
                cntmin = min(cntmin + DELTA_CNTMIN, image_mask.shape[0] * image_mask.shape[1])
                print(colorama.Fore.LIGHTBLACK_EX + f'[DEBUG] Increased allowed contour area -> {cntmin}')
                contours, bndboxes = find_contours(image_mask, cntmin)
            elif key == 47:
                cntmin = max(cntmin - DELTA_CNTMIN, 0)
                print(colorama.Fore.LIGHTBLACK_EX + f'[DEBUG] Decreased allowed contour area -> {cntmin}')
                contours, bndboxes = find_contours(image_mask, cntmin)
            elif key == 32:
                cntidx = (cntidx + 1) % len(contours) if len(contours) > 0 else -1
            elif (key >= 65 and key <= 90) or (key >= 97 and key <= 122):
                if key >= 97 and key <= 122:
                    key -= 32
                print(colorama.Fore.CYAN + f'[DEBUG] Inserting character -> {chr(key)}')
                try:
                    image_mask, image_edit = edit_char(image_edit, image_mask, contours, bndboxes, cntidx, chr(key), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', NET_F, NET_C)
                except:
                    print(colorama.Fore.LIGHTRED_EX + '[ERROR] Operation failed')
                    continue
                image_gray = cv2.cvtColor(image_edit, cv2.COLOR_BGR2GRAY)
                contours, bndboxes = find_contours(image_mask, cntmin)
                edited = True
            elif key == 8:
                print(colorama.Fore.LIGHTRED_EX + '[DEBUG] Reset modifications')
                image_edit = image_scaled.copy()
                image_gray = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2GRAY)
                image_mask = binarize(image_gray, points, thresh, 255, invert)
                contours, bndboxes = find_contours(image_mask, cntmin)
                edited = False
            elif key == 13:
                step += 1
            
            # perform operations on a working copy of binarized image
            image_work = draw_contours(image_mask, contours, cntidx, (0, 255, 0), cv2.COLOR_GRAY2BGR)
            
            # display the working copy of image
            cv2.imshow('STEFANN', image_work)
        
        # save edited image
        if edited:
            # apply watermark before saving
            image_edit = watermark(image_edit, 'Edited with STEFANN', alpha=0.3, position=3)
            
            root, ext = os.path.splitext(image)
            file_path = root + '_' + timestamp() + ext
            try:
                cv2.imwrite(file_path, image_edit)
                print(colorama.Fore.LIGHTGREEN_EX + f'[DEBUG] Edited image saved as: {file_path}')
            except:
                print(colorama.Fore.LIGHTRED_EX + '[ERROR] Failed to write image')
        
        
        # ---------- Display Results ----------
        # initialize variables
        change = False
        layout = 0
        labels = False
        fscale = 1.0
        
        # create a working image
        image_work = image_edit.copy()
        
        # begin loop
        while step == 3:
            # handle keyboard events
            # - exit loop          : `ENTER` or `ESC`  >> ASCII: 13, 27
            # - select layout      : `SPACE`           >> ASCII: 32
            # - toggle annotations : `TAB`             >> ASCII: 9
            # - scale up           : `+`               >> ASCII: 43
            # - scale down         : `-`               >> ASCII: 45
            # - reset scale        : `R` or `r`        >> ASCII: 82, 114
            # - save layout        : `S` or `s`        >> ASCII: 83, 115
            key = cv2.waitKey(1) & 0xff
            if key ==13 or key == 27:
                break
            elif key == 32:
                layout = (layout + 1) % 6
                print(colorama.Fore.LIGHTBLACK_EX + f'[DEBUG] Select layout -> {layout}')
                change = True
            elif key == 9:
                labels = not labels
                print(colorama.Fore.LIGHTCYAN_EX + f'[DEBUG] Toggled label -> {labels}')
                change = True
            elif key == 43:
                fscale = round(min(fscale + DELTA_FSCALE, 2.5), 1)
                print(colorama.Fore.LIGHTBLACK_EX + f'[DEBUG] Updated scale -> {fscale}x')
                change = True
            elif key == 45:
                fscale = round(max(fscale - DELTA_FSCALE, 0.2), 1)
                print(colorama.Fore.LIGHTBLACK_EX + f'[DEBUG] Updated scale -> {fscale}x')
                change = True
            elif key == 82 or key == 114:
                fscale = 1.0
                print(colorama.Fore.LIGHTBLACK_EX + f'[DEBUG] Updated scale -> {fscale}x')
                change = True
            elif key == 83 or key == 115:
                root, ext = os.path.splitext(image)
                file_path = root + '_' + timestamp() + ext
                try:
                    cv2.imwrite(file_path, image_work)
                    print(colorama.Fore.LIGHTGREEN_EX + f'[DEBUG] Image layout saved as: {file_path}')
                except:
                    print(colorama.Fore.LIGHTRED_EX + '[ERROR] Failed to write image')
            
            # update the working image
            if change:
                image_work_0 = cv2.resize(image_scaled, None, fx=fscale, fy=fscale)
                image_work_1 = cv2.resize(image_edit, None, fx=fscale, fy=fscale)
                rows, cols = image_work_0.shape[:2]
                h_bar_1 = numpy.zeros((10, cols, 3), numpy.uint8)
                v_bar_1 = numpy.zeros((rows, 10, 3), numpy.uint8)
                if labels:
                    image_work_0 = watermark(image_work_0, 'ORIGINAL', 20, color=(255, 255, 0), alpha=0.7, position=4)
                    image_work_1 = watermark(image_work_1, 'EDITED', 20, color=(0, 255, 0), alpha=0.7, position=4)
                if layout == 0:
                    image_work = image_work_1.copy()
                elif layout == 1:
                    image_work = image_work_0.copy()
                elif layout == 2:
                    image_work = numpy.hstack((image_work_0, v_bar_1, image_work_1))
                elif layout == 3:
                    image_work = numpy.hstack((image_work_1, v_bar_1, image_work_0))
                elif layout == 4:
                    image_work = numpy.vstack((image_work_0, h_bar_1, image_work_1))
                elif layout == 5:
                    image_work = numpy.vstack((image_work_1, h_bar_1, image_work_0))
                change = False
            
            # display the working image
            cv2.imshow('STEFANN', image_work)
        
        # cleanup
        cv2.destroyAllWindows()
    
    # deinitialize colorama
    colorama.deinit()
    
    # delay
    time.sleep(2)
