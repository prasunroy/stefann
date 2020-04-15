# -*- coding: utf-8 -*-
"""
Utility functions.
Created on Wed Oct 10 17:00:00 2018
Author: Prasun Roy | https://prasunroy.github.io
GitHub: https://github.com/prasunroy/stefann

"""


# imports
import glob
import numpy
import requests

from io import BytesIO
from PIL import Image


# TelegramIM class
class TelegramIM(object):
    
    def __init__(self, auth_token, chat_id):
        self._api_url = f'https://api.telegram.org/bot{auth_token}'
        self._chat_id = chat_id
    
    def send_message(self, text, parse_mode='Markdown'):
        try:
            requests.post(self._api_url + '/sendMessage', data={
                'chat_id': self._chat_id,
                'text': text,
                'parse_mode': parse_mode
            })
        except:
            pass
    
    def send_photo(self, photo, caption='', parse_mode='Markdown'):
        try:
            requests.post(self._api_url + '/sendPhoto', data={
                'chat_id': self._chat_id,
                'caption': caption,
                'parse_mode': parse_mode
            }, files={
                'photo': self._read_photo(photo)
            })
        except:
            pass
    
    def send_photos(self, img_dir, caption='', parse_mode='Markdown', limit=10):
        inputs = sorted(glob.glob(img_dir + '/**/*.*', recursive=True))
        images = []
        for file in inputs:
            try:
                images.append(Image.open(file).convert('RGB'))
            except:
                pass
            if len(images) >= limit:
                break
        imgs_w = max([image.width for image in images])
        imgs_h = sum([image.height for image in images])
        merged = Image.new('RGB', (imgs_w, imgs_h))
        offset = 0
        for image in images:
            merged.paste(image, (0, offset))
            offset += image.height
        self.send_photo(merged, caption, parse_mode)
    
    def _read_photo(self, photo):
        if isinstance(photo, str):
            photo = Image.open(photo)
        elif isinstance(photo, numpy.ndarray):
            photo = Image.fromarray(photo)
        buffer = BytesIO()
        photo.save(buffer, 'JPEG')
        return buffer.getbuffer()
