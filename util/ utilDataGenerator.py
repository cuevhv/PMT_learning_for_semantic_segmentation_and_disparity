#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import random
import math
import cv2
import numpy as np
from sklearn.utils import shuffle

class LazyFileLoader:
    def __init__(self, array_x_files, array_y_files, page_size, ptr_load_images_func, load_images_config, activation_output, crop, centered):
        assert len(array_x_files) > 0
        assert len(array_x_files)== len(array_y_files)
        self.array_x_files = array_x_files
        self.array_y_files = array_y_files
        self.ptr_load_images_func = ptr_load_images_func
        self.load_images_config = load_images_config
        self.pos = 0
        self.activation_output = activation_output
        self.crop = crop
        self.centered = centered
        if page_size <= 0:
            self.page_size = len(array_x_files)
        else:
            self.page_size = page_size

    def __len__(self):
        return len(self.array_x_files)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def truncate_to_size(self, truncate_to):
        self.array_x_files = self.array_x_files[0:truncate_to]
        self.array_y_files = self.array_y_files[0:truncate_to]

    def set_x_files(self, array_x_files, array_y_files):
        self.array_x_files = array_x_files
        self.array_y_files = array_y_files

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def set_pos(self, pos):
        self.pos = pos

    def shuffle(self):
        self.array_x_files, self.array_y_files = \
                shuffle(self.array_x_files, self.array_y_files, random_state=0)

    def next(self):
        psize = self.page_size
        if self.pos + psize >= len(self.array_x_files):  # last page?
            if self.pos >= len(self.array_x_files):
                raise StopIteration
            else:
                psize = len(self.array_x_files) - self.pos

        print('Loading page from {} to {}...'.format(self.pos, self.pos + psize))
        page_data = self.ptr_load_images_func(
                                                    self.array_x_files[self.pos:self.pos + psize],
                                                    self.array_y_files[self.pos:self.pos + psize],
                                                    self.load_images_config,
                                                    self.activation_output,
                                                    self.crop,
                                                    self.centered)
        self.pos += self.page_size

        return page_data

