import os
import argparse
from PIL import Image, ImageOps
import numpy as np
import math


def get_pixel_by_base(base):
    if base == 'A':
        return [0, 0, 255, 255]
    if base == 'C':
        return [255, 0, 0, 255]
    if base == 'G':
        return [0, 255, 0, 255]
    if base == 'T':
        return [255, 255, 0, 255]
    if base == '*':
        return [255, 192, 203, 255]
    return [0, 0, 0, 255]


def get_base_by_color(base):
    """
    Get color based on a base.
    - Uses different band of the same channel.
    :param base:
    :return:
    """
    if base >= 250:
        return 'A'
    if base >= 180:
        return 'G'
    if base >= 100:
        return 'T'
    if base >= 30:
        return 'C'
    if base >= 10:
        return '*'
    if base >= 5:
        return '.'



def get_alt_support_by_color(is_in_support):
    """
    ***NOT USED YET***
    :param is_in_support:
    :return:
    """
    if 248 <= is_in_support <= 256:
        return 1
    elif 148 <= is_in_support <= 153:
        return 0


def get_quality_by_color(map_quality):
    """
    Get a color spectrum given mapping quality
    :param map_quality: value of mapping quality
    :return:
    """
    color = math.floor(((map_quality / 254) * 9))
    return color

def get_match_ref_color(is_match):
    """
    Get color for base matching to reference
    :param is_match: If true, base matches to reference
    :return:
    """
    if 48 <= is_match <= 52:
        return 0
    elif is_match == 254:
        return 1


def get_strand_color(is_rev):
    """
    Get color for forward and reverse reads
    :param is_rev: True if read is reversed
    :return:
    """
    if 238 <= is_rev <= 244:
        return 1
    else:
        return 0


def get_cigar_by_color(cigar_code):
    """
    ***NOT USED YET***
    :param is_in_support:
    :return:
    """
    if 250 <= cigar_code <= 256:
        return 0
    if 148 <= cigar_code <= 158:
        return 1
    if 70 <= cigar_code <= 78:
        return 2


def np_array_to_img(img, img_width, img_height):
    if isinstance(img, np.ndarray) is False:
        img = img.numpy() * 255
    else:
        img = np.transpose(img, (2, 0, 1))
        img = np.array(img).astype(np.uint8)

    whole_img = []
    for i in range(img_height):
        img_row = []
        for j in range(img_width):
            if i > 5 and get_match_ref_color(img[4][i][j]) == 0:
                img_row.append([255, 255, 255, 255])
            elif img[0][i][j] != 0:
                pixel = get_pixel_by_base(get_base_by_color(img[0][i][j]))
                img_row.append(pixel)
            else:
                img_row.append([0, 0, 0, 255])
        whole_img.append(img_row)

    return whole_img


def analyze_np_array(img, img_height, img_width):
    if isinstance(img, np.ndarray) is False:
        img = img.numpy() * 255
    else:
        img = np.transpose(img, (2, 0, 1))
        img = np.array(img).astype(np.uint8)

    print("BASE CHANNEL")
    for i in range(img_height):
        for j in range(img_width):
            # print(img[0][i][j], get_base_by_color(img[0][i][j]))
            if img[0][i][j] != 0:
                print(get_base_by_color(img[0][i][j]), end='')
            else:
                print(' ', end='')
        print()

    print("SUPPORT CHANNEL")
    for i in range(img_height):
        for j in range(img_width):
            if img[5][i][j] != 0:
                print(get_alt_support_by_color(img[5][i][j]), end='')
            else:
                print(' ', end='')
        print()

    print("BASE QULAITY CHANNEL")
    for i in range(img_height):
        for j in range(img_width):
            if img[1][i][j] != 0:
                print(get_quality_by_color(img[1][i][j]), end='')
            else:
                print(' ', end='')
        print()

    print("MAP QUALITY CHANNEL")
    for i in range(img_height):
        for j in range(img_width):
            if img[2][i][j] != 0:
                print(get_quality_by_color(img[2][i][j]), end='')
            else:
                print(' ', end='')
        print()

    print("MISMATCH CHANNEL")
    for i in range(img_height):
        for j in range(img_width):
            if img[4][i][j] != 0:
                print(get_match_ref_color(img[4][i][j]), end='')
            else:
                print(' ', end='')
        print()

    print("STRAND CHANNEL")
    for i in range(img_height):
        for j in range(img_width):
            if img[3][i][j] != 0:
                print(get_strand_color(img[3][i][j]), end='')
            else:
                print(' ', end='')
        print()
