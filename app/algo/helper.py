import struct
import random
import numpy as np
import math
from sys import byteorder
from math import log2, ceil
from typing import List
from random import randrange
from difflib import SequenceMatcher


def float_to_binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


def binary_to_float(binary):
    binary = int(binary, 2)
    return struct.unpack('f', struct.pack('I',  binary))[0]


def int_to_binary(num, is_32Bit=False):
    if is_32Bit:
        return '{:032b}'.format(num)
    return bin(num)


def binary_to_int(binary):
    return int(binary, 2)


def bytes_to_binary(byte_arr):
    return ''.join([format(b, '08b') for b in byte_arr])


def binary_to_bytes(bin_str):
    return bytes([int(bin_str[i:i+8], 2) for i in range(0, len(bin_str), 8)])


def split_image_to_block(image, block_size):
    blocks = []
    for vert_slice in np.vsplit(image, int(image.shape[0] / block_size)):
        for horz_slice in np.hsplit(vert_slice, int(image.shape[1] / block_size)):
            blocks.append(horz_slice)
    return blocks


def stitch_8x8_blocks_back_together(size, block_segments):
    image_rows = []
    temp = []
    for i in range(len(block_segments)):
        if i > 0 and not (i % int(size / 8)):
            image_rows.append(temp)
            temp = [block_segments[i]]
        else:
            temp.append(block_segments[i])
    image_rows.append(temp)

    return np.block(image_rows)


def array_to_image(image, channel):
    stego_image = np.empty_like(image)
    width = image.shape[1]
    stego_image[:, :, 0] = np.asarray(
        stitch_8x8_blocks_back_together(width, channel[0]))
    stego_image[:, :, 1] = np.asarray(
        stitch_8x8_blocks_back_together(width, channel[1]))
    stego_image[:, :, 2] = np.asarray(
        stitch_8x8_blocks_back_together(width, channel[2]))
    return stego_image


def MSE(original, stego):
    mse = np.mean((original - stego) ** 2)
    return mse


def PSNR(original, stego):
    mse = MSE(original, stego)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def similarity_string(x, y):
    n = min(len(x), len(y))
    common = 0
    for i in range(n):
        if (x[i] == y[i]):
            common += 1
    return common/math.sqrt(len(x)*len(y))


def max_bit_cap(m, n, bit_embedded):
    # bit_embedded is embedded bit in block 8x8
    return (m*n/64) * bit_embedded


def __hamming_common(src: List[List[int]], s_num: int, encode=True) -> None:
    """
    Here's the real magic =)
    """
    s_range = range(s_num)

    for i in src:
        sindrome = 0
        for s in s_range:
            sind = 0
            for p in range(2 ** s, len(i) + 1, 2 ** (s + 1)):
                for j in range(2 ** s):
                    if (p + j) > len(i):
                        break
                    sind ^= i[p + j - 1]

            if encode:
                i[2 ** s - 1] = sind
            else:
                sindrome += (2 ** s * sind)

        if (not encode) and sindrome:
            i[sindrome - 1] = int(not i[sindrome - 1])


def hamming_encode(msg: str, mode: int = 8) -> str:
    """
    Encoding the message with Hamming code.
    :param msg: Message string to encode
    :param mode: number of significant bits
    :return: 
    """

    result = ""

    msg_b = binary_to_bytes(msg)
    s_num = ceil(log2(log2(mode + 1) + mode + 1))   # number of control bits
    bit_seq = []
    for byte in msg_b:  # get bytes to binary values; every bits store to sublist
        bit_seq += list(map(int, f"{byte:08b}"))

    res_len = ceil((len(msg_b) * 8) / mode)     # length of result (bytes)
    bit_seq += [0] * (res_len * mode - len(bit_seq))    # filling zeros

    to_hamming = []

    for i in range(res_len):    # insert control bits into specified positions
        code = bit_seq[i * mode:i * mode + mode]
        for j in range(s_num):
            code.insert(2 ** j - 1, 0)
        to_hamming.append(code)

    __hamming_common(to_hamming, s_num, True)   # process

    for i in to_hamming:
        result += "".join(map(str, i))

    return result


def hamming_decode(msg: str, mode: int = 8) -> str:
    """
    Decoding the message with Hamming code.
    :param msg: Message string to decode
    :param mode: number of significant bits
    :return: 
    """

    result = ""

    s_num = ceil(log2(log2(mode + 1) + mode + 1))   # number of control bits
    res_len = len(msg) // (mode + s_num)    # length of result (bytes)
    code_len = mode + s_num     # length of one code sequence

    to_hamming = []

    for i in range(res_len):    # convert binary-like string to int-list
        code = list(map(int, msg[i * code_len:i * code_len + code_len]))
        to_hamming.append(code)

    __hamming_common(to_hamming, s_num, False)  # process

    for i in to_hamming:    # delete control bits
        for j in range(s_num):
            i.pop(2 ** j - 1 - j)
        result += "".join(map(str, i))

    msg_l = []

    for i in range(len(result) // 8):   # convert from binary-sring value to integer
        val = "".join(result[i * 8:i * 8 + 8])
        msg_l.append(int(val, 2))

    # finally decode to a regular string
    # result = bytes(msg_l).decode("utf-8")

    return result
