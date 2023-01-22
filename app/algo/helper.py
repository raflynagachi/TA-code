import struct
from sys import byteorder


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


def bytes_to_binary(B):
    return bin(int.from_bytes(B, byteorder))[2:]
