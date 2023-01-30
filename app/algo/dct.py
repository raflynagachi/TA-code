import numpy as np
import math
import zigzag as zg
import zlib
import struct
import cv2
from bitarray import bitarray
from sys import getsizeof, byteorder
from difflib import SequenceMatcher
import helper

"""
Pseudocode:
1. konversi nilai piksel BGR menjadi YCbCr
2. Citra digital dipecah menjadi blok 8x8
3. Matriks orisinil dikurangi 128
4. Perhitungan DCT
5. Kompresi blok dengan matriks kuantisasi JPEG
6. Bentuk matriks 1 dimensi dengan zig-zag scanning
7. Sisipkan pesan rahasia (biner) ke middle frequency LSB
8. Dekuantisasi blok 8x8 dan Invers DCT
9. Blok 8x8 ditambah 128
"""


class DCT:
    """
    A implementation of the DCT Algorithm
    """
    BLOCK_SIZE = 8
    QUANTIZATION_TABLE = [[16,  11,  10,  16,  24,  40,  51,  61],
                          [12,  12,  14,  19,  26,  58,  60,  55],
                          [14,  13,  16,  24,  40,  57,  69,  56],
                          [14,  17,  22,  29,  51,  87,  80,  62],
                          [18,  22,  37,  56,  68, 109, 103,  77],
                          [24,  35,  55,  64,  81, 104, 113,  92],
                          [49,  64,  78,  87, 103, 121, 120, 101],
                          [72,  92,  95,  98, 112, 100, 103,  99]]

    def __init__(self, decode=False, cover_image=None):
        self.quant_table = np.float64(self.QUANTIZATION_TABLE)
        if not decode:
            self.ori_img, self.image = prep_image(cover_image)
            self.height, self.width = self.image.shape[:2]
            self.ori_width, self.ori_height = cover_image.shape[:2]
            self.channel = [
                np.float64(self.split_image_to_block(
                    self.image[:, :, 0], self.BLOCK_SIZE)),
                np.float64(self.split_image_to_block(
                    self.image[:, :, 1], self.BLOCK_SIZE)),
                np.float64(self.split_image_to_block(
                    self.image[:, :, 2], self.BLOCK_SIZE))
            ]

    def split_image_to_block(self, image, block_size):
        blocks = []
        for vert_slice in np.vsplit(image, int(image.shape[0] / block_size)):
            for horz_slice in np.hsplit(vert_slice, int(image.shape[1] / block_size)):
                blocks.append(horz_slice)
        return blocks

    def stitch_8x8_blocks_back_together(self, block_segments):
        image_rows = []
        temp = []
        for i in range(len(block_segments)):
            if i > 0 and not(i % int(self.width / 8)):
                image_rows.append(temp)
                temp = [block_segments[i]]
            else:
                temp.append(block_segments[i])
        image_rows.append(temp)

        return np.block(image_rows)

    def array_to_image(self, channel):
        stego_image = np.empty_like(self.image)
        stego_image[:, :, 0] = np.asarray(
            self.stitch_8x8_blocks_back_together(channel[0]))
        stego_image[:, :, 1] = np.asarray(
            self.stitch_8x8_blocks_back_together(channel[1]))
        stego_image[:, :, 2] = np.asarray(
            self.stitch_8x8_blocks_back_together(channel[2]))
        return stego_image

    def dct(self, block, verbose=False):
        dctMat = []
        for u in range(8):
            cu = float(1 / math.sqrt(2)) if u == 0 else 1
            dctMat.append([])

            for v in range(8):
                cv = float(1 / math.sqrt(2)) if v == 0 else 1

                foo = 0.0
                for x in range(8):
                    for y in range(8):
                        val = math.cos((2*x+1)*(u*math.pi)/16) * \
                            math.cos((2*y+1)*(v*math.pi)/16)
                        foo += (block[x][y] * val)

                foo = 0.25 * cu * cv * foo
                dctMat[u].append(foo)

                if verbose:
                    print(block[u][v] * val, end=" + ")
        return dctMat

    def idct(self, block, verbose=False):
        dctMat = []
        for x in range(8):
            dctMat.append([])
            for y in range(8):
                foo = 0.0
                for u in range(8):
                    for v in range(8):
                        cu = float(1 / math.sqrt(2)) if u == 0 else 1
                        cv = float(1 / math.sqrt(2)) if v == 0 else 1
                        val = math.cos((2*x+1)*(u*math.pi)/16) * \
                            math.cos((2*y+1)*(v*math.pi)/16)
                        foo += (block[u][v] * cu * cv * val)

                foo = round(0.25 * foo)
                dctMat[x].append(foo)

                if verbose:
                    print(block[x][y] * val, end=" + ")
        return dctMat

    def embed_message(self, block, encoded_data):
        start_idx, end_idx = 12, 32
        i = 0
        # print("block: ", block[start_idx:end_idx])
        for item in block[start_idx: end_idx]:
            if len(encoded_data) == 0:
                break

            # TODO
            # avoid embedding in padded pixel
            # if ... : do something

            embed = helper.int_to_binary(item)
            embed = embed[:-1] + encoded_data[0]
            # print(encoded_data[0], "\t", embed, end="\n")
            block[start_idx + i] = helper.binary_to_int(embed)

            encoded_data = encoded_data[1:]
            i += 1
        return block, encoded_data

    def extract_message(self, block, message, max_char):
        start_idx, end_idx = 12, 32
        i = 0 if message == "" else len(message)
        # print("block: ", block[start_idx:end_idx])
        for item in block[start_idx: end_idx]:
            if max_char != 0 and max_char == len(message):
                break

            embed = helper.int_to_binary(item)
            message += embed[-1]

            if i == 31 and max_char == 0:
                print("message 31: ", message)
                max_char = helper.binary_to_int(message)
                message = ""

            i += 1
        return message, max_char

    def post_image(self, stego):
        # retrieve cropped pixel
        image = self.ori_img
        image[:stego.shape[0], :stego.shape[1]] = stego
        # image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
        img = np.uint8(np.clip(image, 0, 255))
        cv2.imwrite("stego_image.png", img)
        print("stego ori2 BGR: ", np.array(img)[0][0])

    def encode(self, message):
        if message == None:
            return

        # index for Y (luminance) layer
        idx_channel = 0
        message = helper.int_to_binary(len(message), True) + message
        # TODO
        # len message is 1493 but got 29846 in binary form
        # print("MAX char len msg: ", helper.binary_to_int(
        #     '0b'+helper.int_to_binary(len(message), True)))

        # modify only for specific layer
        dct_blocks = [np.subtract(block, 128)
                      for block in self.channel[idx_channel]]

        # forward dct stage
        dct_blocks = [cv2.dct(block.astype(float))
                      for block in dct_blocks]

        # quantize
        dct_blocks = [np.round(block/self.quant_table) for block in dct_blocks]

        for idx, block in enumerate(dct_blocks):
            if len(message) == 0:
                print("last index: ", idx)
                break

            # sort dct coefficient by frequency
            sorted_coef = np.rint(zg.zigzag(block)).astype(int)
            # print("before: ", sorted_coef)

            # embed message into DCT coefficient
            sorted_coef, message = self.embed_message(sorted_coef, message)
            # print("after: ", sorted_coef)

            # unpack zigzag
            dct_blocks[idx] = zg.inverse_zigzag(sorted_coef, 8, 8)

        embedded_block = [block * self.quant_table for block in dct_blocks]
        embedded_block = np.rint([cv2.idct(block.astype(float))
                                  for block in embedded_block]).astype(int)
        embedded_block = [np.add(block, 128) for block in embedded_block]

        stego_channel = []
        for i in range(3):
            if idx_channel == i:
                stego_channel.append(embedded_block)
                continue
            stego_channel.append(self.channel[i])
        stego_image = self.array_to_image(stego_channel)
        print("stego ori: ", stego_image[0][0])
        # idx0, idx1, idx2 = 1, 4, 1
        # print("FLOAT: ", stego_image[idx0][idx1][idx2],
        #       helper.float_to_binary(stego_image[idx0][idx1][idx2]))
        # print("INT: ", np.uint8(stego_image)[
        #       idx0][idx1][idx2], helper.int_to_binary(np.uint8(stego_image)[idx0][idx1][idx2]))
        # print("INT: ", np.float32(np.uint8(stego_image))[
        #       idx0][idx1][idx2], helper.int_to_binary(np.uint8(stego_image)[idx0][idx1][idx2]))

        # saving image
        print("stego uint8: ", np.uint8(stego_image)[0][0])
        self.post_image(stego_image)

        return stego_image

    def decode(self, imageArr):
        print("====decode====")
        message = ""
        max_char = 0

        # modify only for Y (luminance) layer
        idx_channel = 0
        imageArr = self.split_image_to_block(
            np.float64(imageArr)[:, :, idx_channel], self.BLOCK_SIZE)
        stego_image = [np.subtract(block, 128)
                       for block in imageArr]

        # forward dct stage
        dct_blocks = [cv2.dct(block)
                      for block in stego_image]
        # quantize
        dct_blocks = [np.round(block/self.quant_table) for block in dct_blocks]

        for idx, block in enumerate(dct_blocks):
            if max_char != 0 and len(message) == max_char:
                print("last index: ", idx)
                break

            # sort dct coefficient by frequency
            sorted_coef = np.rint(zg.zigzag(block)).astype(int)

            # embed message into DCT coefficient
            message, max_char = self.extract_message(
                sorted_coef, message, max_char)
        print("max char: ", max_char)
        return message


def prep_image(img):
    cover_image = np.float32(img)
    print("prep BGR: ", np.array(img)[0][0])
    print("prep YCbCr: ", np.array(cover_image)[0][0])
    width, height = cover_image.shape[:2]

    # Padding pixel of image
    # # Calculate the number of pixels to pad in the width and height
    # width_padding = 0 if width % 8 == 0 else (8-(width % 8))
    # height_padding = 0 if height % 8 == 0 else (8-(height % 8))
    # print("padding w h: ", width_padding, height_padding)

    # # Create a black pixel with the required padding
    # padded_img = np.zeros((
    #     width + width_padding,
    #     height + height_padding, 3
    # ), dtype=np.float64)

    # # Copy the original image into the padded image
    # padded_img[:width, :height] = cover_image
    # return padded_img

    # Cropped pixel of image
    # Calculate the number of pixels to pad in the width and height
    width_crp = width - (width % 8)
    height_crp = height - (height % 8)
    print("final size of cropped w h: ", width_crp, height_crp)

    # Create a black pixel with the required padding
    ori_img = cover_image
    cropped_img = ori_img[
        :width_crp, :height_crp]

    return ori_img, cropped_img


def PSNR(original, stego):
    mse = np.mean((original - stego) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


if __name__ == "__main__":
    ###########TEXT#############
    # test compression using zlib
    with open('example/text.txt') as f:
        lines = f.readlines()
    originalMessage = str.encode(''.join(lines))
    comp = zlib.compress(originalMessage)
    print("Ori: ", getsizeof(originalMessage))
    print("Comp: ", getsizeof(comp))
    # print("Comp: ", helper.bytes_to_binary(comp))
    ###########TEXT END#############

    ###########ENCODING#############
    image = cv2.imread("example/Lenna.png", flags=cv2.IMREAD_COLOR)
    dctObj = DCT(cover_image=image)
    stego = dctObj.encode(helper.bytes_to_binary(comp))
    # print("size cover: ", image.size)
    # print("PSNR real: ", PSNR(dctObj.image, stego))
    ##########ENCODING END###########

    ###########DECODING###########
    stego2 = cv2.imread("stego_image.png", flags=cv2.IMREAD_COLOR)
    stego2, stego2_cropped = prep_image(stego2)
    print("stego: ", stego[0][0])
    print("stego2: ", stego2[0][0])
    print("stego cropped: ", stego2_cropped[0][0])
    dctObj = DCT(decode=True)
    message = dctObj.decode(stego)
    message2 = dctObj.decode(stego2_cropped)
    # print("message final: ", message)
    print("similarity: ", similar(helper.bytes_to_binary(comp), message))
    print("similarity2: ", similar(helper.bytes_to_binary(comp), message2))

    msg = int(helper.bytes_to_binary(comp)[:40], 2).to_bytes(
        len(helper.bytes_to_binary(comp)[:40]) // 8, byteorder="big")
    # msg = bytes(int(message[i: i + 8], 2) for i in range(0, len(message), 8))
    print("MESSAGE: ", msg)

    # print("similarity bytes: ", similar(comp, msg))
    print(msg[:10])
    print(comp[:10])
    # print("size: ", getsizeof(originalMessage))
    # print(zlib.decompress(msg))
    ###########DECODING END###########

    # print("float: ", helper.bytes_to_binary(struct.pack('>f', 19)))
    # print("float: ", helper.bytes_to_binary(struct.pack('>f', 19.90)))
    # print("float: ", helper.bytes_to_binary(struct.pack('>f', -19.90)))

    # print("\nPSNR: ", PSNR(imageArr, np.round(cv2.idct(np.float32(quantized)))))
    # print("PSNR: ", PSNR(imageArr, np.round(cv2.idct(np.float32(dequantized)))))
    # print("\nPSNR: ", PSNR(imageArr, np.round(
    #     cv2.idct(np.float32(dct)))))
