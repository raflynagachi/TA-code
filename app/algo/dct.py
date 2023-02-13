import numpy as np
import math
from app.algo import zigzag as zg
from app.algo import helper
# import zigzag as zg
# import helper
import zlib
import struct
import cv2
from bitarray import bitarray
from sys import getsizeof, byteorder
from difflib import SequenceMatcher

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
    HAMMING = 4
    QUANTIZATION_TABLE = [[16,  11,  10,  16,  24,  40,  51,  61],
                          [12,  12,  14,  19,  26,  58,  60,  55],
                          [14,  13,  16,  24,  40,  57,  69,  56],
                          [14,  17,  22,  29,  51,  87,  80,  62],
                          [18,  22,  37,  56,  68, 109, 103,  77],
                          [24,  35,  55,  64,  81, 104, 113,  92],
                          [49,  64,  78,  87, 103, 121, 120, 101],
                          [72,  92,  95,  98, 112, 100, 103,  99]]

    def __init__(self, decode=False, cover_image=None):
        self.quant_table = np.float32(self.QUANTIZATION_TABLE)
        self.ori_img, self.image = None, None
        self.height, self.width = 0, 0
        self.ori_width, self.ori_height = 0, 0
        self.channel = None
        self.message = ""
        if not decode:
            self.ori_img, self.image = prep_image(cover_image, conv=True)
            self.height, self.width = self.image.shape[:2]
            self.ori_width, self.ori_height = cover_image.shape[:2]
            self.channel = [
                np.float32(helper.split_image_to_block(
                    self.image[:, :, 0], self.BLOCK_SIZE)),
                np.float32(helper.split_image_to_block(
                    self.image[:, :, 1], self.BLOCK_SIZE)),
                np.float32(helper.split_image_to_block(
                    self.image[:, :, 2], self.BLOCK_SIZE))
            ]

    def set_message(self, message):
        self.message = message

    def set_cover_image(self, cover_image):
        self.ori_img, self.image = prep_image(cover_image, conv=True)
        self.height, self.width = self.image.shape[:2]
        self.ori_width, self.ori_height = cover_image.shape[:2]
        self.channel = [
            np.float32(helper.split_image_to_block(
                self.image[:, :, 0], self.BLOCK_SIZE)),
            np.float32(helper.split_image_to_block(
                self.image[:, :, 1], self.BLOCK_SIZE)),
            np.float32(helper.split_image_to_block(
                self.image[:, :, 2], self.BLOCK_SIZE))
        ]

    def reset(self):
        self.ori_img, self.image = None, None
        self.height, self.width = 0, 0
        self.ori_width, self.ori_height = 0, 0
        self.channel = None
        self.message = ""

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
        start_idx, end_idx = 14, 24
        i = 0
        # print("block: ", block[start_idx:end_idx])
        for item in block[start_idx: end_idx]:
            if len(encoded_data) == 0:
                break

            embed = helper.int_to_binary(item * 2, True)
            embed = embed[:-1] + encoded_data[0]
            embed = helper.binary_to_int(embed)
            block[start_idx + i] = embed

            encoded_data = encoded_data[1:]
            i += 1
        return block, encoded_data

    def extract_message(self, block, message, max_char):
        start_idx, end_idx = 14, 24
        i = 0 if message == "" else len(message)
        # print("block: ", block[start_idx:end_idx])
        for item in block[start_idx: end_idx]:
            if max_char != 0 and max_char == len(message):
                break

            embed = helper.int_to_binary(item // 2, True)
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
        # image = cv2.cvtColor(np.float32(image), cv2.COLOR_YCR_CB2BGR)
        # image = np.uint8(np.clip(image, 0, 255))
        cv2.imwrite("stego_image.png", image)
        print("stego ori2 BGR: ", np.array(image)[0][0])
        return image

    def encode(self, message):
        if message == None:
            return

        # index for embedded
        idx_channel = 0
        message = helper.hamming_encode(message, self.HAMMING)
        message = helper.int_to_binary(len(message), True) + message

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
        stego_image = helper.array_to_image(self.image, stego_channel)
        print("stego ori: ", stego_image[0][0])

        # saving image
        stego_image = self.post_image(stego_image)

        return stego_image

    def decode(self, imageArr):
        print("====decode====")
        message = ""
        max_char = 0

        # modify only for Y (luminance) layer
        idx_channel = 0
        imageArr = helper.split_image_to_block(
            np.float32(imageArr)[:, :, idx_channel], self.BLOCK_SIZE)
        stego_image = [np.subtract(block, 128)
                       for block in imageArr]

        # forward dct stage
        dct_blocks = [cv2.dct(block.astype(float))
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
        return helper.hamming_decode(message, self.HAMMING)


def prep_image(img, conv=False):
    ori_img = np.float32(img)
    if conv:
        # ori_img = cv2.cvtColor(np.float32(ori_img), cv2.COLOR_BGR2YCR_CB)
        pass
    width, height = ori_img.shape[:2]

    # Cropped pixel of image
    # Calculate the number of pixels to pad in the width and height
    width_crp = width - (width % 8)
    height_crp = height - (height % 8)
    print("SHAPE: ", ori_img.shape)
    print("final size of cropped w h: ", width_crp, height_crp)

    # Create a black pixel with the required padding
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
    image = cv2.imread("example/background.png", flags=cv2.IMREAD_COLOR)
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
    message = dctObj.decode(prep_image(stego)[1])
    message2 = dctObj.decode(stego2_cropped)
    # print("message final: ", message)
    print('PSNR: ', PSNR(image, stego2))
    print("similarity: ", similar(helper.bytes_to_binary(comp), message))
    print("similarity2: ", similar(helper.bytes_to_binary(comp), message2))

    msg = helper.binary_to_bytes(message2)

    print("similarity bytes: ", similar(comp, msg))
    print(zlib.decompress(msg).decode("UTF-8")[:100])
    ###########DECODING END###########

    # print("float: ", helper.bytes_to_binary(struct.pack('>f', 19)))
    # print("float: ", helper.bytes_to_binary(struct.pack('>f', 19.90)))
    # print("float: ", helper.bytes_to_binary(struct.pack('>f', -19.90)))

    # print("\nPSNR: ", PSNR(imageArr, np.round(cv2.idct(np.float32(quantized)))))
    # print("PSNR: ", PSNR(imageArr, np.round(cv2.idct(np.float32(dequantized)))))
    # print("\nPSNR: ", PSNR(imageArr, np.round(
    #     cv2.idct(np.float32(dct)))))
