import numpy as np
from PIL import Image
import cv2
import math
from bitarray import bitarray
# from app.algo.zigzag import zigzag as zg


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

    def __init__(self, cover_image):
        self.quant_table = self.QUANTIZATION_TABLE
        self.image = cover_image
        self.channel = [
            self.split_image_to_block(cover_image[:, :, 0], self.BLOCK_SIZE),
            self.split_image_to_block(cover_image[:, :, 1], self.BLOCK_SIZE),
            self.split_image_to_block(cover_image[:, :, 2], self.BLOCK_SIZE),
        ]

    def split_image_to_block(self, image, block_size):
        blocks = []
        for vert_slice in np.vsplit(image, int(image.shape[0] / block_size)):
            for horz_slice in np.hsplit(vert_slice, int(image.shape[1] / block_size)):
                blocks.append(horz_slice)
        return blocks

    def dct(block, verbose=False):
        foo = 0.0
        for u in range(8):
            cu, cv = 1, 1
            if u == 0:
                cu = float(1 / math.sqrt(2))
            for v in range(8):
                cv = 1
                if v == 0:
                    cv = float(1 / math.sqrt(2))

                a = cu * cv * block[u][v]
                b = math.cos((u*math.pi)/16) * math.cos((v*math.pi)/16)
                foo += (a * b)
                if verbose:
                    print(a * b, end=" + ")
        foo *= 0.25
        return foo

    def quantize_blocks(block, verbose=False):
        quantized = []
        for i in range(8):
            quantized.append([])
            for j in range(8):
                exp = round(block[i][j] / QUANTIZATION_TABLE[i][j])
                quantized[i].append(exp)
        if verbose:
            print("quantized block: ", quantized, end="\n")

    def dequantize_block(block, verbose=False):
        dequantized = []
        for i in range(8):
            dequantized.append([])
            for j in range(8):
                back = block[i][j] * QUANTIZATION_TABLE[i][j]
                dequantized[i].append(back)
        if verbose:
            print("dequantized block: ", dequantized, end="\n")

    def encode(self, message):
        stego_image = np.empty_like(image)
        for idx, ch in enumerate(self.channel):
            # forward dct stage
            dct_blocks = [cv2.dct(block) for block in ch]
            print(dct_blocks)

            # quantization stage
            dct_quants = [np.around(np.divide(item, self.quant_table))
                          for item in dct_blocks]

            # sort dct coefficient by frequency
            # sorted_coef = [zg.zigzag(block) for block in dct_quants]

            if idx == 0:
                secret_data = ""

    def decode(image):
        pass


def prep_image(img):
    cover_image = img.convert('YCbCr')
    height, width = cover_image.size

    # Calculate the number of pixels to pad in the width and height
    width_padding = 8 - (width % 8)
    height_padding = 8 - (height % 8)

    # Create a black image with the required padding
    padded_img = np.zeros((
        height + height_padding,
        width + width_padding, 3
    ), dtype=np.uint8)

    # Copy the original image into the padded image
    padded_img[:height, :width] = img

    return padded_img


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


if __name__ == "__main__":
    image = Image.open("example/Lenna.png")
    # imageArr = prep_image(image)
    imageArr = np.float32([
        [45,	45,	32,	60,	44,	52,	42, 31],
        [96,	94,	32,	31,	89,	31,	49,	33],
        [12,	41,	22,	50,	96,	47,	47,	78],
        [18,	21,	19,	30,	76,	47,	87,	93],
        [22,	71,	64,	34,	71,	47,	80,	94],
        [32,	40,	56,	70,	95,	41,	72,	42],
        [44,	10,	93,	18, 85,	51,	95,	93],
        [12,	44,	13,	39,	71,	81,	70,	71]
    ])
    # quantizedEmbed = np.float32([[46, 1, 2, 0, 0, 1, 1, 1],
    #                              [0, 0, 0, 1, 1, 1, 1, 0],
    #                              [0, 0, 0, 1, 0, 1, 0, 0],
    #                              [0, 1, 1, 0, 0, 0, 0, 0],
    #                              [1, 0, 1, 0, 0, 0, 0, 0],
    #                              [1, 1, 0, 0, 0, 0, 0, 0],
    #                              [1, 1, 0, 0, 0, 0, 0, 0],
    #                              [0, 0, 0, 0, 0, 0, 0, 0]])
    QUANTIZATION_TABLE = [[16,  11,  10,  16,  24,  40,  51,  61],
                          [12,  12,  14,  19,  26,  58,  60,  55],
                          [14,  13,  16,  24,  40,  57,  69,  56],
                          [14,  17,  22,  29,  51,  87,  80,  62],
                          [18,  22,  37,  56,  68, 109, 103,  77],
                          [24,  35,  55,  64,  81, 104, 113,  92],
                          [49,  64,  78,  87, 103, 121, 120, 101],
                          [72,  92,  95,  98, 112, 100, 103,  99]]

    idct = np.round(cv2.idct(np.float32(dequantized)))
    print("==============")
    # for i in idct:
    #     print("[", end="")
    #     for j in i:
    #         print("%d" % j, end=",")
    #     print("],")
    # dct = DCT(dequantized)
    # dct.encode()
    # print(idct)

    print("\nPSNR: ", PSNR(imageArr, np.round(cv2.idct(np.float32(quantized)))))
    print("PSNR: ", PSNR(imageArr, np.round(cv2.idct(np.float32(dequantized)))))
    print("\nPSNR: ", PSNR(imageArr, np.round(
        cv2.idct(np.float32(dct)))))
