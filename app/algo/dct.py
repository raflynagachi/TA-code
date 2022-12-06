import numpy as np
from PIL import Image
import cv2
from app.algo.zigzag import zigzag as zg


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
        self.height, self.width = cover_image.shape[:2]
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

    def encode(self, message):
        stego_image = np.empty_like(image)
        for idx, ch in enumerate(self.channel):
            # forward dct stage
            dct_blocks = [cv2.dct(block) for block in ch]

            # quantization stage
            dct_quants = [np.around(np.divide(item, self.quant_table))
                          for item in dct_blocks]

            # sort dct coefficient by frequency
            sorted_coef = [zg.zigzag(block) for block in dct_quants]

            if idx == 0:
                secret_data = ""

    def decode(image):
        pass


def prep_image(image):
    cover_image = image.convert('YCbCr')
    height, width = cover_image.size

    # force image dimension to be 8x8 compliant
    while(height % 8):
        height += 1
    while(width % 8):
        width += 1
    validDim = (width, height)
    cover_image = cover_image.resize(validDim)

    imageArr = np.array(cover_image)
    return imageArr


if __name__ == "__main__":
    image = Image.open("example/Lenna.png")
    imageArr = prep_image(image)
    dct = DCT(imageArr)
    print("width and height: ", dct.width, dct.height)
    print("dct channel: ", dct.channel)
