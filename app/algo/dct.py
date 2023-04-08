import numpy as np
import math
from app.algo.helper import *
# from helper import *
import zlib
import cv2
from bitarray import bitarray
from sys import getsizeof
from difflib import SequenceMatcher


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

    def __init__(self, is_decode=False, cover_image=None):
        self.quant_table = np.float32(self.QUANTIZATION_TABLE)/4
        self.ori_img, self.image = None, None
        self.height, self.width = 0, 0
        self.ori_width, self.ori_height = 0, 0
        self.channel = None
        self.message = ""
        if not is_decode:
            self.set_cover_image(cover_image)

    def set_message(self, message):
        self.message = message

    def set_cover_image(self, cover_image):
        self.ori_img, self.image = prep_image(cover_image, conv=True)
        self.height, self.width = self.image.shape[:2]
        self.ori_width, self.ori_height = cover_image.shape[:2]
        self.channel = [
            np.float32(split_image_to_block(
                self.image[:, :, 0], self.BLOCK_SIZE)),
            np.float32(split_image_to_block(
                self.image[:, :, 1], self.BLOCK_SIZE)),
            np.float32(split_image_to_block(
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
                        # if u == 0:
                        #     print("LOL: ", u, v, (block[u][v] * cu * cv * val))
                        foo += (block[u][v] * cu * cv * val)

                foo = round(0.25 * foo)
                dctMat[x].append(foo)

                if verbose:
                    print(block[x][y] * val, end=" + ")
        return dctMat

    def embed_message(self, block, encoded_data):
        indexes = [((5, 0), (3, 3)), ((2, 2), (1, 4)),
                   ((0, 4), (3, 2)), ((4, 1), (4, 2))]
        for si, ei in indexes:
            if len(encoded_data) == 0:
                break

            # Swapping DCT
            if block[si[0]][si[1]] == block[ei[0]][ei[1]] and encoded_data[0] == "1":
                block[si[0]][si[1]] += 1.55
            if (encoded_data[0] == "0" and block[si[0]][si[1]] < block[ei[0]][ei[1]]) or (encoded_data[0] == "1" and block[si[0]][si[1]] >= block[ei[0]][ei[1]]):
                block[si[0]][si[1]], block[ei[0]][ei[1]
                                                  ] = block[ei[0]][ei[1]], block[si[0]][si[1]]
            encoded_data = encoded_data[1:]

        return encoded_data

    def extract_message(self, block, message, max_char):
        indexes = [((5, 0), (3, 3)), ((2, 2), (1, 4)),
                   ((0, 4), (3, 2)), ((4, 1), (4, 2))]
        i = 0 if message == "" else len(message)
        for si, ei in indexes:
            if max_char != 0 and max_char == len(message):
                break

            # Swapping DCT
            encode_bit = "0" if block[si[0]][si[1]
                                             ] >= block[ei[0]][ei[1]] else "1"
            message += encode_bit

            if i >= 31 and max_char == 0:
                max_char = binary_to_int(message[:32])
                message = message[32:]
            i += 1
        return message, max_char

    def post_image(self, stego):
        # retrieve cropped pixel
        image = np.copy(self.ori_img)
        image[:stego.shape[0], :stego.shape[1]] = stego
        image = cv2.cvtColor(np.float32(image), cv2.COLOR_YCR_CB2BGR)
        image = np.uint8(np.clip(image, 0, 255))
        cv2.imwrite("stego_image.png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return image

    def encode(self, message):
        if message == None:
            return

        # add length of message as binary
        message = int_to_binary(len(message), True) + message
        if getsizeof(binary_to_bytes(message)) > max_bit_cap(self.width, self.height, 4)/8:
            raise Exception("not enough block\n")

        # index for embedded
        idx_channel = 1
        dct_blocks = self.channel[idx_channel]

        dct_blocks = [np.subtract(block, 128)
                      for block in dct_blocks]

        # forward to dct stage
        dct_blocks = [cv2.dct(block.astype(float))
                      for block in dct_blocks]

        # quantize
        dct_blocks = [np.round(block/self.quant_table) for block in dct_blocks]

        for idx, block in enumerate(dct_blocks):
            if len(message) == 0:
                break

            # embed message into DCT coefficient
            message = self.embed_message(block, message)

        if len(message) != 0:
            raise Exception("not enough block\n")

        embedded_block = [block * self.quant_table for block in dct_blocks]
        embedded_block = [cv2.idct(block)
                          for block in embedded_block]
        embedded_block = [np.add(block, 128) for block in embedded_block]

        stego_channel = []
        for i in range(3):
            if idx_channel == i:
                stego_channel.append(embedded_block)
                continue
            stego_channel.append(self.channel[i])
        stego_image = array_to_image(self.image, stego_channel)

        # saving image
        stego_image = self.post_image(stego_image)

        return stego_image

    def decode(self, imageArr):
        message = ""
        max_char = 0

        # modify only for Cr layer
        idx_channel = 1
        imageArr = split_image_to_block(
            np.float32(imageArr)[:, :, idx_channel], self.BLOCK_SIZE)
        imageArr = [np.subtract(block, 128)
                    for block in imageArr]

        # forward dct stage
        dct_blocks = [cv2.dct(block.astype(float))
                      for block in imageArr]

        # quantize
        dct_blocks = [np.round(block/self.quant_table) for block in dct_blocks]

        for _, block in enumerate(dct_blocks):
            if max_char != 0 and len(message) == max_char:
                break

            # embed message into DCT coefficient
            message, max_char = self.extract_message(
                block, message, max_char)
        return message


def prep_image(img, conv=True):
    ori_img = np.float32(img)
    if conv:
        imgconv = cv2.cvtColor(ori_img, cv2.COLOR_BGR2YCR_CB)
    width, height = imgconv.shape[:2]

    # Cropped pixel of image
    # Calculate the number of pixels to crop
    width_crp = width - (width % 8)
    height_crp = height - (height % 8)

    # remove pixel out of range width_crp and height_crp
    cropped_img = imgconv[
        :width_crp, :height_crp]

    return ori_img, cropped_img


if __name__ == "__main__":
    ########### TEXT#############
    # test compression using zlib
    with open('example/bulphrek.txt') as f:
        lines = f.readlines()
    originalMessage = str.encode(''.join(lines))
    comp = zlib.compress(originalMessage)
    print("Ori: ", getsizeof(originalMessage))
    print("Comp: ", getsizeof(comp))

    # penyisipan citra
    # with open('example/rafly.png', 'rb') as file_image:
    #     f = file_image.read()
    # comp = zlib.compress(f)
    # print("Ori: ", getsizeof(f))
    # print("Comp: ", getsizeof(comp))
    # print("Comp: ", bytes_to_binary(comp))
    ########### TEXT END#############

    ########### ENCODING#############
    coverImage = cv2.imread("example/animal.jpg", flags=cv2.IMREAD_COLOR)
    dctObj = DCT(cover_image=coverImage)
    # matn = np.rint(cv2.dct(np.rint(cv2.idct(np.float32(dctObj.QUANTIZATION_TABLE2)
    # * dctObj.QUANTIZATION_TABLE)))/dctObj.QUANTIZATION_TABLE)
    # mmm = "\\left[\\begin{matrix}"
    # for x in matn:
    #     for y in x:
    #         mmm += "{}&".format(round(y))
    #         print("{:.3f}".format(y), end=" ")
    #     mmm += "\\\\"
    #     print("\n")
    # mmm += "\\\\\end{matrix}\\right]"
    # print("MES: ", mmm)
    print("MAX BYTE CAPACITY: ", max_bit_cap(
        dctObj.image.shape[0], dctObj.image.shape[1], 4)/8)
    try:
        stego = dctObj.encode(bytes_to_binary(comp))
    except Exception as err:
        print("Unexpected error: ", err)
        stego = None
    # print("size cover: ", image.size)
    # print("PSNR real: ", PSNR(dctObj.image, stego))
    ########## ENCODING END###########

    ########### DECODING###########
    stego2 = cv2.imread("stego_image.png", flags=cv2.IMREAD_COLOR)
    _, stego2_cropped = prep_image(stego2)
    # print("stego: ", stego[0][0])
    # print("stego2: ", stego2[0][0])
    # print("stego cropped: ", stego2_cropped[0][0])
    dctObj = DCT(is_decode=True)
    message = dctObj.decode(prep_image(stego)[1])
    message2 = dctObj.decode(stego2_cropped)
    # print("message final: ", message)
    # stego2 = cv2.cvtColor(np.float32(stego2), cv2.COLOR_YCR_CB2BGR)
    print('PSNR: ', PSNR(coverImage, stego2))
    print("similarity: {:.10f}".format(similar(
        bytes_to_binary(comp), message)))
    print("similarity2: {:.10f}".format(similar(
        bytes_to_binary(comp), message2)))

    msg = binary_to_bytes(message2)

    # print("similarity bytes: {:.5f}".format(
    #     similar(comp, msg)))
    # print("=====\nmessage extracted: \n",
    #       zlib.decompress(msg).decode("UTF-8")[:100])

    coverImage2 = cv2.imread("example/rafly.png", flags=cv2.IMREAD_COLOR)
    msg = zlib.decompress(msg)
    jpg_as_np = np.frombuffer(msg, dtype=np.uint8)
    img_np = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    print("PSNR sisip: ", PSNR(coverImage2, img_np))
    ########### DECODING END###########
