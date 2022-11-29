# from app.model.data import Data
# from app.view.template import head, body
# from app.algo.dct import encoding
from app.algo.huffman import huffmanEncoding


def main():
    # data = Data()
    # head()
    # body(data)
    text = "gambar ini adalah gambar gambar gambar gambar gambar gambar ini adalah gambar gambar gambar gambar gambar"
    tree, encoded_data = huffmanEncoding(text)
    print("size: ", getsizeof(text))
    print("size: ", getsizeof(encoded_data))


main()
