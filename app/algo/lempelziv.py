from sys import getsizeof
import re


class LZ77Compressor:
    """
    A simplified implementation of the LZ77 Compression Algorithm
    """
    MAX_WINDOW_SIZE = 400

    def __init__(self, window_size=20):
        self.window_size = min(window_size, self.MAX_WINDOW_SIZE)
        self.lookahead_buffer_size = 15  # length of match is at most 4 bits

    def compress(self, data, verbose=False):
        """
        Given text to compressed by applying a simple LZ77 compression algorithm.

        The compressed format is:
        pair of symbol (x, y, z) which mean x is flag, y is distance, z is symbol.

        Returned as a string of symbols

        if verbose is enabled, the compression description is printed to standard output
        """
        i = 0
        out_data = []

        while i < len(data):
            match = self.findLongestMatch(data, i)
            symbol = ""

            if match:
                # Add 1 bit flag
                (bestMatchDistance, bestMatchLength) = match

                symbol = "(1,{},{})".format(
                    bestMatchDistance, bestMatchLength)
                if verbose:
                    print(symbol, end='')

                i += bestMatchLength

            else:
                # No useful match was found. Add 0 bit flag
                symbol = "(0,{})".format(data[i])
                if verbose:
                    print(symbol, end='')

                i += 1
            out_data.append(symbol)

        # return the compressed data
        # return out_data
        return "".join(out_data)

    def decompress(self, data):
        """
        Given a string of the compressed data, the data is decompressed back to its
        original form the decompressed data is returned as a string
        """
        out_data = []
        pattern = "(\([^)]+\))"  # look for data inside parenthesis
        symbols = re.findall(pattern, data)
        print(symbols)

        for symbol in symbols:
            ext = symbol[1:-1].split(",")
            flag = False if ext[0] == "0" else True

            if not flag:
                character = ext[1]
                out_data.append(character)
            else:
                distance = int(ext[1])
                length = int(ext[2])

                for _ in range(length):
                    out_data.append(out_data[-distance])

        # return the decompressed data
        # return out_data
        return "".join(out_data)

    def findLongestMatch(self, data, current_position):
        """
        Finds the longest match to a substring starting at the current_position
        in the lookahead buffer from the history window
        """
        end_of_buffer = min(current_position +
                            self.lookahead_buffer_size, len(data) + 1)

        best_match_distance = -1
        best_match_length = -1

        for i in range(current_position + 1, end_of_buffer):

            start_index = max(0, current_position - self.window_size)
            substring = data[current_position:i]

            for j in range(start_index, current_position):

                repetitions = len(substring) // (current_position - j)

                last = len(substring) % (current_position - j)

                matched_string = data[j:current_position] * \
                    repetitions + data[j:j+last]

                if matched_string == substring and len(substring) > best_match_length:
                    best_match_distance = current_position - j
                    best_match_length = len(substring)

        if best_match_distance > 0 and best_match_length > 0:
            return (best_match_distance, best_match_length)
        return None


if __name__ == "__main__":
    text = """
    Given a string of the compressed data, the data is decompressed back to its
    original form, and written into the output file path if provided.
    the decompressed data is returned as a string. Given a string of the compressed data, the data is decompressed back to its
    original form, and written into the output file path if provided.
    the decompressed data is returned as a string"""

    compressor = LZ77Compressor(window_size=100)  # window_size is optional
    # or assign compressed data into a variable
    compressed_data = compressor.compress(text, False)

    print("\n")
    print("text: ", text, end="\n")
    print("compressed: ", compressed_data, end="\n")
    print("size: ", getsizeof(text))
    print("size: ", getsizeof(compressed_data))

    decoded = compressor.decompress(compressed_data)
    print("decoded: ", decoded)
