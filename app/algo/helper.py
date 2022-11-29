def strToBinary(str):
    binary = []
    for s in str:
        binary.append(int(bin(ord(s))[2:]))

    return binary
