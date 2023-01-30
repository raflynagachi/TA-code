import numpy as np


def forward_lift(x, y):
    diff = (y - x) % 0xff
    average = (x + (diff >> 1)) % 0xff
    return (average, diff)


def reverse_lift(average, diff):
    x = (average - (diff >> 1)) % 0xff
    y = (x + diff) % 0xff
    return (x, y)


def RGB_to_YCoCg24(rgb):
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    (temp, Co) = forward_lift(red, blue)
    (Y, Cg) = forward_lift(green, temp)
    return (np.dstack((Y, Cg, Co)) * 255.999).astype(np.uint8)


def YCoCg24_to_RGB(ycgco):
    Y, Cg, Co = ycgco[:, :, 0], ycgco[:, :, 1], ycgco[:, :, 2]
    (green, temp) = reverse_lift(Y, Cg)
    (red, blue) = reverse_lift(temp, Co)
    return (np.dstack((red, green, blue)) * 255.999).astype(np.uint8)
