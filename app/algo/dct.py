def encoding(data):
    image = data.getImage()
    message = data.getMessage()

    coverImage = image.convert('YCbCr')
    height, width = coverImage.size

    # force image dimension to be 8x8 compliant
    while(height % 8): height += 1
    while(width % 8): width += 1
    validDim = (width, height)
    coverImage = coverImage.resize(validDim)

    

