class Data:
    message = ""
    image = None
    imageArr = []

    def __init__(self):
        pass

    def setMessage(self, message):
        self.message = message

    def getMessage(self):
        return self.message
    
    def setImage(self, image):
        self.image = image

    def getImage(self):
        return self.image

    def setImageArr(self, imageArr):
        self.imageArr = imageArr

    def getImageArr(self):
        return self.imageArr
