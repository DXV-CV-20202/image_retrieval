import cv2

class HOG():
    winSize = (32, 32) # _winSize=(img.shape[1] // cell_size[1] * cell_size[1], img.shape[0] // cell_size[0] * cell_size[0]),
    blockSize = (32, 32) # _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
    blockStride = (2, 2) # _blockStride=(cell_size[1], cell_size[0])
    cellSize = (16, 16) #  _cellSize=(cell_size[1], cell_size[0]),
    nbins = 9

    def __init__(self):
        self.extractor = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins)

    def compute(self, image):
        descriptor = self.extractor.compute(image)
        return descriptor