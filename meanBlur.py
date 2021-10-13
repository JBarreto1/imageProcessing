from PIL import Image
import numpy as np

image = Image.open('koala.jpeg')

# print(image.format)
# print(image.size)
# print(image.mode)

imArray = np.asarray(image)

def neighbors(radius, rowNumber, columnNumber, grid):
     return [[grid[i][j] if  i >= 0 and i < len(grid) and j >= 0 and j < len(grid[0]) else [0,0,0]
                for j in range(columnNumber-radius, columnNumber+1+radius)]
                    for i in range(rowNumber-radius, rowNumber+1+radius)]

def findMean(posX,posY,imArray,radius):
    neighborhood = neighbors(radius,posX,posY,imArray)
    total = [0,0,0]
    width = len(neighborhood)
    height = len(neighborhood[0])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                total[k] += neighborhood[i][j][k]
    result = list(map(lambda n: int(n / (width * height)), total))
    return result

def applyFilter(imArray,radius):
    s = (len(imArray),len(imArray[0]),3)
    newImage = np.zeros(s)
    for i in range(0,len(imArray)):
        for j in range(0,len(imArray[0])):
            newImage[i][j] = findMean(i,j,imArray,radius)
    return newImage

test = [[[151, 127, 81],[144, 120,  74],[135, 111,  67]],
  [[1,2,3],[4,5,6],[7,8,9]],
  [[10,11,12],[100,200,200],[200,200,200]]]

filteredArray = np.asarray(applyFilter(imArray,10))
image2 = Image.fromarray((filteredArray).astype(np.uint8))

# summarize image details
# print(image2.mode)
# print(image2.size)

# show the image
image.show()
image2.show()