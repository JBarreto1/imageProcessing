from PIL import Image
import numpy as np
import copy

image = Image.open('koala.jpeg')

# print(image.format)
# print(image.size)
# print(image.mode)

imArray = np.asarray(image)

def neighbors(radius, rowNumber, columnNumber, grid):
     return [[grid[i][j] if  i >= 0 + radius and i < len(grid)-radius and j >= 0+radius and j < len(grid[0]) - radius else [0,0,0]
                for j in range(columnNumber-radius, columnNumber+1+radius)]
                    for i in range(rowNumber-radius, rowNumber+1+radius)]

def blurPixel(posX,posY,imArray,kernel,totalKernel):
    """Given a pixel position and radius, return the blurred pixel according to the weighted kernel"""
    radius = int((len(kernel) - 1) / 2) #if this doesn't line up things don't go well later on
    neighborhood = neighbors(radius,posX,posY,imArray)
    total = [0,0,0]
    width = len(neighborhood)
    height = len(neighborhood[0])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                total[k] += neighborhood[i][j][k] * kernel[i][j]
    result = list(map(lambda n: int(n / totalKernel), total))
    return result

def findMean(posX,posY,imArray,radius):
    """Given a pixel position and radius, return the average pixel weight for the sourrounding"""
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

def meanBlur(imArray,radius):
    """Mean blur an image with a given radius"""
    s = (len(imArray),len(imArray[0]),3)
    newImage = np.zeros(s)
    kernel = [[1]*(radius+2) for i in range(radius+2)]
    totalKernel = np.sum(kernel)
    for i in range(0,len(imArray)):
        for j in range(0,len(imArray[0])):
            newImage[i][j] = blurPixel(i,j,imArray,kernel,totalKernel)
    return newImage

def removeRed(imArray):
    """Remove the red channel of the entire image"""
    s = (len(imArray),len(imArray[0]),3)
    newImage = np.zeros(s)
    for i in range(0,len(imArray)):
        for j in range(0,len(imArray[0])):
            newImage[i][j] = [0,imArray[i][j][1],imArray[i][j][2]]
    return newImage

def gaussKernel(radius):
    """create a gaussian kernel with the given radius"""
    k = []
    for i in range(radius,-1,-1):
        k.append(2**i)
        k.insert(0,2**i)
    k.pop(radius)
    return np.outer(k,k)


def gaussianBlur(imArray,radius):
    """blur the image according to a gaussian kernel"""
    s = (len(imArray),len(imArray[0]),3)
    newImage = np.zeros(s)
    kernel = gaussKernel(radius)
    totalKernel = np.sum(kernel)
    for i in range(0,len(imArray)):
        for j in range(0,len(imArray[0])):
            newImage[i][j] = blurPixel(i,j,imArray,kernel,totalKernel)
    return newImage

test = [[[151, 127, 81],[144, 120,  74],[135, 111,  67]],
  [[1,2,3],[4,5,6],[7,8,9]],
  [[10,11,12],[100,200,200],[200,200,200]]]

# kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
# print(gaussKernel(2))

# print(gaussianBlur(test,1))

# filteredArray = np.asarray(meanBlur(imArray,10))
# filteredArray = np.asarray(removeRed(imArray))
# image2 = Image.fromarray((filteredArray).astype(np.uint8))

filteredArray = np.asarray(gaussianBlur(imArray,2))
image2 = Image.fromarray((filteredArray).astype(np.uint8))

# # summarize image details
# # print(image2.mode)
# # print(image2.size)

# show the image
image.show()
image2.show()