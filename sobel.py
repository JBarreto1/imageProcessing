from PIL import Image
import numpy as np
import copy
import kernelAdjust as ka

#Sobel only makes sense on a grayscale image

#apply a gaussian blur

#apply the kernel in the x direction
#apply the kernel in y
#calculate the magnitude of the kernel (0-255)

def edge(imArray):
    """blur the image according to a gaussian kernel"""
    s = (len(imArray),len(imArray[0]))
    s1 = (len(imArray),len(imArray[0]),3)
    newImageX = np.zeros(s)
    newImageY = np.zeros(s)
    newImage = np.zeros(s1)
    kernelX = [[-1,0,1],[-2,0,2],[-1,0,1]]
    kernelY = [[-1,-2,-1],[0,0,0],[1,2,1]]
    totalKernel = 1 # sum this for gaussian, but the sum for sobel operator would be zero (divide by this later)
    color = False
    maxRadian = np.arctan(1)
    for i in range(0,len(imArray)):
        for j in range(0,len(imArray[0])):
            newImageX[i][j] = ka.blurPixel(i,j,imArray,kernelX,totalKernel,color)
            newImageY[i][j] = ka.blurPixel(i,j,imArray,kernelY,totalKernel,color)
            result = np.sqrt(np.square(newImageX[i][j]) + np.square(newImageY[i][j]))
            newImage[i][j][0] = np.interp(result, (0, 1442), (0, 255)) #if there's a full black to white edge the max kernel result is sqrt(2* (4*255)^2)
            gx = int(newImageX[i][j])
            gy = int(newImageY[i][j])
            if gx == 0:
                gx = 1
            tangent = gy/gx
            tanInterp = np.interp(tangent, (0, 1020), (0, 1))
            angle = np.arctan(tanInterp)
            angleInterp = int(np.interp(angle, (0, maxRadian), (0, 255)))
            newImage[i][j][1] = angleInterp
            newImage[i][j][2] = angleInterp
    # both = [newImage,newImageX,newImageY]
    return newImage

def main():
    image = Image.open('koala.jpeg').convert('L')

    imArray = np.asarray(image)
    # imArray = [[[151, 127, 81],[144, 120,  74],[135, 111,  67]],
    #             [[1,2,3],[4,5,6],[7,8,9]],
    #             [[0,0,0],[255,255,255],[50,200,200]]]
    # print(imArray)
    # imArray = [[0,0,0],[255,255,255],[50,200,200]]
    color = False
    blurred = ka.gaussianBlur(imArray,2,color)
    # print(blurred)
    filteredArray = np.asarray(blurred)
    image2 = Image.fromarray((filteredArray).astype(np.uint8))
    
    # image2 = Image.fromarray((filteredArray).astype(np.uint8))
    filteredArray = np.asarray(edge(blurred))
    image3 = Image.fromarray((filteredArray).astype(np.uint8))
    # filteredArray = np.asarray(blurred)
    # image2 = Image.fromarray((filteredArray).astype(np.uint8))

    # filteredArray = np.asarray(imageGrayscale(test))
    # image2 = Image.fromarray((filteredArray).astype(np.uint8))

    # summarize image details
    # print(image2.mode)
    # print(image2.size)

    #show the image
    image.show()
    image2.show()
    image3.show()


#gray scale the image
#I spent a long time trying to grayscale the image, but PIL has a built in function I'll use (.convert('L') piece above)
def imageGrayscale(imArray):
    grayIM = []
    scaleMin = 0
    scaleMax = int(np.sqrt(np.square(255)+np.square(255)+np.square(255)))
    for i in range(len(imArray)):
        row = []
        for j in range(len(imArray[0])):
            # newPix = np.sqrt(int(np.square(imArray[i][j][0]))+int(np.square(imArray[i][j][1]))+int(np.square(imArray[i][j][2])))
            # newPixInterp = int(np.interp(newPix, (scaleMin, scaleMax), (0, 255)))
            # row.append(newPixInterp)
            # newPix = int(imArray[i][j][0])+int(imArray[i][j][1])+int(imArray[i][j][2]) / 3
            #Gamma distrbution: calc c linear and then nonlinear gamma dist
            #Clinear = 0.2126 R + 0.7152 G + 0.0722 B
            # Csrgb = 12.92 Clinear when Clinear <= 0.0031308
            # Csrgb = 1.055 Clinear1/2.4 - 0.055 when Clinear > 0.0031308
            clinear = .2126*(int(imArray[i][j][0])/255)+.7152*(int(imArray[i][j][1])/255)+.0722*(int(imArray[i][j][2])/255)
            row.append(newPix)
        grayIM.append(row)
    return grayIM

if __name__ == '__main__':
    main()