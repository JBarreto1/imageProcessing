import csv
import numpy as np
import matplotlib.pyplot as plt

def initParams():
    weights1 = np.random.rand(16,784) - 0.5 #16 rows, 784 col
    bias1 = np.random.rand(16, 1) - 0.5
    weights2 = np.random.rand(16,16) - 0.5
    bias2 = np.random.rand(16, 1) - 0.5
    weights3 = np.random.rand(10,16) - 0.5 #10 rows, 16 col
    bias3 = np.random.rand(10, 1) - 0.5
    return weights1,bias1,weights2,bias2,weights3,bias3

def forwardProp(pixels,w1,b1,w2,b2,w3,b3):
    # print('pixels',pixels.shape)
    Z1 = w1.dot(pixels)+b1
    A1 = RELU(Z1)
    # print('layer1',A1.shape)
    Z2 = w2.dot(A1)+b2
    A2 = RELU(Z2)
    # print('layer2',A2.shape)
    Z3 = w3.dot(A2)+b3
    Z3[np.abs(Z3) > 700] = 700 #getting overflow error because the softmax exp function can't handle big values. Anything over 700 will turn out to be 1 anyway
    A3 = softmax(Z3)
    # print('layer3',A3.shape)
    return Z1,A1,Z2,A2,Z3,A3

def backProp(m,Z1,A1,Z2,A2,Z3,A3,w2,w3,X,target):
    hotTar = targetArr(target)
    dZ3 = A3 - hotTar
    dW3 = 1 / m * dZ3.dot(A2.T)
    dB3 = 1 / m * np.sum(dZ3)
    dZ2 = w3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2)
    dZ1 = w2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = 1 / m * np.sum(dZ1)
    return dW1,dB1,dW2,dB2,dW3,dB3

def sigmoid(value):
    return 1/(1+np.exp(-value))

def softmax(x):
    A = np.exp(x) / sum(np.exp(x))
    return A

def RELU(value):
    return np.maximum(value,0)

def ReLU_deriv(value):
    return value > 0

def targetArr(target):
    arr = np.zeros((target.size, 10))
    arr[np.arange(target.size), target] = 1
    arr = arr.T
    return arr

def updateParams(w1,b1,w2,b2,w3,b3,dW1,dB1,dW2,dB2,dW3,dB3,alpha):
    w1 = w1 + dW1 * alpha
    w2 = w2 + dW2 * alpha
    w3 = w3 + dW3 * alpha
    b1 = b1 + dB1 * alpha
    b2 = b2 + dB2 * alpha
    b3 = b3 + dB3 * alpha
    return w1,b1,w2,b2,w3,b3

def costFunc(arr,target):
    """cost function takes a cost function and the array of output nodes (0-9) and returns the cost"""
    cost = []
    for i in range(10):
        if i == target:
            cost.append((arr[i]-1)**2)
        else:
            cost.append(arr[i]**2)
    return cost

def prediction(A3):
    return np.argmax(A3,0)

def accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def grad_descent(m,X,Y,w1,b1,w2,b2,w3,b3,alpha):
    Z1,A1,Z2,A2,Z3,A3 = forwardProp(X,w1,b1,w2,b2,w3,b3)
    dW1,dB1,dW2,dB2,dW3,dB3 = backProp(m,Z1,A1,Z2,A2,Z3,A3,w2,w3,X,Y)
    w1,b1,w2,b2,w3,b3 = updateParams(w1,b1,w2,b2,w3,b3,dW1,dB1,dW2,dB2,dW3,dB3,alpha)
    predictions = prediction(A3)
    acc = accuracy(predictions,Y)
    return w1,b1,w2,b2,w3,b3,acc

def readData(dataLoc):
    data = []
    with open(dataLoc) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader) #strip the first row because its just the header/names of columns
        # np.random.shuffle(X) Might want to shuffle later on for better results
        for row in csv_reader:
            # target = int(row[0]) #first item in the row is the target
            # data.append([target])
            # imgTuple = (int(row[0]),int(row[1:])/255)
            # pixels = row[1:]
            # # pixels = [data[-1].append((int(i)/255)) for i in pixels]
            pixels = [int(i) for i in row]
            data.append(pixels)
            # X.append(pixels)
    return data

def testModel(data,w1,b1,w2,b2,w3,b3):
    nCorrect = 0
    nIncorrect = 0
    for i in data:
        X = np.array([i[1:] / 255])
        X = X.T
        _,_,_,_,_,A3 = forwardProp(X,w1,b1,w2,b2,w3,b3)
        # print(A3,prediction(A3),i[0])
        if prediction(A3)[0] == i[0]:
            nCorrect += 1
        else:
            nIncorrect += 1
    return nCorrect,nIncorrect

def main(alpha,iterations,batchSize):
    w1,b1,w2,b2,w3,b3 = initParams()
    accList = []
    data = readData('../train.csv')
    allData = np.array(data)
    data = allData[0:30000]
    for j in range(iterations):
        np.random.shuffle(data)
        dataTrans = data.T
        X = np.array(dataTrans[1:])
        Y = np.array(dataTrans[0])
        X = X.T
        X = X / 255
        m, n = X.shape
        if j % 10:
            print("iterations: ", j)
        for i in range(int(m/batchSize)+1):
            miniBatchX = X[i*batchSize:(i+1)*batchSize]
            miniBatchY = Y[i*batchSize:(i+1)*batchSize]
            if len(miniBatchX) > 0:
                miniBatchX = miniBatchX.T
                w1,b1,w2,b2,w3,b3,_ = grad_descent(batchSize,miniBatchX,miniBatchY,w1,b1,w2,b2,w3,b3,alpha)
                # accList.append(acc)
                if np.isnan(w1).any():
                    print(i)
                    break
    # plt.plot(accList, linewidth=2)
    # plt.show()
    print(testModel(allData[30000:30300],w1,b1,w2,b2,w3,b3))
    return w1,b1,w2,b2,w3,b3

if __name__ == '__main__':
    batchSize = 20
    iterations = 100
    alpha = 0.05
    params = main(alpha,iterations,batchSize)
    with open('paramsOutput.csv', mode='w') as output:
        output_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in params:
            output_writer.writerow(i)
