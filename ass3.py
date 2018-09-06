import numpy as np 
import matplotlib.pyplot as plt 
import os

PI = 3.1415962

def generateData(numOfCluster, piK, numOfData, miuK, sigmaK):
    # p not enough
    if len(piK) != numOfCluster:
        return -1
    # sum is not 1
    for i in range(1, numOfCluster):
        piK[i] += piK[i-1]
    if piK[-1] != 1:
        return -1
    
    with open('./data.txt', 'w') as f:
        zIdx = np.random.rand(numOfData)
        for i in range(numOfCluster):
            zIdx[zIdx < piK[i]] = int(i+1)
        zIdx = (zIdx - 1).astype(np.int16)
        z = np.eye(numOfCluster)[zIdx].astype(np.int16)

        cnt = [0] * numOfCluster
        for i in range(numOfData):
            cnt[zIdx[i]] += 1
        
        x = []
        for i in range(numOfCluster):
            x.append(0)
            xk = np.random.multivariate_normal(miuK[i], sigmaK[i], size=cnt[i])
            for j in range(len(xk)):
                for k in range(len(xk[j])):
                    xk[j][k] = round(xk[j][k], 4)
            x[-1:] = xk
            print ("miu for ", i, " th data is: ", np.mean(xk))
            
        for i in range(numOfData):
            for j in range(len(x[i])):
                f.write(str(x[i][j]) + ' ')
            f.write('\n')     

        return   

def loadData(baseDir):
    path = os.path.join(baseDir, 'data.txt')
    data = []
    with open(path, 'r') as f:
        for line in f:
            x = line.split(' ')
            x[0] = float(x[0])
            x[1] = float(x[1])
            data.append(x[:2])
    return np.array(data)

def draw(data, classify, title):
    for i in range(len(data)):
        if classify[i] == 0:
            plt.plot(data[i,0], data[i,1], 'r+')
        elif classify[i] == 1:
            plt.plot(data[i,0], data[i,1], 'g+')
        else:
            plt.plot(data[i,0], data[i,1], 'b+')
    plt.title(title)
    plt.show()

def multiGauss(x, miu, sigma):
    global PI

    if miu.ndim < 2:
        miu = miu[np.newaxis]
    if x.ndim < 2:
        x = x[np.newaxis]
    
    dim = miu.shape[1]

    if x.shape[1] != dim:
        return -1
    if sigma.shape[0] != sigma.shape[1] or dim != sigma.shape[0]:
        return -1

    a = np.dot((x - miu), np.mat(sigma).I)
    b = np.diag(np.dot(a, np.transpose(x - miu)))
    f = np.exp(-0.5 * b) / (np.sqrt(np.power(2 * PI, dim) * np.linalg.det(sigma)))

    return np.ravel(f)

class GMMModel():
    def __init__(self, piK, miuK, sigmaK):
        self.piK = piK
        self.miuK = miuK
        self.sigmaK = sigmaK

    def EStep(self, data, numOfCluster):
        N = data.shape[0]

        gama = np.zeros((numOfCluster, N))
        for i in range(numOfCluster):
            gama[i] = self.piK[i] * multiGauss(data, self.miuK[i], self.sigmaK[i])
        sumK = np.sum(gama, axis=0)
        gama = gama/sumK
        self.gamaKN = gama
        self.gamaK = np.sum(self.gamaKN, axis=1)

    def MStep(self, data, numOfCluster):
        N = data.shape[0]
        dim = data.shape[1]

        # new miu
        self.mmiuK = np.dot(self.gamaKN, data) / self.gamaK[:, None]
        # new sigma
        self.msigmaK = np.zeros((numOfCluster, dim, dim))
        for k in range(numOfCluster):
            gamak = self.gamaKN[k,:]
            gamakxmiu = (data - self.miuK[k]) * gamak[:,None]
            xmiu2 = np.dot(np.transpose(gamakxmiu), data - self.miuK[k])
            self.msigmaK[k] = xmiu2 / self.gamaK[k]
        # new pi
        self.mpiK = self.gamaK / N
    
    def Train(self, data, numOfCluster, errorTolerant):
        epoch = 0
        while(True):
            self.EStep(data, numOfCluster)
            self.MStep(data, numOfCluster)
            if np.linalg.norm(self.mmiuK - self.miuK) < errorTolerant['miu'] and np.linalg.norm(self.mpiK - self.piK)\
                                 < errorTolerant['pi'] and np.linalg.norm(self.msigmaK - self.sigmaK) < errorTolerant['sigma']:
                break
            self.miuK = self.mmiuK
            self.sigmaK = self.msigmaK
            self.piK = self.mpiK
            epoch += 1

            if epoch % 15 == 0:
                draw(data, self.getClassify(), "15 Iterations of E and M Steps")
        print ("Total epoches: ", epoch)
    
    def getClassify(self):
        classify = np.argmax(self.gamaKN, axis=0)
        return classify

def initialize(data, numOfCluster):
    """# random pick center points
    centerPt = np.random.randint(0, len(data), numOfCluster)
    classList = []
    # calculate distance and determine class
    for i in range(len(data)):
        dist = []
        for j in range(numOfCluster):
            distij = np.linalg.norm(centerPt[j] - data[i])
            dist.append(distij)
        classList.append(dist.index(min(dist)))
    # calculate pi and miu
    piKN = np.zeros(numOfCluster)
    miuKN = np.zeros((numOfCluster, len(data[0])))
    for i in range(len(classList)):
        piKN[classList[i]] += 1
        miuKN[classList[i]] += data[i]
    miuK = miuKN / piKN[:,None]
    piK = piKN / len(data)
    # calculate varince
    sigmaK2N = np.zeros((numOfCluster, len(data[0]), len(data[0])))
    for i in range(len(classList)):
        sigmaK2N[classList[i]] += np.diag(pow(data[i] - miuK[classList[i]], 2))
    sigmaK = np.sqrt(sigmaK2N / len(data))
    return {'piK': piK, 'miuK': miuK, 'sigmaK': sigmaK}"""
    # initialize pi
    cnt = 0
    piK = np.zeros(numOfCluster)
    for i in range(numOfCluster - 1):
        piK[i] = np.random.uniform(0, 1-cnt)
        cnt += piK[i]
    piK[-1] = 1 - cnt
    # initialize miu
    miuK = np.zeros((numOfCluster, data.shape[1]))
    miuidx = np.random.randint(0, data.shape[0], size=numOfCluster)
    for i in range(len(miuidx)):
        miuK[i] = data[miuidx[i]]
    # initialize sigma
    sigmaK = np.zeros((numOfCluster, data.shape[1], data.shape[1]))
    for i in range(numOfCluster):
        norm = np.linalg.norm(data - miuK[i], axis=0)
        sigmaK[i] = np.diag(norm / np.sqrt(data.shape[0]))
    
    return {'piK': piK, 'miuK': miuK, 'sigmaK': sigmaK}

if __name__ == '__main__':
    #rtn = generateData(3, [0.3, 0.3, 0.4], 9, [[0, 0],[3, 3], [6, 6]], [[[0.5, 1], [1, 1]],[[1, -1], [-1, 0.5]], [[0.5, 1], [1, 1]]])
    #if rtn == -1:
    #    print ("genertate Data error!")
    data = loadData('.')
    param = initialize(data, 3)
    GMM = GMMModel(param['piK'], param['miuK'], param['sigmaK'])
    errorTolerant = {'miu': 0.1, 'pi': 0.1, 'sigma':0.1}
    GMM.Train(data, 3, errorTolerant)
    draw(data, GMM.getClassify(), "After Clustering with GMM Model")
    #draw(data, np.zeros(len(data)), "Raw Data Generated with Ancestor Sampling")
