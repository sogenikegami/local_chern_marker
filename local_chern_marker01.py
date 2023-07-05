import matplotlib.pyplot as plt
import math
import numpy as np

def Projector(H,fermi_energy):     # define and calculate the Projection P
    eigval,eigvec = np.linalg.eig(H)
    N = len(eigval)
    P = np.zeros((N,N))*0j
    for i in range(N):
        if eigval[i].real < fermi_energy:
            eigenvector = np.zeros((N,1))*0j
            for j in range(N):
                eigenvector[j][0] += eigvec[j][i] 
            P += eigenvector * eigenvector.conj().T
    return P

def xmatrix(H,vertexdata):
    N = len(H[0])
    X = np.zeros((N,N))*0j
    inner_deg_free = int(N/len(vertexdata))
    for j in range(inner_deg_free):
        for i in range(len(vertexdata)):
            pos = vertexdata[i]["pos"]
            X[i+j*len(vertexdata)][i+j*len(vertexdata)] += pos[0]
    return X

def ymatrix(H,vertexdata):
    N = len(H[0])
    Y = np.zeros((N,N))*0j
    inner_deg_free = int(N/len(vertexdata))
    for j in range(inner_deg_free):
        for i in range(N):
            pos = vertexdata[i]["pos"]
            Y[i+j*len(vertexdata)][i+j*len(vertexdata)] += pos[1]
    return Y

def local_chern_marker(H,vertexdata,fermi_energy):
    N = len(H[0])
    inner_deg_free = int(N/len(vertexdata))
    P = Projector(H,fermi_energy)
    X = xmatrix(H,vertexdata)
    Y = ymatrix(H,vertexdata)
    prod = P @ X @ P @ Y @ P
    local_c = []
    for i in range(len(vertexdata)):
        temp = 0
        for j in range(inner_deg_free):
            temp += 4 * math.pi * prod[i+j*len(vertexdata)].imag
        local_c.append(temp)
    return local_c

def plot(H,vertexdata,fermi_energy,filepath,imgname):
    local_c = local_chern_marker(H,vertexdata,fermi_energy)
    N = len(vertexdata)
    X = []
    Y = []
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(N):
        X.append(vertexdata[i]["pos"][0])
        Y.append(vertexdata[i]["pos"][1])
        for j in range(len(vertexdata[i]["neighbor"])):
            neinum = vertexdata[i]["neighbor"][j]
            neipos = vertexdata[j]["pos"]
            nowpos = vertexdata[i]["pos"]
            plt.plot([nowpos[0],neipos[0]],[nowpos[1],neipos[1]],color="black",linewidth = 1)
    ax.scatter(X,Y,s = 1000 * np.abs(local_c),c = local_c,cmap = 'bwr')
    fig.show()
    plt.savefig(filepath + imgname)
    return 0



def main():
    return 0

if __name__ == "__main__":
    main()