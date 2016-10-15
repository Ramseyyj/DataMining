import numpy as np

D=np.array([[0,411,213,219,296,397],
            [411,0,204,203,120,152],
            [213,204,0,73,136,245],
            [219,203,73,0,90,191],
            [296,120,136,90,0,109],
            [ 397,152,245,191,109,0]])

N = D.shape[0]
T = np.zeros((N,N))

# solution 1
# ss = 1.0/N**2*np.sum(D**2)
# for i in range(N):
#    for j in range(i,N):
#        T[i,j] = T[j,i] = -0.5*(D[i,j]**2 -1.0/N*np.dot(D[i,:],D[i,:]) -1.0/N*np.dot(D[:,j],D[:,j])+ss)


# solution 2
# K = np.dot(D,np.transpose(D))
D2 = D**2
H = np.eye(N) - 1/N
B = -0.5*np.dot(np.dot(H, D2), H)

eigVal, eigVec = np.linalg.eig(B)
X = np.dot(eigVec[:, :2], np.diag(np.sqrt(eigVal[:2])))
print(X.shape)

# X即降维后的矩阵
