import numpy as np
import math
import matplotlib.pyplot as plt

total = 1000
yaxis = [] * total
xaxis = [] * total

initial_pos = 105
initial_spd = 0.01
g = 9.8
dt = 0.001
estimated_init = np.array([[105],[0]])
hk = initial_pos
hk_dot = initial_spd
R = 4
v = math.sqrt(R)*np.random.randn(1, total)

x = np.array([[hk],[hk_dot]])
Fk_1 = np.array([[1, dt],[0, 1]])
Gk_1 = np.array([[-1/2*(dt**2)],[-dt]])
uk_1 = g
H = np.array([[1, 0]])
y = np.dot(H, x) + v
size = (2,2)
Qk_1 = np.zeros(size)
Pk = np.array([[10, 0],[0, 0.01]])
identity = np.identity(2,dtype = float)

for i in range(total):
    #predict
    x = np.dot(Fk_1, x)+np.dot(Gk_1,uk_1)
    P = np.dot(Fk_1,np.dot(Pk,Fk_1.T)) + Qk_1
    #update
    K = np.divide(np.dot(P,H.T), np.add(np.dot(H, np.dot(P, H.T)), R))
    xhat_k = x + np.dot(K,np.subtract(y[0,i], np.dot(H, x)))
    Pk = np.dot(np.subtract(identity,np.dot(K,H)),P)
    t = np.dot(dt, np.array([[i]]))
    #print(xhat_k)
    #print('\n')
    xaxis.append(xhat_k[0,0])
    yaxis.append(xhat_k[1,0])
    #print(P)
    #print('\n')
    #print(Pk)
    #print('\n')
    #print(K)
    #print('\n')
    
#for i in range(total):
 #   print('\n')


#print(xaxis)
plt.plot(yaxis)
plt.show()
    

