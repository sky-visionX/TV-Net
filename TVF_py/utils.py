import numpy as np
from numba import jit

#good
#@jit()
def convert_tensor_ev(T, *args):
    if not args:
        K11 = T[:,:,0,0]
        K12 = T[:,:,0,1]
        K21 = T[:,:,1,0]
        K22 = T[:,:,1,1]
        #print(K11)
        n , p = K11.shape
        
        o1 = np.zeros((n,p,2))
        o2 = np.zeros((n,p,2))
        o3 = np.zeros((n,p))
        o4 = np.zeros((n,p))

        tt = (K11 + K22)/2

        a = K11 - tt
        b = K12
        ab2 = np.sqrt(a**2 + b**2)
        o3 = ab2 + tt
        o4 = -ab2 + tt

        theta = np.arctan2(ab2 -a , b)

        o1[:,:,0] = np.cos(theta)
        o1[:,:,1] = np.sin(theta)
        o2[:,:,0] = -np.sin(theta)
        o2[:,:,1] = np.cos(theta)
        return o1 , o2, o3 , o4
    
    else:
        i1 = T
        i2 = args[0]
        i3 = args[1]
        i4 = args[2]
        o1 = np.zeros((i3.shape[0],i3.shape[1],2,2))
        o1[:,:,0,0] = i3 * (i1[:,:,0]**2) + i4 * (i2[:,:,0]**2)
        o1[:,:,0,1] = i3 * i1[:,:,0] * i1[:,:,1] + i4 * i2[:,:,0] * i2[:,:,1]
        o1[:,:,1,0] = o1[:,:,0,1]
        o1[:,:,1,1] = i3 * (i1[:,:,1]**2) + i4 * (i2[:,:,1]**2)

        return o1

if __name__ == "__main__":
    a = np.ones((5,5,2,2))
    t = convert_tensor_ev(a)
    #t = np.array(t)
    c = 0
    for i in t:

        print("t1",c,"_______",i[:,:,0])
        print("t2", c, "_______", i[:,:,1])
        c+=1

@jit()
def ind2sub(array_shape, ind):

    ind[ind < 0] = -1

    ind[ind >= array_shape[0]*array_shape[1]] = -1

    rows = (ind.astype('int') / array_shape[1])

    cols = ind % array_shape[1]

    return rows, cols

