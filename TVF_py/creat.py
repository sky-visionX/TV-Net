import numpy as np

from numba import jit

import cupy as cp




    


#good
#@profile
#@jit()
def create_stick_tensorfield( *args):
    #uv, sigma
    if len(args)<1:
        uv = np.array([1,0])
    if len(args)<2:
        sigma =  18.25#18.25
    else:
        uv = args[0]
        sigma = args[1]
    #print("uv",uv)

    ws = np.floor( np.ceil(np.sqrt(-np.log(0.01)* (sigma**2))*2) / 2 )*2 + 1
    whalf = int((ws-1)/2)


    uv1 = np.append([[-uv[1]]],[[uv[0]]],axis=0)
    #rot = [uv,[-uv[1];uv[0]]]/ np.linalg.norm(uv)
    #print("uv.T",np.array([uv]).T)
    #print("uv1",uv1)
    #print("w",np.append(uv.T,uv1,axis=1))
    #print("e",np.linalg.norm(uv))
    rot = np.append(np.array([uv]).T,uv1,axis=1) / np.linalg.norm(uv)


    #print(("rot",rot))
    btheta = np.arctan2(uv[1],uv[0])

    #np.meshgrid(np.linspace(-r,1,r),np.linspace(-r,1,r))
    X,Y = np.meshgrid(range(-whalf,whalf+1,1),range(whalf,-whalf-1,-1))
    #[X(:), Y(:)] 将 X 和 Y 两个 (35, 35) 的矩阵转换为列向量
    #和matlab一样按列展平了
    #print("x",X)
    #print("Y",Y)
    X = X.flatten('F')
    Y = Y.flatten('F')
    """try:
        Z = rot.T * np.append([X],[Y],axis=1).T

    except Exception as e :
            print(np.append([X],[Y],axis=0).T)
            print(rot.T)
            print(e)"""
    Z = np.dot(rot.T, np.array([X,Y]))
    #print("rot",np.append([X],[Y],axis=0))
    X = np.reshape( Z[0,:],(int(ws),int(ws)),order='F')
    Y = np.reshape( Z[1,:],(int(ws),int(ws)),order='F')
    theta = np.arctan2(Y,X)#theta good
    #print("X", X)
    #print("Y", Y)
    #print("theta",theta)



    #Tb = np.reshape([theta,theta,theta,theta],(int(ws),int(ws),2,2),order='F')
    #zhuanzhi ,
    Ttheta = np.array(theta).T
    Tb = np.array([[Ttheta,Ttheta],[Ttheta,Ttheta]]).T#Tb good
    #print("theta",theta)
    #print("T", Tb[:, :, 0, 0])
    #print((Tb.shape))
    #print("btheta,",btheta)
    T1 = -np.sin(2*Tb+btheta)
    T2 = np.cos(2*Tb+btheta)
    T3 = T1.copy()
    T4 = T2.copy()
    T1[:,:,1,0:2] = 1
    T2[:,:,0:2,0] = 1
    T3[:,:,0:2,1] = 1
    T4[:,:,0,0:2] = 1
    """print("T1")
    print("T1", T1[:, :, 0, 0])
    print("T2", T1[:, :, 0, 1])
    print("T3", T1[:, :, 1, 0])
    print("T4", T1[:, :, 1, 1])"""

    T = T1 * T2 * T3 * T4 #T good

    #print("theta",theta)
    """print("T1", T[:, :, 0, 0])
    print("T2", T[:, :, 0, 1])
    print("T3", T[:, :, 1, 0])
    print("T4", T[:, :, 1, 1])"""

    theta = np.abs(theta)
    theta[theta>np.pi/2] = np.pi - theta[theta> np.pi/2]
    theta = 4*theta

    s = np.zeros((int(ws),int(ws)))
    k = np.zeros((int(ws),int(ws)))
    #Calculate the attenuation field.
    l = np.sqrt(X**2 + Y**2)
    c = (-16* np.log2(0.1)*(sigma-1))/(np.pi**2)
    #print(s.shape, l.shape , theta.shape)

    arrlogiT = np.logical_and(l!=0 , theta!=0)
    arrlogiF = np.logical_or(l==0 , theta==0)
    s[arrlogiT] = (theta[arrlogiT] * l[arrlogiT])/np.sin(theta[arrlogiT])
    s[arrlogiF] = l[arrlogiF]
    k[l!=0] = 2* np.sin(theta[l!=0]) / l[l!=0]
    DF = np.exp(-((s**2 + c*(k**2))/(sigma**2)))
    DF[theta> np.pi/2] = 0

    #print("DF",DF)
    #Generate the final tensor field
    TDF= np.array(DF).T
    TTDF = np.array([[TDF,TDF],[TDF,TDF]]).T


    #T good DF good
    T = T * TTDF#np.reshape([DF,DF,DF,DF],(int(ws),int(ws),2,2),order='F')
    """print("T1", T[:, :, 0, 0])
    print("T2", T[:, :, 0, 1])
    print("T3", T[:, :, 1, 0])
    print("T4", T[:, :, 1, 1])"""
    return T



if __name__ == "__main__":

    a = np.array([-0.7071067811865475, 0.7071067811865476])
    T= create_stick_tensorfield(a,2 )


#@jit()
def create_cached_vf( sigma ):
    ws = np.floor( np.ceil(np.sqrt(-np.log(0.01)*(sigma**2))*2) / 2 )*2 + 1
    out = np.zeros((180,int(ws),int(ws),2,2))
    for i in range(0,180):
        x = [np.cos(i* np.pi/180)]
        y = [np.sin(i* np.pi/180)]
        """try :
            v = np.append(x,y,axis=0)
        except Exception as e :
            print(x)
            print(y)
            print(e)"""
        #v = np.concatenate(([x],[y]),axis=1)
        v = np.append(x,y,axis=0)
        #print("v",v)
        Fk = create_stick_tensorfield(v,sigma)
        out[i,:,:,:,:] = Fk

    return out


#@jit()
#good
def create_ball_tensorfield(*args):
    if not args :
        sigma = 18.25
    else:
        sigma = args[0]

    wsize = np.ceil(np.sqrt(-np.log(0.01)*sigma**2)*2)
    wsize = np.floor(wsize/2)*2+1
    
    T = np.zeros((int(wsize),int(wsize),2,2))
    #for theta = (0:1/32:1-1/32)*2*pi:0 到（1-1/32），步长为1：32
    for theta in np.arange(0, (1-1/32)*np.pi, 1/32*np.pi):    
        #v = [cos(theta);sin(theta)]matlab同一行中的元素间用逗号( , )或者空格分隔，不同行之间用分号( ; )隔开。
        v= np.append([np.cos(theta)],[np.sin(theta)],axis=0).T#V shape 2*2
        B = create_stick_tensorfield(v,sigma)
        T = T + B
    
    T = T/32
    #print("T",T[:,:,0,0])
    return T







