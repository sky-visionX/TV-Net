import numpy as np
import utils
import creat
from numba import jit

def decm(a):
    w,h,_,_ = a.shape
    for i in range(w):
        for j in range(h):
            a[i,j,0,0]= round(a[i,j,0,0],4)
            a[i, j, 0, 1] = round(a[i, j, 0, 1], 4)
            a[i, j, 1, 0] = round(a[i, j, 1, 0], 4)
            a[i, j, 1, 1] = round(a[i, j, 1, 1], 4)

    return a

#@jit()
def calc_ortho_extreme(T,r,epsilon ):

    e1,e2,l1,l2= utils.convert_tensor_ev(T)
    q = l1-l2
    h,w = l1.shape
    re = np.zeros((h,w))
    
    
    #X,Y = np.meshgrid(-r:1:r,-r:1:r)
    #X,Y = np.meshgrid(np.linspace(-r,1,r),np.linspace(-r,1,r))
    X,Y = np.meshgrid(range(-r,r,1),range(-r,r,1))
    t = np.arctan2(Y,X)
    l = np.sqrt(X**2 + Y**2)
    q1 = np.zeros((2*r+h, 2*r+w))
    q1[r:h+r, r:w+r] = q
    D = np.array(np.where(q1>0)).T
    #h, w = q1.shape
    
    for i in range(0,D.shape[0]):
        #y,x = utils.ind2sub(h, D[i])
        y = D[i][0]
        x = D[i][1]
        X2 = l * np.cos(t + np.arctan2(e1[y-r,x-r,1],e1[y-r,x-r,0] ))
        Y2 = l * np.sin(t + np.arctan2(e1[y-r,x-r,1],e1[y-r,x-r,0] ))
        t2 = np.abs(np.arctan2(Y2,X2))
        t2[t2 > np.pi/2] = np.pi - t2[t2 > np.pi/2]
        
        t2[t2 <= epsilon] = 1
        t2[t2!=1] = 0
        t2[l > r]= 0
        
        z = np.array(q1[y-r:y+r,x-r:x+r])*t2
        z = np.max(z > q1[y,x])
        if np.max(z) == 0:
            re[y-r,x-r] = 1
    
    return re

#good
#@jit()
def calc_vote_stick(T, *args):
    #T,sigma,cachedvf
    if not args:
        sigma = 18.25
    else:
        sigma = args[0]
        cachedvf = args[1]


    wsize = np.floor( np.ceil(np.sqrt(-np.log(0.01)*sigma**2)*2) / 2 )*2 + 1
    wsize_half = int((wsize-1)/2)
    Th = T.shape[0]
    Tw = T.shape[1]

    Tn = np.zeros((Th+wsize_half*2,Tw+wsize_half*2,2,2))
    Tn[wsize_half:wsize_half+Th, wsize_half:wsize_half+Tw, :, :] = T[0:,0:,:,:]
    T = Tn.copy()
    #kuochong
    #tv
    e1,e2,l1,l2 = utils.convert_tensor_ev(T)
    """
    I = np.where( l1-l2 > 0 )

    d = waitbar(0,'Please wait, stick voting...');

    a = 0

    u,v = utils.ind2sub(l1.shape,I.T)
    p = u.shape[1]
    D = np.zeros((2,p))
    D[1,:] = u
    D[2,:] = v
    op = np.ceil(p*0.01)
    """
    D = np.array(np.where((l1-l2)>0)).T
    #print("D",D)
    a = 0
    for s in D:
        #print("s",s)
        a=a+1
        #if np.mod(a,op) == 0:

        v = e1[s[0],s[1],:]
        #v = v.flatten('F')
        if len(args) < 6:
            Fk = creat.create_stick_tensorfield([-v[1],v[0]],sigma)#Fk good
            """if a == 1:
                print("v",[-v[1],v[0]])
                print("FK1", Fk[:, :, 0, 0])
                print("FK2", Fk[:, :, 0, 1])
                print("FK3", Fk[:, :, 1, 0])
                print("FK4", Fk[:, :, 1, 1])"""

        else:
            angle = np.round(180/np.pi * np.arctan(v[1]/v[0]))
            if angle < 1:
                angle = angle + 180
            
            #shiftdim删除前面的长度为 1 的维度
            Fk = shiftdim(cachedvf[angle,:,:,:,:])
            #Fk = np.squeeze(cachedvf[angle,:,:,:,:])
            #print("shifouzhix______________________________")

        #FK good
        Fk = (l1[s[0],s[1]]-l2[s[0],s[1]])*Fk

        """if a ==1:
            print("fffffffffff",(l1[s[0],s[1]]-l2[s[0],s[1]]))
            print("FK1", Fk[:, :, 0, 0])
            print("FK2", Fk[:, :, 0, 1])
            print("FK3", Fk[:, :, 1, 0])
            print("FK4", Fk[:, :, 1, 1])"""
        beginy = s[0] - wsize_half
        endy = s[0] + wsize_half
        beginx = s[1] - wsize_half
        endx = s[1] + wsize_half
        T[beginy:endy+1,beginx:endx+1,:,:]= T[beginy:endy+1,beginx:endx+1,:,:] + Fk

    T = T[(wsize_half):(wsize_half+Th), (wsize_half):(wsize_half+Tw), :, :]

    """print("T1", T[:, :, 0, 0])
    print("T2", T[:, :, 0, 1])
    print("T3", T[:, :, 1, 0])
    print("T4", T[:, :, 1, 1])"""

    return T




#@jit()
def calc_vote_ball(T,im,sigma):
    Fk = creat.create_ball_tensorfield(sigma)
    wsize = np.floor( np.ceil(np.sqrt(-np.log(0.01)*sigma**2)*2) / 2 )*2 + 1
    wsize_half = int((wsize-1)/2)
    #print("wa",wsize_half)
    Th = T.shape[0]
    Tw = T.shape[1]

    Tn = np.zeros((Th+wsize_half*2,Tw+wsize_half*2,2,2))
    Tn[wsize_half:wsize_half+Th, wsize_half:wsize_half+Tw, :, :] = T[0:,0:,:,:]
    T = Tn.copy()
    e1,e2,l1,l2 = utils.convert_tensor_ev(T)

    """I = np.array(np.where(l1>0))

    #d=waitbar

    a=0

    u,v = utils.ind2sub(l1.shape,I.T)
    """
    
    """p = u.shape[0]
    D = np.zeros((2,p))
    print(D.shape, u.shape , v.shape)
    D[0,:] = u
    D[1,:] = v
    op = np.ceil(p*0.01)
    """
    #print(im.shape,"ims")
    D = np.array(np.where(l1>0,)).T
    #print("l1",l1.shape)(365, 621)
    #print("D",D)#[  0   0]...[364 620]]
    a = 0
    #s=np.zeros((2))
    #D= D.T
    #j=0
    for s in D:
    #for i in range(len(D[0])):
        """s[0]=D[0,(i%6)*5+j]
        s[1]=D[1,(i%6)*5+j]
        if i%6 == 0 and i!=0:
            j=j+1"""
        #print("s",s[0],s[1])
        a=a+1
        #print("s",s)
        #if np.mod(a,op) == 0:

        Zk = im[int(s[0]-wsize_half), int(s[1]-wsize_half)] * Fk #Zk good
        """if a == 1:
            print("T1", Zk [:, :, 0, 0])
            print("T2", Zk [:, :, 0, 1])
            print("T3", Zk [:, :, 1, 0])
            print("T4", Zk [:, :, 1, 1])"""
        
        beginy = s[0] - wsize_half
        endy = s[0] + wsize_half
        beginx = s[1]-wsize_half
        endx = s[1]+wsize_half
        #print("bex",beginx,"edx",endx)
        #print("bey", beginy, "edy", endy)
        #print("T.shape",T.shape)
        """T = decm(T)
        Zk = decm(Zk)"""
        T[int(beginy):int(endy+1),int(beginx):int(endx+1),:,:]= T[int(beginy):int(endy+1),int(beginx):int(endx+1),:,:] + Zk
        #T good
        """if a ==3:
            print("T1", np.round(T[:, :, 0, 0],decimals=4))
            print("T2", np.round(T[:, :, 0, 1],decimals=4))
            print("T3", np.round(T[:, :, 1, 0],decimals=4))
            print("Tt4", np.round(T[:, :, 1, 1],decimals=4))"""
    #Trim the T of the margins we used.

    """print("T1", T[:, :, 0, 0])
    print("T2", T[:, :, 0, 1])
    print("T3", T[:, :, 1, 0])
    print("T4", T[:, :, 1, 1])"""

    T = T[(wsize_half):(wsize_half+Th), (wsize_half):(wsize_half+Tw), :, :]
    """print("T1", T[:, :, 0, 0])
    print("T2", T[:, :, 0, 1])
    print("T3", T[:, :, 1, 0])
    print("T4", T[:, :, 1, 1])"""
    #np.round bushi sishewuru
    return T
if __name__ == "__main__":
    image = np.ones((6, 5))
    aa = np.ones((6,5,2,2))
    T= calc_vote_ball(aa,image,2)

#good
#@jit()
def calc_sparse_field( image ):
    h,w = image.shape
    #h = 5
    #w = 5
    T = np.zeros((h,w,2,2))
    rows,cols = np.where(image>0)#matlab find
    n = rows.shape[0]
    for i in range(0,n):
        T[rows[i],cols[i],:,:] = [[1,0],[0,1]]
    #print("T1",T[: ,: , 0, 0])
    #print("T2", T[:, :, 0, 1])
    #print("T3", T[:, :, 1, 0])
    #print("T4", T[:, :, 1, 1])
    return T



#good
#@jit()
def calc_refined_field( tf, im, sigma ):
    
    ball_vf = calc_vote_ball(tf,im,sigma)
    rows,cols = np.where(im==0)
    s= rows.shape[0]
    for i in range(0,s):
        ball_vf[rows[i],cols[i],0,0] = 0
        ball_vf[rows[i],cols[i],0,1] = 0
        ball_vf[rows[i],cols[i],1,0] = 0
        ball_vf[rows[i],cols[i],1,1] = 0

    T = tf + ball_vf

    return T





