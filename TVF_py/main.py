import os
import cv2
import numpy as np
import sliding_window
import utils
import features
import calc
import time
from scipy import ndimage

def rdnumpy(txtname):
    f = open(txtname)
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    A_row = 0  # 表示矩阵的行，从0行开始
    count=0
    row = 0
    for l in lines:
        row = row +1
    A = np.zeros((row,3),dtype=float)
    for line in lines:  # 把lines中的数据逐行读取出来
        count=count+1
        if count ==1:
            list = line.strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
            A[A_row:] = list[0:1]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
            A_row += 1
        else:
            list = line.strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
            #print((list))
            A[A_row:] = list[0:3]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
            A_row += 1
    return A




#dir_path = "./test"
#save_path = "./tv_image"
dir_path = "/media/rmz/749d3547-4bb1-4302-a40c-63292a590f0a/data/2D公开数据集整理/crack500/val/images"
save_path = "/media/rmz/749d3547-4bb1-4302-a40c-63292a590f0a/data/2D公开数据集整理/crack500/val/tv_images"

if __name__ == "__main__":
    count = 0
    for filename in os.listdir(dir_path):
        bt = time.time()
        file_path =  os.path.join(dir_path,filename)
        if not os.path.isfile(file_path):
            print("file not exist!")
        else:

            gs_name = filename.split(".")[1]
            if gs_name !="jpg":
                continue

            count += 1
            split_filename = filename.split(".")[0]
            image = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)

            row, col = image.shape
            if row*col>256*400 and row*col<=800*600:#(512,512),(800,600)
                image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
            elif row*col>800*600 and row*col<=2000*1500:
                image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)

            elif row*col>2000*1500: #(2590,1904)
                image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
            #image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
            row,col = image.shape



            ###########################Tmask###########
            #T = image[image>=0 and image <=10]
            _, T = cv2.threshold(image,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            #T = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
            #T = cv2.Canny(image, 200, 255)



            #mask = ndimage.binary_fill_holes(T)

            #image[mask<=0] = 0
            image_mask = 255-T
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # ksize=5,5
            image_mask = cv2.erode(image_mask, kernel, iterations=1)
            image_mask = cv2.medianBlur(image_mask, 3)


            """cv2.imshow("w",image)
            cv2.waitKey(0)"""
            #############################################3

            #count_row = 1
            #image = np.ones((row,col))
            big_image = np.zeros((row+2,col+2))
            #print(big_image.shape)
            #原灰度图一圈扩展1个像素
            big_image[1:row+1,1:col+1] = image
            #sobel算子x方向模板
            Hx = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
            #sobel算子y方向模板
            Hy = Hx.T
            gradx_image = np.zeros((row+2,col+2))
            grady_image = np.zeros((row + 2, col + 2))
            W = np.zeros((3,3))
            #gradx_image,grady_image = sliding_window.match_template(big_image, Hx, Hy)
            for i in range(row):
                for j in range(col):
                    W = [[big_image[i,j], big_image[i,j+1], big_image[i,j+2]],
                    [big_image[i+1, j], big_image[i+1, j + 1], big_image[i+1, j + 2]],
                    [big_image[i+2, j], big_image[i+2, j + 1], big_image[i+2, j + 2]]]
                    Sx = Hx * W
                    Sy = Hy * W
                    gradx_image[i+1,j+1] = np.sum(Sx)
                    grady_image[i+1,j+1] = np.sum(Sy)


            gradx = np.zeros((row,col))
            grady = np.zeros((row,col))
            gradx = gradx_image[1:row+1,1:col+1]
            grady = grady_image[1:row+1,1:col+1]
            count_rows = np.sum(image_mask>0)


            grad = gradx + grady


            """count_rows = np.sum(image > 0)
            gradx = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
            grady = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)

            abs_grad_x = cv2.convertScaleAbs(gradx)
            abs_grad_y = cv2.convertScaleAbs(grady)
            grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            img_sobel_xdst = abs_grad_x.flatten()
            img_sobel_ydst = abs_grad_y.flatten()
            img_sobel_angle_arr = np.arctan2(img_sobel_ydst, img_sobel_xdst) * 180 / np.pi
            #edge_image = img_sobel_angle_arr[img_sobel_angle_arr>0]
            #print(img_sobel_angle_arr[img_sobel_angle_arr>0])
            #img_sobel_angle = img_sobel_angle_arr.reshape(H, W, C)
            countt = 0
            h_max=0
            w_max=0
            for i in range(0,row):
                for j in range(0,col):
                    if image[i][j]>0:
                        if i> h_max:
                            h_max = i
                        if j>w_max:
                            w_max = j
            h = h_max
            w = w_max
            T = np.zeros((int(h) + 1, int(w) + 1, 2, 2))
            for i in range(0,row):
                for j in range(0,col):
                    if image[i][j]>0:
                        x = np.cos(img_sobel_angle_arr[countt] * np.pi / 180 + 90 * np.pi / 180)
                        y = np.sin(img_sobel_angle_arr[countt]  * np.pi / 180 + 90 * np.pi / 180)
                        T[i, j, 0, 0] = x ** 2
                        T[i, j, 0, 1] = x * y
                        T[i, j, 1, 0] = x * y
                        T[i, j, 1, 1] = y ** 2

                    countt= countt +1
"""
            #计算梯度方向角
            edge_image = np.zeros((count_rows+1,3))
            ei = 1
            #print("row,",row,col)
            edge_image[0][0] = count_rows-1
            for i in range(0,row):
                for j in range(0,col):
                    if image_mask[i][j]>0:
                        fiangle = np.arctan2(grady[i][j],gradx[i][j])
                        #print("fiagle",fiangle)
                        angle = fiangle/np.pi *180 +180
                        edge_image[ei][0] = row+1-i
                        edge_image[ei][1] = j
                        edge_image[ei][2] = angle
                        ei = ei+1
            #print("edge_image", edge_image)
            #这里的edge_name是countrows行，3列，第二行开始，第一轮是位置下，第二列是位置y，第三列是angle
            #print(edge_image)

            #edge_image = rdnumpy("./edge_test/208.txt")
            #print("ed,",edge_image)
            ##############################做方向角变换#######################
            h = np.max(edge_image[1:,0])
            w = np.max(edge_image[1:,1])
            T = np.zeros((int(h)+1,int(w)+1,2,2))
            for i in range(1,int(edge_image[0,0])):
                x = np.cos(edge_image[i,2] *np.pi/180 + 90*np.pi/180)
                y = np.sin(edge_image[i,2] *np.pi/180 + 90*np.pi/180)
                #^全部都改**
                T[int(h+1 - edge_image[i,0]), int(edge_image[i,1]),0,0] = x**2
                T[int(h+1 - edge_image[i,0]), int(edge_image[i,1]),0,1] = x*y
                T[int(h+1 - edge_image[i,0]), int(edge_image[i,1]),1,0] = x*y
                T[int(h+1 - edge_image[i,0]), int(edge_image[i,1]),1,1] = y**2

            ##############################张量投票#######################
            """print(T.shape)
            print("T0",T[:,:,0,0])
            print("T1", T[:, :, 0, 1])
            print("T2", T[:, :, 1, 0])
            print("T3", T[:, :, 1, 1])"""
            e1, e2, l1, l2 = utils.convert_tensor_ev(T)

            T = features.find_features(l1,15)
            """print("T1", T[:, :, 0, 0])
            print("T2", T[:, :, 0, 1])
            print("T3", T[:, :, 1, 0])
            print("T4", T[:, :, 1, 1])"""
            e1, e2, l1, l2 = utils.convert_tensor_ev(T)

            z = l1-l2
            #l1[z<0.3] = 0
            #l2[z<0.3] = 0
            l12 = l1 - l2

            b = (l12-np.min(l12))/(np.max(l12)-np.min(l12))*255
            """cv2.imshow("t", b)
            cv2.waitKey(0)"""
            #T = utils.convert_tensor_ev(e1, e2, l1, l2 )
            #re = calc.calc_ortho_extreme(T,5, np.pi/36)

            TV = b



            fid = os.path.join(save_path,split_filename)+".jpg"
            #cv2.imshow("tv",TV)
            #cv2.waitKey(0)
            cv2.imwrite(fid,TV)
            print("目前处理好的图片名称是：",split_filename)
            
            et = time.time()
            alltime = et -bt
            print("用时-----------------",alltime)
            print("已经处理",count,"张图片！")
    
          










            
            







            



