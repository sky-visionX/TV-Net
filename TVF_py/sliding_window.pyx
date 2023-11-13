import pyximport
pyximport.install()
import numpy as np
cimport numpy as np
cimport cython
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def match_template(np.ndarray img, np.ndarray Hx, np.ndarray Hy):
    W = Hx
    cdef float mindist = float('inf')
    cdef float dist
    cdef unsigned int x, y
    cdef int row = img.shape[0] -2
    cdef int col = img.shape[1] -2
    #cdef int template_width = template.shape[0]
    #cdef int template_height = template.shape[1]
    #cdef int range_x = img_width-template_width+1
    #cdef int range_y = img_height-template_height+1
    #cdef int** grandx_img
    #cdef int** grandy_img
    
    #grandx_img = <int**>malloc(img_height * sizeof(int*))
    #grandy_img = <int**>malloc(img_height * sizeof(int*))
    """for i in range(img_height):
        grandx_img[i] = <int*>malloc(img_width * sizeof(int))
        grandy_img[i] = <int*>malloc(img_width * sizeof(int))
        memset(grandx_img[i], 0, y * sizeof(int))
        memset(grandy_img[i], 0, y * sizeof(int))"""
    grandx_img = np.zeros((row+2,col+2))
    grandy_img = np.zeros((row+2,col+2))
    for y in range(0,row-1):
        for x in range(0,col-1):
            
            W = [[img[y,x],img[y,x+1],img[y,x+2]],[img[y+1,x],img[y+1,x+1],img[y+1,x+2]],[img[y+2,x],img[y+2,x+1],img[y+2,x+2]]]
            Sx = Hx * W
            Sy = Hy * W
            
            grandx_img[y+1,x+1] = np.sum(Sx)
            grandy_img[y+1,x+1] = np.sum(Sy)


    return grandx_img ,grandy_img

"""
img = np.asarray(img, dtype=DTYPE)
template = np.asarray(template, dtype=DTYPE)
match_template(img, template)
"""