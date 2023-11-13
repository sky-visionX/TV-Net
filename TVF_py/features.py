import numpy as np
import utils

import calc
import creat
def find_features(img , sigma):
    cached_vtf = creat.create_cached_vf(sigma)
    im = img / np.max(img)
    #im =img
    sparse_tf = calc.calc_sparse_field(im)

    refined_tf = calc.calc_refined_field(sparse_tf,im,sigma)

    e1,e2,l1,l2 = utils.convert_tensor_ev(refined_tf)
    l2 = np.zeros(l2.shape)

    zerol2_tf = utils.convert_tensor_ev(e1,e2,l1,l2)

    T = calc.calc_vote_stick(zerol2_tf, sigma, cached_vtf)
    #T = calc.calc_vote_stick(zerol2_tf,sigma)
    return T


