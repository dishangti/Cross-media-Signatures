# A much faster version using multiprocessing

import numpy as np
from scipy import integrate
import cv2
import os
from math import sqrt
import json
import multiprocessing as mp
import time

comb = np.array([1,3,3,1,0])
dires = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
dires = [np.array(x) for x in dires]

def get_bezier_parameters(X, Y):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb[k]
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(3, t, k) for k in range(3 + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))
    
    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return final

def dfs_conn(p, mp):
    mp[p[0]][p[1]] = 255
    
    for dire in dires:
        new_p = p + dire
        if new_p[0] >= 0 and new_p[0] < mp.shape[0] and new_p[1] >= 0 and new_p[1] < mp.shape[1] and mp[new_p[0]][new_p[1]] == 0:
            dfs_conn(new_p, mp)


def conn_comp(mp, sig):
    """
    Calculate the number of connected components.
    """
    mp = mp.copy()
    num = 0
    for i in range(sig.shape[0]):
        if mp[sig[i][0]][sig[i][1]] == 0:
                dfs_conn(sig[i], mp)
                num += 1
    return num

def dr(t, rx, ry):
    '''
    Compute r'(t)
    '''
    tt = t * t
    rt = np.array([1, -1, 2*t, -4*t, 2*t, -tt, 3*tt, -3*tt, tt])
#     rx = np.array([x2, x1, x1, x2, x3, x1, x2, x3, x4])
#     ry = np.array([y2, y1, y1, y2, y3, y1, y2, y3, y4])
    r1 = np.dot(rt, rx)
    r2 = np.dot(rt, ry)
    return r1, r2

def atan_r(t, rx, ry):
    """
    Compute the degree of tangent of B3 by point
    i.e. arctan(dy/dx(t))
    """
    return np.arctan2(*dr(t, rx, ry))

def ds(t, rx, ry):
    """
    Compute element of arc lengthof B3 by t
    i.e. ds = |r'(t)| dt
    """
    r1, r2 = dr(t, rx, ry)
    return sqrt(r1 * r1 + r2 * r2)

def kr(t, c1, c2, c3, rx, ry):
    """
    Compute reletive curvature element of B3 by arc-length parameterization s
    i.e. kr(s) ds = kr(s(t))|r'(t)| dt
    
    We introduce c1, c2, c3, rx, ry in order to fasten the computation
    """
    r = ds(t, rx, ry)
    tt = t * t
    return (c1 + tt*c2 + t*c3) / (r * r)
#     x12 = x1-x2
#     y12 = y1-y2
#     return 18*(x1*(y2-y3) + x2*(y3-y1) + x3*y12\
#                + tt*(x1*(y2+y4) - (x2+x4)*y1 + x3*y4 - x4*y3\
#                + 2*(-x1*y3 + x3*y1 - x2*y4 + x4*y2) + 3*(x2*y3 - x3*y2))\
#                + t*(2*(-x1*y2 + x2*y1) + 3*(x12*y3 - x3*y12) - x12*y4 + x4*y12))\
#                / (r * r)

def k(t, c1, c2, c3, rx, ry):
    """
    Compute curvature element of B3 by arc-length parameterization s
    i.e. k(s) ds = k(s(t))|r'(t)| dt = |kr(s(t))r'(t)| dt
    """

    return abs(kr(t, c1, c2, c3, rx, ry))

def k_para(p):
    p0, p1, p2, p3 = p
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    x4, y4 = p3
    
    x12 = x1-x2
    y12 = y1-y2
    c1 = x1*(y2-y3) + x2*(y3-y1) + x3*y12
    c2 = x1*(y2+y4) - (x2+x4)*y1 + x3*y4 - x4*y3\
          + 2*(-x1*y3 + x3*y1 - x2*y4 + x4*y2) + 3*(x2*y3 - x3*y2)
    c3 = 2*(-x1*y2 + x2*y1) + 3*(x12*y3 - x3*y12) - x12*y4 + x4*y12
    return 18*c1, 18*c2, 18*c3

def dr_para(p):
    p0, p1, p2, p3 = p
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    x4, y4 = p3
    
    rx = np.array([x2, x1, x1, x2, x3, x1, x2, x3, x4])
    ry = np.array([y2, y1, y1, y2, y3, y1, y2, y3, y4])
    return 3 * rx, 3 * ry

def mean_curv(p):
    c1, c2, c3 = k_para(p)
    rx, ry = dr_para(p)
    arc_len, _ = integrate.quad(ds, 0, 1, args=(rx, ry)) # Length of arc
    total_curv, _ = integrate.quad(k, 0, 1, args=(c1, c2, c3, rx, ry))    # Sum of curvature
    
    return total_curv, arc_len

def winding_deg(p):
    c1, c2, c3 = k_para(p)
    rx, ry = dr_para(p)
    deg, _ = integrate.quad(k, 0, 1, args=(c1, c2, c3, rx, ry))
    
    return deg

def dfs_nearpoint(p, mp, near_set, bs):
    mp[p[0]][p[1]] = 255
    near_set.append(p)
    next_pos = []
    
    for dire in dires:
        if len(near_set) == bs:
            next_pos.append(p)
            break
        new_p = p + dire
        if new_p[0] >= 0 and new_p[0] < mp.shape[0] and new_p[1] >= 0 and new_p[1] < mp.shape[0] and mp[new_p[0]][new_p[1]] == 0:
            next_pos.extend(dfs_nearpoint(new_p, mp, near_set, bs))
    return next_pos

def find_nearpoint(mp, bs):
    mp = mp.copy()
    near_sets = []
    next_pos = []
    for i in range(mp.shape[0]):
        for j in range(mp.shape[1]):
            if mp[i][j] == 0:
                next_pos.append(np.array((i, j)))
                while len(next_pos) > 0:
                    near_set = []
                    r, c = next_pos[-1]; next_pos.pop()
                    
                    next_pos.extend(dfs_nearpoint(np.array((r, c)), mp, near_set, bs))
                    if 5 <= len(near_set) <= bs:
                        near_sets.append(near_set)
            
    return near_sets

def get_feature(person, im_num):
    if os.path.exists(f'./skel/{person}/{im_num}' + '.png'):        # Judge image type
        file = f'./skel/{person}/{im_num}'+'.png'
    elif os.path.exists(f'./skel/{person}/{im_num}' + '.jpg'):
        file = f'./skel/{person}/{im_num}'+'.jpg'
    im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    num_conn = 0

    sig = np.dstack(np.nonzero(255 - im)).squeeze()
    num_conn = conn_comp(im, sig)

    bs = 6

    winding_tmp = []
    curv_tmp = []
                
    near_sets = find_nearpoint(im, bs)
    for near_set in near_sets:
        near_set  = np.array(near_set)
        near_xpts = near_set[:,0]
        near_ypts = near_set[:,1]
        data = get_bezier_parameters(near_ypts, near_xpts)
        
        # Calculate features
        angle  = winding_deg(data) * 180 / np.pi
        winding_tmp.append(angle)
        curv_tmp.append(mean_curv(data))
    
    return num_conn, winding_tmp, curv_tmp

def get_features(typ, first, last):
    winding = []
    curv = []
    num_conn = []
    for i in range(first, last + 1):  # People
        winding.append([])
        curv.append([])
        num_conn.append([])
        for j in range(10 * typ + 1, 10 * typ + 11):
            if (not os.path.exists(f'./skel/{i}/{j}'+'.png')) and (not os.path.exists(f'./skel/{i}/{j}'+'.jpg')):
                continue
            num_conn_tmp, winding_tmp, curv_tmp = get_feature(i, j)
            winding[i - first].append(winding_tmp)
            curv[i - first].append(curv_tmp)
            num_conn[i - first].append(num_conn_tmp)

    return num_conn, winding, curv

def get_features_multi(pool, typ, last):
    winding = [[]]
    curv = [[]]
    num_conn = [[]]

    
    all_num = last
    sub_num = all_num // num_cores
    res_num = all_num % num_cores
    sub_task = []
    if last < num_cores:
        for i in range(last):
            sub_task.append((i + 1, i + 1))
    else:
        for i in range(num_cores):
            if i < res_num:
                sub_task.append((i * sub_num + 1 + i, (i + 1) * sub_num  + i + 1))
            else:
                sub_task.append((i * sub_num + 1 + res_num, (i + 1) * sub_num  + res_num))
    results = [pool.apply_async(get_features, args=(typ, beg, end)) for beg, end in sub_task]
    results = [p.get() for p in results]
    for result in results:
        num_conn.extend(result[0])
        winding.extend(result[1])
        curv.extend(result[2])

    return num_conn, winding, curv

if __name__ == '__main__':
    person = 11
    num_cores = int(mp.cpu_count())   # Use all the CPU cores as default
    pool = mp.Pool(num_cores)
    num_conn = []
    winding = []
    curv = []
    start = time.time()
    for i in range(2):
        num_conn_tmp, winding_tmp, curv_tmp = get_features_multi(pool, i, person)
        num_conn.append(num_conn_tmp)
        winding.append(winding_tmp)
        curv.append(curv_tmp)
        print(f'{(i + 1)/2*100}% Finished')
    end = time.time()
    pool.close()
    print(end-start)
    #Save data
    with open('control_num_conn.dat', 'w') as f:
        json.dump(num_conn, f)
    with open('control_winding.dat', 'w') as f:
        json.dump(winding, f)
    with open('control_curv.dat', 'w') as f:
        json.dump(curv, f)