import numpy as np
import math
import sys

def e(len, i):
    v = np.zeros(len)
    v[i] = 1
    return v

def primal_simplex(c, a, b, x):
    res = dict()
    #init
    m = len(b) #number of basic var (constraints)
    n = len(c) #number of non-basic var

    all_var = np.arange(0, n + m)
    n_idx = np.arange(0, n)
    b_idx = np.arange(n, n+m)

    B = np.identity(m)
    N = a
    a = np.concatenate((N ,B), axis=1)

    c_b = np.zeros(m)
    c_n = c

    x_b = b
    z_n = -c_n
    x = np.concatenate((z_n, x_b))
    
    #loop
    while True:
        if np.all(z_n >= 0):
            status = 'optimal'
            print('optimal')
            break
        
        j = np.where(z_n < 0)[0][0]
        delta_x_b = np.dot(np.dot(np.linalg.inv(B), N), e(n, j))

        ratio = delta_x_b / x_b
        ratio[ratio == np.Infinity] = 0

        t = np.max(ratio) ** -1
        i = np.argmax(ratio)

        if t <= 0:
            status = 'unbouned'
            print('unbounded')
            break

        delta_z_n = - np.dot(np.dot(np.linalg.inv(B), N).T, e(m, i))

        s = z_n[j] / delta_z_n[j]
        

        x_b = x_b - t * delta_x_b #update xb
        x_b[j] = t
        z_n = z_n - s * delta_z_n #update zn
        z_n[i] = s

        #update optimal variable value
        x[list(b_idx)[j]] = t
        x[list(n_idx)[i]] = s

        xi = np.dot(np.dot(c_b.T, np.linalg.inv(B)), b)  - np.dot((np.dot(np.dot(np.linalg.inv(B), N).T, c_b) - c_n).T, z_n)

        print(b_idx, i, j)
        temp_b = b_idx[j]
        temp_n = n_idx[i]

        b_idx = sorted(b_idx.difference(set(n_idx[i])).union(set(b_idx[j])))
        n_idx = sorted(all_var.difference(b_idx))

        print(b_idx)
        B = a[:, b_idx]
        N = a[:, n_idx]

    if status == 'optimal':
        res = {
        'status': status,
        'optimal value': xi,
        'x': x
        }

    return res



def dual_simplex(c, a, b, x):
    #init

    m = len(b) #number of basic var (constraints)
    n = len(c) #number of non-basic var

    all_var = set(range(1, n+m+1))
    n_idx = set(range(1, n+1))
    b_idx = set(range(n+1, n+m+1))

    B = np.identity(m)
    N = a
    a = np.concatenate((n ,b), axis=1)

    c_b = np.zeros(m)
    c_n = c

    x_b = b
    z_n = -c_n
    x = np.concatenate((z_n, x_b), axis=1)

    while True:
        if z_n.all() >= 0:
            status = 'optimal'
            break
    
        i = np.where(x_b < 0)[0]
        delta_z_n = - np.dot(np.dot(np.linalg.inv(B), N).T, e(n, j))

        ratio = delta_z_n / z_n
        ratio[ratio == np.Infinity] = 0

        t = np.max(ratio) ** -1
        j = np.argmax(ratio)

        if t <= 0:
            status = 'unbouned'
            break

        delta_x_b = np.dot(np.dot(np.linalg.inv(B), N), e(m, i))

        s = x_b[j] / delta_x_b[j]
        s[s == np.Infinity] = 0

        x_b = x_b - t * delta_x_b #update xb
        x_b[j] = t
        z_n = z_n - s * delta_z_n #update zn
        z_n[i] = s

        #update optimal value
        xi = - np.dot(np.dot(c_b.T, np.linalg.inv(B)), b) - np.dot(np.dot(np.linalg.inv(B), b).T, z_n)

        #update optimal variable value
        x[B[j]] = t
        x[N[i]] = s

        b_idx = sorted(b_idx.difference(set([i])).union(set([j])))
        n_idx = sorted(all_var.difference(b_idx))

        B = a[:, b_idx]
        N = a[:, n_idx]

        res = {
            'status': status,
            'optimal value': xi,
            'x': x
        } if status == 'optimal' else {
            'status': status,
            'optimal value': np.array([]),
            'x': np.array([])
        }
    return res

def simplex(c, a, b):
    return 0

if __name__=="__main__":
    c = np.array([4,-3])
    a = np.array([[1,-1],[2,-1],[0,1]])
    b = np.array([1, 3, 5])
    x = np.array([])
    print(primal_simplex(c, a, b, x))