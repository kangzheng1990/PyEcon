
from scipy import *
from scipy import sparse, linalg


# HP Filtering -----------------------------------------------------------------

def HPF(D1timeseries,SmoothLevel=1600):

    # make sure the inputs are the right shape
    y=D1timeseries
    y=y.reshape(y.shape[0],1)
    w=SmoothLevel
    m,n  = y.shape
    if m < n:
        y = y.T
        m = n

    a    = array([w, -4*w, ((6*w+1)/2.)])
    d    = tile(a, (m,1))

    d[0,1]   = -2.*w
    d[m-2,1] = -2.*w
    d[0,2]   = (1+w)/2.
    d[m-1,2] = (1+w)/2.
    d[1,2]   = (5*w+1)/2.
    d[m-2,2] = (5*w+1)/2.

    B = sparse.spdiags(d.T, [-2,-1,0], m, m)
    B = B+B.T

    # report the filtered series, s
    s = dot(linalg.inv(B.todense()),y)
    return s.reshape(y.shape[0],)



def HPF_mlt(LongRawDD,SmoothLevel=1600):
    DD=LongRawDD
    return array([HPF(DD[:,x],SmoothLevel) for x in range(DD.shape[1])])

