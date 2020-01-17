# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 00:58:29 2019
Updated 
@author: Nicole Creange and xll

Last updated: Jan 17 2020
Issues seem to arise if tau vector is > 1.5* freq vector, if freq vector is < 100
"""


def cal_Basis(f,t,k=1e4):
    import numpy as np
    nf = f.size
    nt = t.size
    Ar = np.zeros((nf,nt - 1))
    Ai = np.zeros((nf,nt - 1))
    sv = []
    for ii in range(nt-1):
        sv.append(np.random.uniform(t[ii],t[ii+1],np.int(k)))
    for jj in range(nf):
        dum = f[jj]*np.asarray(sv)*np.pi*2j
        dum1 = sum(1/(1+dum).T)/k
        dum1 = dum1.reshape((1,len(dum1)))
        Ar[jj,:] = dum1.real
        Ai[jj,:] = dum1.imag
    return Ar,Ai


def cholinsert(R, x, X): 
    import numpy as np
    diag_k = np.matmul(x.T, x)
    if R.size == 0:
        R = np.sqrt(diag_k)
    else:
        col_k = np.matmul(x.T,X)
        if isinstance(col_k,np.float64) or isinstance(R,np.float64):
            R_k = col_k/R
        else:
            R_k = np.linalg.lstsq(R.T,col_k.T,rcond=1e-4)[0] 
        R_k = R_k.reshape(R_k.size,1)
        if np.matmul(R_k.T,R_k) > diag_k:
            print('Issue')
            np.linalg.matrix_rank(R)
            R = np.vstack(R,(np.zeros((1,R.shape[1]))))
            return R
        else:
            R_kk = np.sqrt(diag_k - np.matmul(R_k.T,R_k))
        if isinstance(R,np.float64):
            R = np.vstack((np.hstack((np.ones((1,1))*R,R_k)),np.hstack((np.zeros((1,1)),R_kk))))
        else:
            R = np.vstack((np.hstack((R,R_k)),np.hstack((np.zeros((1,R.shape[1])),R_kk))))
    return R
        

def choldown(Lt, k):
    import numpy as np
    from copy import deepcopy
    p = len(Lt)
    Temp = deepcopy(Lt)
    Temp = np.delete(Temp,k-1,1) 
    for ii in range(k,p):
            a = Temp[ii-1,ii-1]
            b = Temp[ii,ii-1]
            r = np.sqrt(np.sum(Lt[:,ii]**2)-np.sum(Temp[0:ii-2,ii-1]**2))
            c = r*a/(a**2 + b**2)
            s = r*b/(a**2 + b**2)
            Hrowi = np.zeros((1,p))
            Hrowi[:,ii-1] = c
            Hrowi[:,ii] = s
            Hrowi1 = np.zeros((1,p))
            Hrowi1[:,ii-1] = -s
            Hrowi1[:,ii] = c
            v = np.zeros((2,p-1))
            v[0,ii-1:p-1] = np.matmul(Hrowi,Temp[:,ii-1:p-1])
            v[1,ii:p-1] = np.matmul(Hrowi1,Temp[:,ii:p-1])
            Temp[ii-1:ii+1,:] = v;
          
    Lkt = Temp[0:p-1,:]
    return Lkt

def NN_LARS(A,b,delta): 
  import numpy as np
  n, p = A.shape 
  
  A = np.sqrt(1+delta)*np.vstack((A,np.eye(p)))
  b = b.reshape((len(b),1))
  b = np.vstack((b,np.zeros((p,1))))
  
  n, p = A.shape
  
  b0 =  np.matmul(np.linalg.pinv(A),b)
  sigma2e = np.sum(np.power(b-np.matmul(A,b0),2))/n
  
  indices = np.arange(p).reshape(p,1)
  nvars = np.min([n,p])
  maxk = np.Inf
  maxvar = 0
  I = np.arange(p)
  
  P = np.array([]).astype(np.int)
  R = np.array([])
  
  lassocond = 0
  earlystopcond = 0
  k = 0
  cnvars = 0
  x = np.zeros((p,1))
  
  c = np.matmul(A.T,b)
  
  err = np.array([])
  X = np.array([])
  df = np.array([])
  Cp = np.array([])
  
  while cnvars < nvars and not earlystopcond and k < maxk:
    k = k + 1
    j = np.argmax(c[I])
    C = np.max(c[I])
    if C < np.finfo(float).eps:
      AS = indices[x != 0]
      df=np.append(df,np.size(AS))
      err = np.append(err,np.sum(np.power(np.matmul(A,x)-b,2)))
      X = np.hstack((X,x))
      Cp = np.append(Cp,err[k-1]/sigma2e - n +2*df[k-1])
      return df,err,X,Cp
    j = I[j]
    
    if ~lassocond:
      R  = cholinsert(R, A[:,j], A[:,P]);
      P = np.hstack((P,j))
      I = np.delete(I,np.argwhere(I == j))
      cnvars = cnvars + 1
      
    s = np.ones((cnvars,1))
    if isinstance(s,np.float64) or isinstance(R,np.float64):
        GA1 = (s/R)/R 
    else:
        GA1 = np.linalg.lstsq(R,np.linalg.lstsq(R.T,s,rcond=1e-4)[0],rcond=None)[0]
    GA1 = GA1.reshape(GA1.size,1)
    AA = 1/np.sqrt(np.sum(GA1))
    w = AA*GA1
    
    u = np.matmul(A[:,P],w)
    if cnvars == nvars:
      gamma = C/AA
    else:
      a = np.matmul(A.T,u)
      temp = (C - c[I])/(AA - a[I])
      temp = temp[temp>np.finfo(float).eps]
      gamma = np.amin(np.vstack((temp.reshape(len(temp),1),np.asarray([C/AA]).reshape((1,1)))))
    
    lassocond = 0
    temp = -x[P]/w
    temp = temp.T
    gamma_tilde = np.amin(np.hstack((temp[temp > 0],gamma)))
    j = np.argwhere(np.abs(temp - gamma_tilde) < np.finfo(float).eps)
    if gamma_tilde < gamma:
      gamma = gamma_tilde
      lassocond = 1
    
    x[P] = x[P] + gamma*w
    
    if lassocond == 1:
      lj = len(j)
      for jj in range(lj):
          R = choldown(R,j[jj][1])
          I = np.append(I,P[j[jj][1]])
          P = np.delete(P,j[jj][1])
    cnvars = cnvars - len(j)
      
    if maxvar > 0:
      earlystopcond = cnvars >= maxvar
    
    c = np.matmul(A.T,b-np.matmul(A[:,P],x[P]))
    AS = indices[x != 0]
    if k ==1:
        df = len(AS)
        err = np.sum(np.power(np.matmul(A,x)-b,2))
        X = x
        Cp = err/sigma2e - n +2*df
    else:
        df=np.append(df,len(AS))
        err=np.append(err,np.sum(np.power(np.matmul(A,x)-b,2)))
        X=np.hstack((X,x))
        Cp=np.append(Cp,err[k-1]/sigma2e - n +2*df[k-1])
  return df,err,X,Cp

def sms_DRT(Ar,Ai,Z,lamb=None,i_L=0,f=None):
    import numpy as np
    import lsq_lin
    Zr = Z.real
    Zi = Z.imag
    if lamb is None:
        lamb = np.logspace(-10,1,10)
    l = len(lamb)
    [n,m] = Ai.shape
    beta = np.zeros((m,l))
    Rp = np.zeros((1,l))
    Rinf = np.zeros((1,l))
    L = np.zeros((1,l))
    Error = np.zeros((1,l))
    
    for ii in range(l):
        reg = lamb[ii]
        if i_L == 0:
            df,err,X,Cp = NN_LARS(Ai,Zi,reg)
            index = np.argmin(Cp)
            beta_aug = X[:,index]
            beta_hat = (1+reg)**0.5*beta_aug
            rp = np.sum(beta_hat) # need to check which axis to sum
        else:
            LL = 0
            for jj in range(20):
                df,err,X,Cp = NN_LARS(Ai,Zi-2*np.pi*LL*f,reg)
                index = np.argmin(Cp)
                beta_aug = X[:,index]
                beta_hat = (1+reg)**0.5*beta_aug
                rp = np.sum(beta_hat) # need to check which axis to sum
                dum = lsqlin.lsqnonneg(2*np.pi*f,Zi-(np.matmul(Ai,beta_hat)),{'show_progress': False}) 
                LL = dum['x'][0]
            LL = dum['x'][0]
        
        rr = np.sum(Zr.T-(np.matmul(Ar,beta_hat)))/n
        
        beta[:,ii] = beta_hat
        Rp[:,ii] = rp
        Rinf[:,ii] = rr
        re_err = np.matmul((Zr.T-rr-np.matmul(Ar,beta_hat)),(Zr.T-rr-np.matmul(Ar,beta_hat)).T)
        if i_L == 0:
            im_err = np.matmul((Zi.T-np.matmul(Ai,beta_hat)),(Zi.T-np.matmul(Ai,beta_hat)).T)
        else:
            im_err = np.matmul((Zi.T- 2*np.pi*LL*f-np.matmul(Ai,beta_hat)),(Zi.T- 2*np.pi*LL*f-np.matmul(Ai,beta_hat)).T)
        Error[:,ii] = re_err+im_err
        
    index = np.argmin(Error)
    R_p = Rp[:,index]
    R_inf = Rinf[:,index]
    Beta = beta[:,index]/Rp[:,index]
    Z_real = Rinf[:,index]+np.matmul(Ar,beta[:,index])
    
    if i_L == 0:
        inductance = 'not considered'
        Z_img = np.matmul(Ai,beta[:,index])
    else:
        inductance = L[:,index]
        Z_img = np.matmul(Ai,beta[:,index]) + 2*np.pi*L[:,index]*f
        
    model = {
            'Rp':R_p,
            'Rinf' : R_inf,
            'beta' : Beta,
            'Zreal': Z_real,
            'Zimag': Z_img,
            'inductance': inductance
        }
    return model

def pyDRT(w,Z,t=None,Ar=None,Ai=None,L=0):  
    import numpy as np
    if t is None:
        NF = len(w)
        NT = int(NF*1.5)
        f = w
        t = np.logspace(-np.log10(np.max(w)),-np.log10(np.min(w)),NT)
        Ar, Ai = cal_Basis(f,t)
    else:
        f = w
    reg = np.logspace(-10,1,10)
    model = sms_DRT(Ar,Ai,Z,reg,i_L=L,f=f)

    tc = (t[1:]+t[:-1])/2
    model.update({
            'tau' : tc,
            })
    return model

def plot_sim(model,Z,t):
    import matplotlib.pyplot as plt
    DRT_est = model['beta']
    Z_est_r = model['Zreal']
    Z_est_i = model['Zimag']
    tc = (t[1:]+t[:-1])/2

    plt.figure()
    plt.semilogx(tc,DRT_est,'r-x',label='estimation')
    plt.legend()
    plt.xlabel(r'$\tau$ s')
    plt.ylabel(r'G($\tau$)')
    
    plt.figure()
    plt.plot(Z.real,-Z.imag,'k-o',label='simulation')
    plt.plot(Z_est_r,-Z_est_i,'r-*',label='DRT estimation')
    plt.axis('equal')
    plt.legend()
    plt.xlabel('Re(Z) $\Omega$')
    plt.ylabel('-Im(Z) $\Omega$')
    