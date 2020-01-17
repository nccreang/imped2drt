#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:36:57 2018
Last Updated: Jan 17 2020
@author: NicoleCreange

Based on work by Yoed Tsur
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

def KKT_i2r(ww,z): 
    l = len(z)
    KKTr = []
    ww= ww*2*np.pi
    for ii in range(0,l):
        KKT = []
        for jj in range(0,l):
            if jj!=ii:
                x = (ww[jj]*z[jj].imag - ww[ii]*z[ii].imag)/(ww[jj]**2 - ww[ii]**2)
            if jj==ii and jj!=0 and jj!=l-1:
                x = 0.5*((z[jj].imag/ww[jj]) + ((z[jj+1].imag - z[jj-1].imag)/(ww[jj+1] - ww[jj-1])))
            if jj==ii and jj==0 and jj!=l-1:
                x = 0.5*(z[jj].imag/ww[jj] + ((z[jj+1].imag-z[jj].imag)/(ww[jj+1]-ww[jj])))
            if jj==ii and jj!=0 and jj==l-1:
                x = 0.5*((z[jj].imag/ww[jj]) + ((z[jj].imag-z[jj-1].imag)/(ww[jj]-ww[jj-1])))
            KKT.append(x)
        from scipy.interpolate import CubicSpline as scaps
        cs = scaps(ww,KKT)
        rekk = 0
        for mm in range(l-1):
            trap = (KKT[mm+1] + KKT[mm])/2
            dw = ww[mm+1]-ww[mm]
            rekk = rekk + trap*dw
        KKTr.append((2/np.pi)*rekk + z[-1].real)
    return KKTr
def KKT_r2i(ww,z): #what Yoed has
    l = len(z)
    ww= ww*2*np.pi
    KKTi = []
    for ii in range(0,l):
        KKT = []
        for jj in range(0,l):
            if jj!=ii:
                x = (z[jj].real - z[ii].real)/(ww[jj]**2 - ww[ii]**2)
            if jj==ii and jj!=0 and jj!=l-1:
                x = ((z[jj+1].real - z[jj-1].real)/(ww[jj+1] - ww[jj-1]))/(2*ww[jj])
            if jj==ii and jj==0 and jj!=l-1:
                x = ((z[jj+1].real - z[jj].real)/(ww[jj+1]-ww[jj]))/(2*ww[jj])
            if jj==ii and jj!=0 and jj==l-1:
                x = ((z[jj].real - z[jj-1].real)/(ww[jj]-ww[jj-1]))/(2*ww[jj])
            KKT.append(x)
        from scipy.interpolate import CubicSpline as scaps
        cs = scaps(ww,KKT)
        imkk = 0
        for mm in range(l-1):
            trap = (KKT[mm+1] + KKT[mm])/2
            dw = ww[mm+1]-ww[mm]
            imkk = imkk + trap*dw
        KKTi.append((2*ww[ii]/np.pi)*imkk)
    return KKTi

#%%

def KKT(w,Z,thres=0.2,clean=False):
    import matplotlib.pyplot as plt
    # check for order of data.  Needs to be low frequency to high frequency
    order = w[0]-w[-1]
    print(order)
    if order > 0:
        w = np.flipud(w)
        Z = np.flipud(Z)
    zr = Z.real
    zi = Z.imag
    z = zr-1j*zi
    KKTimag = KKT_r2i(w,z)
    KKTreal = KKT_i2r(w,z)
    KKw = w
    fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(8,5))
    if clean==True:
        KKTdata = np.asarray(KKTreal)+1j*np.asarray(KKTimag)
        wc,Zdata = KKT_clean(w,Z,KKTdata,thres)
        axs[0,1].semilogx(wc,Zdata.real,'b.',label = "$Z_{exp}$'")
        axs[0,1].semilogx(wc,-Zdata.imag,'k.',label = '$Z_{exp}$"')
        axs[0,1].semilogx(w,Z.real,'b:')
        axs[0,1].semilogx(w,-Z.imag,'k:')
        axs[0,1].semilogx(KKw,np.asarray(KKTreal),'g',label = 'KKT real')
        axs[0,1].semilogx(KKw,-np.asarray(KKTimag),'r',label = 'KKT imag')
        axs[0,1].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        axs[0,1].set_xlabel('Frequency (Hz)')
        axs[0,1].set_ylabel('|Z|')
        axs[0,1].legend(fontsize=8)
        return fig,axs,wc,Zdata
    else:
        axs[0,1].semilogx(w,Z.real,'b.',label = "$Z_{exp}$'")
        axs[0,1].semilogx(w,-Z.imag,'k.',label = '$Z_{exp}$"')
        axs[0,1].semilogx(KKw,np.asarray(KKTreal),'g',label = 'KKT real')
        axs[0,1].semilogx(KKw,-np.asarray(KKTimag),'r',label = 'KKT imag')
        axs[0,1].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        axs[0,1].set_xlabel('Frequency (Hz)')
        axs[0,1].set_ylabel('|Z|')
        axs[0,1].legend(fontsize=8)
        return fig,axs,w,Z

def KKT_clean(w,Z,KKTdata,thres):
    import numpy as np
    
    res_real = (KKTdata.real-Z.real)
    res_imag = (KKTdata.imag-Z.imag)
    if np.sum(res_real) > np.sum(res_imag):
        rem_r = [np.abs(res_real) > np.abs(Z.real)*thres]
        indx_r = np.where(rem_r[0]==True)
        z_clean = np.delete(Z,indx_r)
        wc = np.delete(w,indx_r)
    else:
        rem_i = [np.abs(res_imag) > np.abs(Z.imag)*thres]
        indx_i = np.where(rem_i[0]==True)
        z_clean = np.delete(Z,indx_i)
        wc = np.delete(w,indx_i)
    return wc,z_clean

