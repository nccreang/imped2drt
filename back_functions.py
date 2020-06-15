#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:30:55 2019
Last updated: Jan 17 2020
@author: NicoleCreange
"""

# back functions for fitting, plotting, etc.
def bash_load(impeds,filenames,dims,data_type = 'Y',plot=True,KKT=True,drt=True,L=0,clean=False,thres=0.2):
    import numpy as np
    from tqdm import tqdm

    fulldata = []
    fullDRTs = []
    A_real = []
    A_imag = []
    for ii in tqdm(range(len(impeds))):
        if data_type == 'Y':
            Zdata = convertYtoZ(impeds[ii])
        else:
            if int(impeds[ii][0,2])!=0 or impeds[ii][0,1]>1e2: #assumes length between electrodes will be less than 100 cm
                dims = dims
                Zdata = impeds[ii]
            else:
                Zdata = impeds[ii][1:,:]
                dims = impeds[ii][0,0:2]
        if monotonicity_check(Zdata[:,0])==False:
            Zdata = Zdata[Zdata[:,0].argsort()]
#            if monotonicity_check(Zdata[:,0])==False:
#                indx = [ii for ii in range(len(Zdata)-1) if Zdata[ii,0]!= Zdata[ii+1,0]]
#                Zdata = np.take(Zdata,indx)
        w,Zexp,Yexp,Mexp,Eexp,data,DRT_data,Ar,Ai = load_data(Zdata,filenames[ii],dims,plot=plot,KKT=KKT,drt=drt,L=L,clean=clean,thres=thres)
        fulldata.append(np.concatenate((w,data),axis=1))
        fullDRTs.append(DRT_data)
        A_real.append(Ar)
        A_imag.append(Ai)
    if drt is True:
        Amatrix = {
                'Ar' : Ar,
                'Ai' : Ai,
                }
    return fulldata,fullDRTs,Amatrix

def monotonicity_check(freq):
    def increasing(freq):
        return all(x>y for x,y in zip(freq,freq[1:]))
    def decreasing(freq):
        return all(x<y for x,y in zip(freq,freq[1:]))
    return increasing(freq) or decreasing(freq)

def load_data(imped,filename,dims,plot = True,KKT=True,drt=True,t=None,Ar=None,Ai=None,L=0,clean=False,thres=0.2):
    import numpy as np
    import matplotlib.pyplot as plt
    import KKlib
    import pyDRT
    
    dim = dims
    fog = imped[:,0]
    wog = fog*2*np.pi
    wog.astype(float)  
    Zexpog = imped[:,1]+imped[:,2]*1j
    Mexpog = 1j*wog*Zexpog*8.85e-14*dim[0]/dim[1]      
    if plot is True:
        if KKT is True:
            fig,axs,w,Zexp = KKlib.KKT(wog,Zexpog,thres=thres,clean=clean)
            Yexp = Zexp**-1                                 
            Mexp = 1j*w*Zexp*8.85e-14*dim[0]/dim[1]      
            Eexp = Mexp**-1  
            data = np.array([Zexp,Yexp,Mexp,Eexp]).T
            f = w/(2*np.pi)
            fig.suptitle(filename)
            axs[0,0].plot(Zexpog.real,-Zexpog.imag,':k',label='Data')
            axs[0,0].plot(Zexp.real,-Zexp.imag,'.k',label='Data')
            axs[0,0].set_ylabel('-Z" ($\Omega$)')
            axs[0,0].set_xlabel("Z' ($\Omega$)")
            axs[0,0].ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
            axs[0,0].ticklabel_format(style='sci',axis='x',scilimits=(0,0))##############
            axs[1,0].loglog(f,Mexp.imag,'.k')
            axs[1,0].loglog(fog,Mexpog.imag,':k')
            axs[1,0].set_ylabel('M"')
            axs[1,0].set_xlabel('Frequency (Hz)')
        else: 
            fig,axs = plt.subplots(2,2)
            fig.suptitle(filename)
            w=wog
            Zexp = Zexpog
            Yexp = Zexp**-1                                 
            Mexp = 1j*w*Zexp*8.85e-14*dim[0]/dim[1]      
            Eexp = Mexp**-1  
            data = np.array([Zexp,Yexp,Mexp,Eexp]).T 
            f = w/(2*np.pi)
            axs[0,0].plot(Zexp.real,-Zexp.imag,'.k',label='Data')
            axs[0,0].set_ylabel('-Z" ($\Omega$)')
            axs[0,0].set_xlabel("Z' ($\Omega$)")
            axs[0,0].ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
            axs[0,0].ticklabel_format(style='sci',axis='x',scilimits=(0,0))##############
            axs[1,0].loglog(f,Mexp.imag,'.k')
            axs[1,0].set_ylabel('M"')
            axs[1,0].set_xlabel('Frequency (Hz)')  
        
        freqmax = max(f)
        freqmin = min(f)
        decades = int(np.log10(freqmax))-int(np.log10(freqmin))+1
        xvals = []
        yvals = []
        ind = []
        j=-1
        for i in range(int(np.log10(freqmin)),decades+int(np.log10(freqmin))):
            j=j+1
            xvals.append(min(f,key=lambda x:abs(x-10**i)))
            ind.append(np.where(f==xvals[j]))
            if len(ind[j][0])>1:
                yvals.append(Zexp[ind[j][0][0]])
            else:
                yvals.append(Zexp[ind[j]])
        yvals = np.asarray(yvals)
        xvals = np.asarray(xvals)
        axs[0,0].scatter(yvals.real,-yvals.imag,color='red')
        for i in range(len(xvals)):
            axs[0,0].annotate('%.0e'%int(xvals[i]),xy=(yvals[i].real,-yvals[i].imag),fontsize=10)
        fig.tight_layout()

        if drt is True:
            t = np.logspace(-np.log10(np.max(w))-2,-np.log10(np.min(w))+2,len(f)*2)
            Ar,Ai = pyDRT.cal_Basis(wog,t)
            DRT_data = calc_DRT(wog,Zexpog,t=t,Ar=Ar,Ai=Ai,L=L)
            tau = DRT_data['tau']
            drt = DRT_data['beta']
            Zrsim = DRT_data['Zreal']
            Zisim = DRT_data['Zimag']
            
            axs[1,1].semilogx(tau,drt,'k',linewidth=2)
            axs[1,1].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
            axs[1,1].set_ylabel('DRT')
            axs[1,1].set_xlabel('Relaxation Time (s)')
            axs[0,0].plot(Zrsim,-Zisim,'--g',label='DRT simulation')
#            axs[0,0].legend(fontsize=7)
        else: 
            DRT_data = []
    return w.reshape(w.shape[0],1),Zexp,Yexp,Mexp,Eexp,data,DRT_data,Ar,Ai


def calc_DRT(f,Zexp,t=None,Ar=None,Ai=None,L=0):
    import pyDRT
    DRT_data = pyDRT.pyDRT(f,Zexp,t,Ar,Ai,L=L)
    return DRT_data
    
def convertYtoZ(data):
    import numpy as np
    w = data[:,0]
    w = w.reshape(w.shape[0],1)
    Yr = data[:,1]
    Yi = data[:,2]
    Z = 1/(Yr+1j*Yi)
    Z = Z.reshape(Z.shape[0],1)
    return np.concatenate([w,Z.real,Z.imag],axis=1)
    

def plotall(data,dim,name):
    import matplotlib.pyplot as plt
    import numpy as np
    w = data[:,0]/(2*np.pi)
    Z = data[:,1]
    Y = data[:,2]
    M = data[:,3]
    E = data[:,4]
    plt.figure(figsize=(7,5))
    plt.suptitle(name)
    plt.subplot(3,4,1); plt.plot(Z.real,-Z.imag,'.k')
    plt.title('$Z^*$',fontsize=16)
    plt.ylabel('Complex')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))##############
    plt.subplot(3,4,2); plt.loglog(Y.real,Y.imag,'.k')
    plt.title('$Y^*$',fontsize=16)
    plt.subplot(3,4,3); plt.loglog(M.real,M.imag,'.k')
    plt.title('$M^*$',fontsize=16)
    plt.subplot(3,4,4); plt.plot(E.real,E.imag,'.k')
    plt.title('$\epsilon^*$',fontsize=16)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))##############
    plt.subplot(3,4,5); plt.semilogx(w,Z.real,'.k')
    plt.ylabel('real')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.subplot(3,4,6); plt.loglog(w,Y.real,'.k')
    plt.subplot(3,4,7); plt.semilogx(w,M.real,'.k')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.subplot(3,4,8); plt.semilogx(w,E.real,'.k')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.subplot(3,4,9); plt.semilogx(w,-Z.imag,'.k')
    plt.ylabel('imaginary')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.subplot(3,4,10); plt.loglog(w,Y.imag,'.k')
    plt.subplot(3,4,11); plt.loglog(w,M.imag,'.k')
    plt.subplot(3,4,12); plt.semilogx(w,E.imag,'.k')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    
def plot_formalisms(q,eq,w,dim,data,name):
    import matplotlib.pyplot as plt
    q=q
    Z = eval(eq)
    Y = Z**-1
    M = 1j*w*Z*8.85e-14*dim[0]/dim[1]
    E = M**-1
    Zexp = data[:,0]
    Yexp = data[:,1]
    Mexp = data[:,2]
    Eexp = data[:,3]
    plt.figure(figsize=(7,5))
    plt.suptitle(name)
    plt.subplot(3,4,1); plt.plot(Zexp.real,-Zexp.imag,'.k',Z.real,-Z.imag,'-r')
    plt.title('$Z^*$',fontsize=16)
    plt.ylabel('Complex')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))##############
    plt.subplot(3,4,2); plt.loglog(Yexp.real,Yexp.imag,'.k',Y.real,Y.imag,'-r')
    plt.title('$Y^*$',fontsize=16)
    plt.subplot(3,4,3); plt.loglog(Mexp.real,Mexp.imag,'.k',M.real,M.imag,'-r')
    plt.title('$M^*$',fontsize=16)
    plt.subplot(3,4,4); plt.plot(Eexp.real,Eexp.imag,'.k',E.real,E.imag,'-r')
    plt.title('$\epsilon^*$',fontsize=16)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))##############
    plt.subplot(3,4,5); plt.semilogx(w,Zexp.real,'.k',w,Z.real,'-r')
    plt.ylabel('real')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.subplot(3,4,6); plt.loglog(w,Yexp.real,'.k',w,Y.real,'-r')
    plt.subplot(3,4,7); plt.semilogx(w,Mexp.real,'.k',w,M.real,'-r')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.subplot(3,4,8); plt.semilogx(w,Eexp.real,'.k',w,E.real,'-r')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.subplot(3,4,9); plt.semilogx(w,-Zexp.imag,'.k',w,-Z.imag,'-r')
    plt.ylabel('imaginary')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.subplot(3,4,10); plt.loglog(w,Yexp.imag,'.k',w,Y.imag,'-r')
    plt.subplot(3,4,11); plt.loglog(w,Mexp.imag,'.k',w,M.imag,'-r')
    plt.subplot(3,4,12); plt.semilogx(w,Eexp.imag,'.k',w,E.imag,'-r')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))##############
    plt.pause(0.05)
        
    
def DRT_estimate(DRT_data,Zexp,w):
    import scipy.signal as ss
    import numpy as np
    R_est =[]
    C_est =[]
    
    tau = DRT_data['tau']
    drt = DRT_data['beta']
    Rp = DRT_data['Rp']
    t_peaks = ss.find_peaks(drt,width=5)[0]
    t_vals = tau[t_peaks]
    R_est = Rp*(drt[t_peaks]/np.sum(drt[t_peaks]))
    C_est = t_vals*2*np.pi/R_est
    return R_est,C_est,t_vals
    
    
