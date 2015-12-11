import galsim
import os
import numpy as np
import matplotlib.pyplot as plt
import cg_fns_reloaded as cg 


def bias_all(Args, rt_g=[[0.005,0.005],[0.01,0.01]]):
    """Plot CG bias for Regauss, KSB & model fitting
    at different redshfts. Functions are imported from cg_fns_reloaded.py
    @param Args    Class containing parameters 
    @param rt_g    shear values applied to measure bias  
    """
    gtrue=np.array(rt_g)
    m_z_re, m_z_fit, m_z_ksb =[],[],[]
    c_z_re, c_z_fit,c_z_ksb =[],[],[]
    redshifts=np.linspace(0.,1.2,15)
    for z in redshifts:
        #REGAUSS
        input_p = Args
        input_p.shear_est = 'REGAUSS'
        input_p.redshift = z
        input_p.rt_g = rt_g
        gcg,gnocg=cg.calc_cg_new(input_p)
        fit_fin   = np.polyfit(gtrue.T[0],gcg.T-gnocg.T,1)
        m_z_re.append(fit_fin[0])
        c_z_re.append(fit_fin[1])
        #KSB
        input_p = Args
        input_p.shear_est = 'KSB'
        input_p.redshift = z
        input_p.rt_g = rt_g
        gcg,gnocg = cg.calc_cg_new(input_p, calc_weight=True)
        fit_fin   = np.polyfit(gtrue.T[0],gcg.T-gnocg.T,1)
        m_z_ksb.append(fit_fin[0])
        c_z_ksb.append(fit_fin[1])
        #fit
        input_p = Args
        input_p.shear_est = 'fit'
        input_p.redshift = z
        input_p.rt_g = rt_g
        gcg,gnocg = cg.calc_cg_new(input_p)
        fit_fin   = np.polyfit(gtrue.T[0],gcg.T-gnocg.T,1)
        m_z_fit.append(fit_fin[0])
        c_z_fit.append(fit_fin[1])
        
    #Plots
    plt.rc('legend',**{'fontsize':12})
    plt.figure(figsize=[18,14])
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace = 0.5)
    plt.subplot(2,2,1)
    if Args.telescope is 'Euclid':
        plt.plot(redshifts, -np.array(m_z_re).T[0],label='REGAUSS', linewidth=2.5)
        plt.plot(redshifts, -np.array(m_z_ksb).T[0]*10,label='KSB$\\times 10$',
                 linewidth=2.5)
        plt.plot(redshifts, -np.array(m_z_fit).T[0],label='Model Fitting',
                 linewidth=2.5)
        plt.ylabel(r'-$\rm m_{CG}$',size=22)
    else :
        plt.plot(redshifts, np.array(m_z_re).T[0],label='REGAUSS')
        plt.plot(redshifts, np.array(m_z_ksb).T[0],label='KSB$\\times 10$',
                 linewidth=2.5)
        plt.plot(redshifts, np.array(m_z_fit).T[0],label='Model Fitting',
                 linewidth=2.5)
        plt.ylabel(r'$\rm m_{CG}$',size=22)
    plt.xlabel(r'redshift',size=20)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.subplot(2,2,2)    
    plt.plot(redshifts, np.array(c_z_re).T[0],label='REGAUSS',linewidth=2.5)
    plt.plot(redshifts, np.array(c_z_ksb).T[0],label='KSB', linewidth=2.5)
    plt.plot(redshifts, np.array(c_z_fit).T[0],label='Model Fitting',
             linewidth=2.5)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ylabel(r'$\rm c_{CG}$',size=22)
    plt.xlabel(r'redshift',size=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.suptitle(r'Bias in shape Measurement for {0}'.format(Args.telescope), size=24)



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



def bias_KSB_weight(Args, gal_hlr):
    """Make plots to analyze CG bias with weight function""" 
    mcg_w_ksb, ccg_w_ksb = [],[] 
    weights=np.linspace(0.5,2, 15)
    rt_g=[[0.005,0.005],[0.01,0.01]]
    gtrue=np.array(rt_g)
    for w in weights:
        input_args = Args
        input_args.shear_est='KSB'
        input_args.sig_w = w*gal_hlr
        input_args.rt_g = rt_g
        gcgw,gnocgw=cg.calc_cg(input_args)
        fit_fin   = np.polyfit(gtrue.T[0],gcgw.T-gnocgw.T,1)
        mcg_w_ksb.append(fit_fin[0])
        ccg_w_ksb.append(fit_fin[1])

    k=0
    plt.rc('legend',**{'fontsize':16})
    plt.figure(figsize=[12,8])
    #plt.subplots_adjust(hspace=0.4)
    #plt.subplots_adjust(wspace = 0.4)
    #plt.subplot(221)
    if Args.telescope is 'Euclid':
        plt.plot(weights,-np.array(mcg_w_ksb).T[k],label='KSB',linewidth=2.5)
        plt.ylabel(r'-$\rm m_{CG}$', size=22)
    else:
        plt.plot(weights,np.array(mcg_w_ksb).T[k],label='KSB',linewidth=2.5)
        plt.ylabel(r'$\rm m_{CG}$', size=22)
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    plt.xlabel(r'weight function size/galaxy size',size=20)    
    plt.title(r'Dependence of $\rm m_{CG}$ '+'on weight function size for {0}'.format(Args.telescope),
              size=22)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    #plt.subplot(222)
    #plt.plot(weights,np.array(ccg_w_ksb).T[k],label='KSB')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.title('Variation of bias($c_{CG}$) with weight fn size', size=16)
    #plt.xlabel('weight fn size($\sigma_w/r_h$)',size=16)
    #plt.ylabel('-$m_{CG,S}$', size=19)

    


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def bias_alpha(Args, gal_hlr, rt_g=[[0.005,0.005],[0.01,0.01]]):
    """Plot CG bias for the 4 HSM methods: Regauss, BJ, Linearization & KSB
    at different redshfts. Functions are imported from cg_fns_reloaded.py
    @param Args    Class containing parameters 
    @param rt_g    shear values applied to measure bias  
    """
    gtrue = np.array(rt_g)
    m_a_re, m_a_ksb =[],[]
    c_a_re, c_a_ksb =[],[]
    alphas = np.linspace(-1, 1, 10)
    for alpha in alphas:
        #REGAUSS
        input_p = Args
        input_p.alpha = alpha
        input_p.rt_g = rt_g
        gcg, gnocg = cg.calc_cg_new(input_p)
        fit_fin = np.polyfit(gtrue.T[0],gcg.T-gnocg.T,1)
        m_a_re.append(fit_fin[0])
        c_a_re.append(fit_fin[1])
        
    #Plots
    plt.rc('legend',**{'fontsize':16})
    plt.figure(figsize=[12, 8])
    #plt.subplots_adjust(hspace=0.4)
    #plt.subplots_adjust(wspace = 0.4)
    #plt.subplot(221)
    plt.plot(alphas, np.array(m_a_re).T[0],
             marker='s', color='b', label='REGAUSS 1', markersize=12)
    plt.plot(alphas, np.array(m_a_re).T[1],
             marker='^', color='g', label='REGAUSS 2', markersize=12)
    plt.axhline(0.0, color='k', linestyle='-.')
    plt.axvline(0, color='k', linestyle='-.')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(r'Multiplicative bias for {0} using {1}'.format(Args.telescope,
                                                              Args.shear_est),
               size=22)
    plt.ylabel(r' $\rm m_{CG}$', size=22)
    plt.xlabel(r'PSF size $\lambda$ scaling exponent, $\alpha$', size=20) 
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





