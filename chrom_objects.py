import galsim
import numpy as np

def gal_chr(r_b,r_d,n_b,n_d,t_flux,t_b_flux,**kwargs):
    b_SED = kwargs.get('b_SED', None)
    d_SED = kwargs.get('d_SED', None)
    c_SED = kwargs.get('c_SED', None)
    x0=kwargs.get('x0',0)
    y0=kwargs.get('y0',0)
    e1_gal = kwargs.get('e1', 0)
    e2_gal = kwargs.get('e2', 0)
    e1_b   = kwargs.get('e1_b',0)
    e2_b   = kwargs.get('e2_b',0)
    e1_d   = kwargs.get('e1_d',0)
    e2_d   = kwargs.get('e2_d',0)

    m_bulge=galsim.Sersic(n=n_b,half_light_radius=r_b,flux=t_flux*t_b_flux)
    m_disk=galsim.Sersic(n=n_d,half_light_radius=r_d,flux=t_flux*(1-t_b_flux))
    m_bulge=m_bulge.shear(e1=e1_b,e2=e2_b)
    m_disk=m_disk.shear(e1=e1_d,e2=e2_d)
    if b_SED :
        gal=(m_bulge*b_SED)+(m_disk*d_SED)
    elif c_SED:
        gal=(m_bulge+m_disk)*c_SED
    else:
        print('No SED given. Galaxy is achromatic')
    gal = gal.shear(e1=e1_gal,e2=e2_gal)
    return gal

def gal_chr_gaus(r_b,r_d,t_flux,t_b_flux,**kwargs):
    b_SED = kwargs.get('b_SED', None)
    d_SED = kwargs.get('d_SED', None)
    c_SED = kwargs.get('c_SED', None)
    x0=kwargs.get('x0',0)
    y0=kwargs.get('y0',0)
    e1_gal = kwargs.get('e1', 0)
    e2_gal = kwargs.get('e2', 0)
    e1_b   = kwargs.get('e1_b',0)
    e2_b   = kwargs.get('e2_b',0)
    e1_d   = kwargs.get('e1_d',0)
    e2_d   = kwargs.get('e2_d',0)
    
    m_bulge=galsim.Gaussian(half_light_radius=r_b,flux=t_flux*t_b_flux)
    m_disk=galsim.Gaussian(half_light_radius=r_d,flux=t_flux*(1-t_b_flux))
    m_bulge=m_bulge.shear(e1=e1_b,e2=e2_b)
    m_disk=m_disk.shear(e1=e1_d,e2=e2_d)
    if b_SED :
        gal=(m_bulge*b_SED)+(m_disk*d_SED)
    elif c_SED:
        gal=(m_bulge+m_disk)*c_SED
    else:
        print('No SED given. Galaxy is achromatic')
    gal = gal.shear(e1=e1_gal,e2=e2_gal)
    return gal


#create chromatic psf and image
def psf_chr(base_wavelength,alpha,**kwargs ):
    #beta must be in degrees
    fwhm=kwargs.get('fwhm', None)
    half_light_radius=kwargs.get('half_light_radius', None)
    sigma=kwargs.get('sigma',None)
    g=kwargs.get('g',0.)
    beta=kwargs.get('beta',0.)
    if fwhm:
        mono_PSF = galsim.Gaussian(fwhm=fwhm,flux=1)
    elif half_light_radius:
        mono_PSF = galsim.Gaussian(half_light_radius=half_light_radius,flux=1)
    elif sigma:
        mono_PSF=galsim.Gaussian(sigma=sigma,flux=1)
    else:
        raise AttributeError('Not enough parametrs defined. Give either fwhm, sigma or half_light_radius' )
    chrom_PSF=galsim.ChromaticObject(mono_PSF)
    chrom_PSF= chrom_PSF.dilate(lambda w: (w/base_wavelength)**(alpha))
    chrom_PSF = chrom_PSF.shear(g=g,beta=beta*galsim.degrees)
    return chrom_PSF


def im_chr(gal,psf,I_size,pixel_scale,filter1):
    img = galsim.ImageF(I_size, I_size, scale=pixel_scale)
    conv = galsim.Convolve([gal, psf])
    gal_im=conv.drawImage(filter1,image=img)
    return gal_im

def ring_test_moments1(galaxy,psf,psf_im,im_param):
    g_i=np.array([[0.01,0.01],[0.015,0.015]])
    phi=[0,90,30,120,60,150]
    g_obs=[[],[]]
    rt_e=0.1
    for g in g_i:
        sum=[0,0]
        n=0.
        
        for rt_phi in phi:

            rt_shr=galsim.Shear(e=rt_e,beta=rt_phi*galsim.degrees)
            galaxy1=galaxy.shear(e1=rt_shr.e1,e2=rt_shr.e2)
            galaxy1=galaxy1.shear(g1=g[0],g2=g[1])
            gal_im=im_chr_test(galaxy1,psf,im_param)
            result  = galsim.hsm.EstimateShear(gal_im,psf_im,shear_est='REGAUSS',strict=True)#,weight=weight_fn)
            a=galsim.Shear(e1=result.corrected_e1,e2=result.corrected_e2)
            sum[0]+=a.g1
            sum[1]+=a.g2
            n+=1

        g_obs[0].append(sum[0]/n)
        g_obs[1].append(sum[1]/n)
    regression = np.polyfit([g_i[0][0],g_i[1][0]], g_obs[0], 1)
    m1=(1-regression[0])
    c1=(regression[1])
    regression = np.polyfit([g_i[0][1],g_i[1][1]], g_obs[1], 1)
    m2=(1-regression[0])
    c2=(regression[1])
    return [m1,m2],[c1,c2],g_obs

















