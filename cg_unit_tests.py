import galsim
import os
from astropy.utils.console import ProgressBar
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
import cg_fns_reloaded as cg 
def range_print(val,rng):
	""" Print if test is pass or FAIL
	@param val    value that has to be tested
	@param rng    Range within which val must lie to pass"""
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	if (np.all(val>-rng) and np.all(val<rng)):
		print colored('PASS', 'green')
	else:
		print colored('FAIL', 'red')
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

def cg_unit_tests():
	""" Sanity checks for cg_fns_reloaded.py 

	Compute the CG bias for various galaxy parameters and test is they match
	expected values. Test only for Euclid for the time being """
	T=6
	# bulge_frac=0
	input_args = cg.Eu_Args(shear_est='REGAUSS', bulge_frac=0.)
	g1,g2 = cg.calc_cg(input_args)
	print 'NO Disk : gcg,gncg'
	print g1.T[0],g2.T[0]
	print 'CG,S bias', (g1/g2-1).T[0]
	range_print((g1/g2-1).T[0],1e-05)
	
	# bulge_frac=1
	input_args = cg.Eu_Args(shear_est='REGAUSS', bulge_frac=1.)
	g1,g2 = cg.calc_cg(input_args)
	print 'NO Bulge:  gcg,gncg'
	print g1.T[0],g2.T[0]
	print 'CG,S bias', (g1/g2-1).T[0]
	range_print((g1/g2-1).T[0],1e-05)

	#bulge disk have same SED
	input_args = cg.Eu_Args(shear_est='REGAUSS', bulge_frac=1.)
	input_args.bulge_SED_name = 'E'
	input_args.disk_SED_name = 'E'
	g1,g2 = cg.calc_cg(input_args)
	print 'NO Bulge:  gcg,gncg'
	print g1.T[0],g2.T[0]
	print 'CG,S bias', (g1/g2-1).T[0]
	range_print((g1/g2-1).T[0],1e-05)


	
	# REGAUSS
	input_args = cg.Eu_Args(shear_est='REGAUSS')
	g1_r,g2_r = cg.calc_cg(input_args)
	print 'Default parameters, REGAUSS : gcg,gncg'
	print g1_r.T[0],g2_r.T[0]
	print 'CG,S bias', (g1_r/g2_r-1).T[0]
	
	# KSB
	input_args = cg.Eu_Args(shear_est='KSB')
	g1_k,g2_k = cg.calc_cg(input_args)
	print 'Default parameters, KSB : gcg,gncg'
	print g1_k.T[0],g2_k.T[0]
	print 'CG,S bias', (g1_k/g2_k-1).T[0]
	val=(g1_r/g2_r-1).T[0]-(g1_k/g2_k-1).T[0]
	range_print(val,1e-02)
	
	# KSB with REGAUSS weight size
	input_args = cg.Eu_Args(shear_est='KSB',sig_w=0.19)
	g1_k,g2_k = cg.calc_cg(input_args)
	print 'With weight sigma = 0.19, KSB : gcg,gncg'
	print g1_k.T[0],g2_k.T[0]
	print 'CG,S bias', (g1_k/g2_k-1).T[0]
	
	# S13 with REGAUSS weight size
	input_args = cg.Eu_Args(shear_est='S13',sig_w=0.19)
	g1_s,g2_s = cg.calc_cg(input_args)
	print 'With weight sigma = 0.19, S13 : gcg,gncg'
	print g1_s.T[0],g2_s.T[0]
	print 'CG,S bias', (g1_s/g2_s-1).T[0]
	val=(g1_s/g2_s-1).T[0]-(g1_k/g2_k-1).T[0]
	range_print(val,1e-02)

	#interchange bulge and disk labels
	input_args = cg.Eu_Args(shear_est='REGAUSS', bulge_frac=0.75,
		                    bulge_n=1.0, disk_n=1.5,
                            disk_HLR=0.17,bulge_HLR=1.2)
	input_args.bulge_SED_name='Im'
	input_args.disk_SED_name='E'
	g1_r2,g2_r2 = cg.calc_cg(input_args)
	val=(g1_r/g2_r-1).T[0]-(g1_r2/g2_r2-1).T[0]
	range_print(val,1e-07)


	#large weight function
	input_args = cg.Eu_Args(shear_est='KSB',sig_w=3.)
	g1_k2,g2_k2 = cg.calc_cg(input_args)
	print 'Large weight, KSB : gcg,gncg'
	print g1_k2.T[0],g2_k2.T[0]
	print 'CG,S bias', (g1_k2/g2_k2-1).T[0]
	val=(g1_k2/g2_k2-1).T[0]
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	#bias must be small and negative
	if (np.all(val<0.) and np.all(val>-1e03)):
		print colored('PASS', 'green')
	else:
		print colored('FAIL', 'red')
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')




	
        



if __name__ == '__main__':
	cg_unit_tests()
