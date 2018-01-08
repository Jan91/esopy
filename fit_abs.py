#! usr/bin/python

import scipy, pymc
import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *

from scipy.special import wofz

e = 4.8032E-10  # cm 3/2 g 1/2 s-1
c = 2.998e10 # speed of light cm/s
m_e = 9.10938291e-28 # electron mass g

l0_CrII2056 = 2056.2568
f_CrII2056 = 0.103 #oscillator strength
gamma_CrII2056 = 0.133196217632 # damping constant in km/s


# Load the data from the data.csv file using pandas
# -> save the velocity, the flux, and the flux error in three individual lists
# (i.e. as a pandas.Series: One-dimensional ndarray with axis labels)
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

#velocity = ?
#flux = ?
#flux_err = ? 


#def voigt(x, sigma, gamma):
#	'''
#	1D voigt profile, e.g.:
#	https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
#	gamma: half-width at half-maximum (HWHM) of the Lorentzian profile
#	sigma: the standard deviation of the Gaussian profile
#	HWHM, alpha, of the Gaussian is: alpha = sigma * sqrt(2ln(2))
#	'''
#	
#	z = (x + 1j*gamma) / (sigma * np.sqrt(2.0))
#	V = wofz(z).real / (sigma * np.sqrt(2.0*np.pi))
#	return V


# finish defining the add_abs_velo() function

#def add_abs_velo(v, N, b, gamma, f, l0):
#	'''
#	Add absorption line l0 in velocity space v, given the oscillator strength,
#	the damping constant gamma, column density N, and broadening b
#	'''
#
#
#
#	
#	return np.exp(-tau)



# finish defining model() function
# add the other 3 priors (N, b, BG)
# finish defining the data (y_val = pymc.Normal())


#def model(velocity, flux, flux_err):
#
#	v0 = pymc.Uniform('v0',lower=-400,upper=400, doc='v0')
#
#	@pymc.deterministic(plot=False) #Deterministic Decorator
#	def add_voigt(velocity=velocity,N=N,b=b,v0=v0, BG=BG):
#
#		f = np.ones(len(velocity))  
#		return f
#
#	#Data with Gaussian errors, for Likelihood
#	y_val = pymc.Normal()
#
#	return locals()


# finish defining the mcmc() function

#def mcmc(velocity, flux, flux_err):
#
#	MDL = pymc.MCMC(...,db='pickle',dbname='results.pickle')
#
#	MDL.db
#	MDL.sample()
#	MDL.db.close()
#
#	y_fit = MDL.stats()['add_voigt']['mean']
#
#	return



# try to plot the results






