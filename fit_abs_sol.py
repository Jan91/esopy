#! usrt/bin/python

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

l0_CrII2056 = 2056.
f_CrII2056 = 0.103 #oscillator strength
gamma_CrII2056 = 0.133196217632 # damping constant in km/s


# Load the data fro the data.csv file
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.from_csv.html
df = pd.DataFrame.from_csv("data.csv", header=0, sep=', ', index_col=None)

def voigt(x, sigma, gamma):
	'''
	1D voigt profile, e.g.:
	https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
	gamma: half-width at half-maximum (HWHM) of the Lorentzian profile
	sigma: the standard deviation of the Gaussian profile
	HWHM, alpha, of the Gaussian is: alpha = sigma * sqrt(2ln(2))
	'''
	
	z = (x + 1j*gamma) / (sigma * np.sqrt(2.0))
	V = wofz(z).real / (sigma * np.sqrt(2.0*np.pi))
	return V


def add_abs_velo(v, N, b, gamma, f, l0):
	'''
	Add absorption line l0 in velocity space v, given the oscillator strength,
	the damping constant gamma, column density N, and broadening b
	'''
	A = (((np.pi*e**2)/(m_e*c))*f*l0*1E-13) * (10**N)
	tau = A * voigt(v,b/np.sqrt(2.0),gamma)

	return np.exp(-tau)


def model(velocity, flux, flux_err):

	v0 = pymc.Uniform('v0',lower=-400,upper=400, doc='v0')
	N  = pymc.Normal('N',mu=15.0,tau=1.0/(10**2), doc='N')
	b  = pymc.Normal('b',mu=15.0,tau=1.0/(10**2), doc='b')
	BG = pymc.Normal('BG',mu=1.0,tau=1./(0.05*2), doc='BG')

	@pymc.deterministic(plot=False) #Deterministic Decorator
	def add_voigt(velocity=velocity,N=N,b=b,v0=v0, BG=BG):

		f = np.ones(len(velocity))*BG #Background, Continuum
		v = velocity - v0
		f *= add_abs_velo(v, N, b, gamma_CrII2056, f_CrII2056, l0_CrII2056)       
		return f

	#Data with Gaussian errors, for Likelihood
	y_val = pymc.Normal('y_val',mu=add_voigt,tau=1/(flux_err**2),value=flux,observed=True)

	return locals()


def mcmc(velocity, flux, flux_err):

	MDL = pymc.MCMC(model(velocity,flux,flux_err),
		db='pickle',dbname='results.pickle')

	MDL.db
	MDL.sample(20000, 0)
	MDL.db.close()

	y_min = MDL.stats()['add_voigt']['quantiles'][2.5]
	y_max = MDL.stats()['add_voigt']['quantiles'][97.5]
	y_fit = MDL.stats()['add_voigt']['mean']

	return y_fit, y_min, y_max




y_fit, y_min, y_max, = mcmc(df['velocity'], df['flux'], df['flux_err'])


errorbar(df['velocity'], df['flux'], yerr=df['flux_err'], fmt="o")
plot(df['velocity'], y_fit)
show()

results = pd.read_pickle('results.pickle')
df = pd.DataFrame(results)

f, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

x = np.arange(0, len(df['N'][0]), 1)

axes[0].plot(x, df['BG'][0])
axes[1].plot(x, df['N'][0])
axes[2].plot(x, df['b'][0])
axes[3].plot(x, df['v0'][0])

axes[0].set_ylabel('BG')
axes[1].set_ylabel('N')
axes[2].set_ylabel('b')
axes[3].set_ylabel('v0')	
axes[3].set_xlabel("Iterations")

show()







