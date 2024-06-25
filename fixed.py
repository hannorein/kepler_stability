import rebound
import numpy as np
import h5py
import os
import sys
from multiprocess import Pool


def hdf5_to_rebound(row):
     row = row.reshape((-1,5))
     Pmin = np.min(row[:,1])
     TCmin = np.min(row[:,4])
     epoch = TCmin - 0.1 * Pmin
    
     sim = rebound.Simulation()
     sim.add(m=1)
    
     for m, per, h, k, Tc in row:
         e, pmg = np.sqrt(h*h + k*k), np.arctan2(k,h)
         l0 = np.mod(2.0*np.pi*(epoch - Tc)/per, 2*np.pi) + 0.5*np.pi
         inc = 1e-3*np.random.normal() # small random inclination
         per = per/365.2569 * np.pi*2  # days to year/2pi
         sim.add(m=m, P=per, e=e, pomega=pmg, M=l0-pmg, inc=inc, Omega="uniform", primary=sim.particles[0])

     sim.move_to_com()
     return sim

def isStable(sim):
    simc = sim.copy() # keep initial simulation
    simc.integrator = "whfast"
    Pmin = min([p.P for p in simc.particles[1:]])
    simc.dt = Pmin * 0.023456789
    tmax = 1e7
    for k in range(int(tmax/1e4)):
        simc.integrate(simc.t+1e4 * Pmin, exact_finish_time=0)
        for i in range(1, simc.N):
            a0 = sim.particles[i].a
            a1 = simc.particles[i].a
            if np.abs((a0-a1)/a0) > 0.1:
                return simc.t/Pmin
    return 0.0 # stable
    
def run(params):
    system, sample = params
    mcmc_posterior = h5py.File("NBody_MCMC_Posteriors.hdf5", "r")[system]['DefaultPriors']['PosteriorSample']
    with open("output/"+system+"/%05d.txt"%sample, 'w') as f:
        sim = hdf5_to_rebound(mcmc_posterior[sample])
        s = isStable(sim)
        print(sample, s, file=f)
        for i in range(1,sim.N):
            print(sim.particles[i].m, sim.particles[i].a, sim.particles[i].e, file=f)


system = sys.argv[1] 
try:
    os.mkdir("output")
except:
    pass
try:    
    os.mkdir("output/"+system)
except:
    pass

mcmc_posterior = h5py.File("NBody_MCMC_Posteriors.hdf5", "r")[system]['DefaultPriors']['PosteriorSample']
params = [ [system, sample] for sample in np.random.choice(len(mcmc_posterior), replace=False, size=500)]

with Pool() as pool:
    results = pool.map(run,params)

