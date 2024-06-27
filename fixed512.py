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
         per = per/Pmin  # rescaled such that min(per)=1
         sim.add(m=m, P=per, e=e, pomega=pmg, M=l0-pmg, inc=inc, Omega="uniform", primary=sim.particles[0])

     sim.move_to_com()
     return sim

def isStable(sim, N_systems):
    simc = sim.copy() # keep initial simulation
    simc.integrator = "whfast512"
    simc.ri_whfast512.N_systems = N_systems
    simc.dt = 0.023456789
    tmax = 1e7
    times = [0.0 for i in range(N_systems)]
    for k in range(int(tmax/1e4)):
        simc.integrate(simc.t+1e4, exact_finish_time=0)
        for i in range(N_systems):
            for j in range(1, simc.N//N_systems):
                a0 = sim.particles[i*(N_planets+1)+j].orbit(primary=sim.particles[i*(N_planets+1)]).a
                a1 = simc.particles[i*(N_planets+1)+j].orbit(primary=simc.particles[i*(N_planets+1)]).a
                if np.abs((a0-a1)/a0) > 0.1:
                    if times[i] == 0.0:
                        times[i] = simc.t
    return times 
    
def run(params):
    system, sample = params
    mcmc_posterior = h5py.File("NBody_MCMC_Posteriors.hdf5", "r")[system]['DefaultPriors']['PosteriorSample']
    N_planets = mcmc_posterior.shape[1]//5
    N_systems = 8//N_planets
    simcombined = rebound.Simulation()
    for i in range(N_systems):
        sim = hdf5_to_rebound(mcmc_posterior[sample*N_systems+i])
        for p in sim.particles:
            simcombined.add(p)
    s = isStable(simcombined,N_systems)
    for i in range(N_systems):
        with open("output512/"+system+"/%05d.txt"%(sample*N_systems+i), 'w') as f:
            print((sample*N_systems+i), s[i], file=f)
            for j in range(1,N_planets+1):
                o = simcombined.particles[i*(N_planets+1)+j].orbit(primary=simcombined.particles[i*(N_planets+1)])
                print(simcombined.particles[i*(N_planets+1)+j].m, o.a, o.e, file=f)


keys = [k for k in h5py.File("NBody_MCMC_Posteriors.hdf5", "r")]
system = keys[int(sys.argv[1])]
try:
    os.mkdir("output512")
except:
    pass
try:    
    os.mkdir("output512/"+system)
except:
    pass

mcmc_posterior = h5py.File("NBody_MCMC_Posteriors.hdf5", "r")[system]['DefaultPriors']['PosteriorSample']
N_planets = mcmc_posterior.shape[1]//5
N_systems = 8//N_planets
params = [ [system, sample] for sample in np.random.choice(len(mcmc_posterior)//N_systems, replace=False, size=512//N_systems)]

with Pool(80) as pool:
    results = pool.map(run,params)

