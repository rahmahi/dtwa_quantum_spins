#!/usr/bin/env python
"""
This example scripts computes the hopping matrix of a 2D lattice with
power law (alpha) decay in the hopping amplitude
"""
import numpy as np
from mpi4py import MPI
import dtwa_quantum_spins as dtwa
import time as tm
import matplotlib.pyplot as plt

def run_dtwa():
    start = tm.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Parameters
    niter = 500
    lattice_size = 8
    alpha = 0.0
    jx, jy, jz = -1.0, 0.0, 0.0
    hx, hy, hz = 0.0, 0.0, -1.0
    amp = 25.0
    f = 0.0
    hdc = 0.1

    #seed(s)
    #Build the hopping matrix
    size = lattice_size
    jmat = np.zeros((size, size))
    for mu in xrange(size):
        for nu in xrange(mu, size):
            if mu != nu:
                dmn = np.abs(mu-nu)
                jmat[mu,nu] = 1.0/pow(dmn,alpha)

    #Initiate the parameters in object
    mid = np.floor(size/2).astype(int)
    kacnorm =2.0 * np.sum(1/(pow(np.arange(1, mid+1), alpha).astype(float)))
    #kacnorm = 1.0
    p = dtwa.ParamData(hopmat=(jmat+jmat.T),norm = kacnorm, latsize=size,\
                              jx=jx, jy=jy, jz=jz, hx=hx, hy=hy, hz=hz, omega=f, hdc=hdc, amp=amp)

    #Initiate the DTWA system with the parameters and niter
    #d = dtwa.Dtwa_BBGKY_Lindblad_System(p, comm, n_t=niter, seed_offset = 0,\
    #					 decoherence=(0.0, 0.0, 0.0), verbose=True)
  
    d = dtwa.Dtwa_System(p, comm, n_t=niter, verbose=False)
    #Prepare the times
    t0 = 0.0
    ncyc = 100.0
    nsteps = 3001

    data = d.evolve((t0, ncyc, nsteps), sampling="spr")

    if rank == 0:
        #Prepare the output files. One for each observable
        append_all = "_n"+str(lattice_size)+"_hdc_"+str(hdc)+"_hz"+str(hz)+"_f_"+str(f)+"_.txt"

        outfile_magx = "sx" + append_all
        outfile_magy = "sy" + append_all
        outfile_magz = "sz" + append_all

        outfile_sxvar = "sxvar" + append_all
        outfile_syvar = "syvar" + append_all
        outfile_szvar = "szvar" + append_all

        outfile_sxyvar = "sxyvar" + append_all
        outfile_sxzvar = "sxzvar" + append_all
        outfile_syzvar = "syzvar" + append_all

        outfile_magx = "sx" + append_all
        outfile_magy = "sy" + append_all
        outfile_magz = "sz" + append_all

        outfile_sxvar = "sxvar" + append_all
        outfile_syvar = "syvar" + append_all
        outfile_szvar = "szvar" + append_all

        outfile_sxyvar = "sxyvar" + append_all
        outfile_sxzvar = "sxzvar" + append_all
        outfile_syzvar = "syzvar" + append_all

        #Dump each observable to a separate file
        np.savetxt(outfile_magx, \
          np.vstack((data.t_output, data.sx.real,data.sx.imag)).T, delimiter=' ')
        np.savetxt(outfile_magz, \
          np.vstack((data.t_output, data.sz.real,data.sz.imag)).T, delimiter=' ')
        np.savetxt(outfile_sxvar, \
          np.vstack((data.t_output, data.sxvar.real, data.sxvar.imag)).T, delimiter=' ')
        np.savetxt(outfile_sxyvar, \
          np.vstack((data.t_output, data.sxyvar.real, data.sxyvar.imag)).T, delimiter=' ')
        np.savetxt(outfile_sxzvar, \
          np.vstack((data.t_output, data.sxzvar.real, data.sxzvar.imag)).T, delimiter=' ')
        np.savetxt(outfile_syzvar, \
          np.vstack((data.t_output, data.syzvar.real, data.syzvar.imag)).T, delimiter=' ')
        
        duration = tm.time() - start
	print "time taken = ",duration
        #plt.title("l8_hz-1_hdc_0p1_amp_25_dtwa")
        plt.plot(data.t_output, data.sz.real, label = "sz dtwa")
        plt.plot(data.t_output, data.sx.real, label = "sx dtwa")
 	plt.xlabel("time")
        plt.ylim(-1.1,1.1)
	plt.grid()
	plt.legend()
	plt.show()

if __name__ == '__main__':
    run_dtwa()
