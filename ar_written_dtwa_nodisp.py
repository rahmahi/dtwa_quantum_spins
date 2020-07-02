#!/usr/bin/env python
"""
This example scripts computes the hopping matrix of a 2D lattice with
power law (alpha) decay in the hopping amplitude
"""
import numpy as np
from mpi4py import MPI
import dtwa_quantum_spins as dtwa

def run_dtwa():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Parameters
    lattice_size = 9
    alpha = 0.0
    jx, jy, jz = 0.0, 0.0, 1.0
    hx, hy, hz = 0.0, 0.0, 25.0 * np.cos(14.9309177084877)
    niter = 200

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
    p = dtwa.ParamData(hopmat=(jmat+jmat.T),norm=1.0, latsize=size,\
                              jx=jx, jy=jy, jz=jz, hx=hx, hy=hy, hz=hz)

    #Initiate the DTWA system with the parameters and niter
    #d = dtwa.Dtwa_BBGKY_Lindblad_System(p, comm, n_t=niter, seed_offset = 0,\
    #					 decoherence=(0.0, 0.0, 0.0), verbose=True)
  
    d = dtwa.Dtwa_System(p, comm, n_t=niter, verbose=False)
    
    #Prepare the times
    t0 = 0.0
    ncyc = 6.0
    nsteps = 1000

    data = d.evolve((t0, ncyc, nsteps), sampling="spr")

    if rank == 0:
        #Prepare the output files. One for each observable
        append_all = "_time_alpha_" + str(alpha) + "_N_"+str(lattice_size)+\
                                                                "_2ndorder.txt"

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
        np.savetxt(outfile_magy, \
          np.vstack((data.t_output, data.sy.real,data.sy.imag)).T, delimiter=' ')
        np.savetxt(outfile_magz, \
          np.vstack((data.t_output, data.sz.real, data.sz.imag)).T, delimiter=' ')
        np.savetxt(outfile_sxvar, \
          np.vstack((data.t_output, data.sxvar.real, data.sxvar.imag)).T, delimiter=' ')
        np.savetxt(outfile_syvar, \
          np.vstack((data.t_output, data.syvar.real, data.syvar.imag)).T, delimiter=' ')
        np.savetxt(outfile_szvar, \
          np.vstack((data.t_output, data.szvar.real, data.szvar.imag)).T, delimiter=' ')
        np.savetxt(outfile_sxyvar, \
          np.vstack((data.t_output, data.sxyvar.real, data.sxyvar.imag)).T, delimiter=' ')
        np.savetxt(outfile_sxzvar, \
          np.vstack((data.t_output, data.sxzvar.real, data.sxzvar.imag)).T, delimiter=' ')
        np.savetxt(outfile_syzvar, \
          np.vstack((data.t_output, data.syzvar.real, data.syzvar.imag)).T, delimiter=' ')

if __name__ == '__main__':
    run_dtwa()
