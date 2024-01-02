import numpy as np
from pymatgen.io.vasp.inputs import Poscar
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
from datetime import datetime

class WannierMethod():
    
    def __init__(self, poscar, hr_path):
        # get crystal structure by pymatgen
        structure = Poscar.from_file(poscar).structure.as_dict()
        self.cell = structure["lattice"]
        self.bcell = 2*np.pi*np.linalg.inv(self.cell["matrix"])
                # read in hamiltonian matrix, in eV
        with open(hr_path, "r") as f:
            lines = f.readlines()
        f.close()
        # get number of wannier functions
        self.num_wan = int(lines[1])    
        # get number of Wigner-Seitz points
        self.num_ws = int(lines[2])
        # get degenereacies of Wigner-Seitz points
        deg_ws=[]
        for j in range(3,len(lines)):
            sp=lines[j].split()
            for s in sp:
                deg_ws.append(int(s))
            if len(deg_ws)==self.num_ws:
                last_j=j
                break
            if len(deg_ws)>self.num_ws:
                raise Exception("Too many degeneracies for WS points!")
        self.deg_ws=np.array(deg_ws,dtype=int)
        # read in the matrix elements
        ham_r = np.zeros([self.num_wan**2*self.num_ws, 7])
        for j in range(last_j+1, len(lines)):
            ham_r[j-last_j-1] = lines[j].split()
        self.ham_r = ham_r
        
        self.Rws = np.dot(self.ham_r[:,0:3].reshape([self.num_ws, self.num_wan, self.num_wan,3])[:,0,0], self.cell["matrix"])
        self.hmn_r = (self.ham_r[:,-2] + 1j*self.ham_r[:,-1]).reshape([self.num_ws, self.num_wan, self.num_wan])
        # units
        self.PlanckConstant = 4.13566733e-15 # [eV s] 
        self.hbar = self.PlanckConstant/(2*np.pi) # [eV s]
        self.kb_eV = 8.617343e-5 # [eV/K]
    
    def k_generator(self, kpoints):
        x=(np.array(range(kpoints[0]))+1/2)/kpoints[0]-1/2
        y=(np.array(range(kpoints[1]))+1/2)/kpoints[1]-1/2
        z=(np.array(range(kpoints[2]))+1/2)/kpoints[2]-1/2
        
        X, Y, Z = np.meshgrid(x, y, z)
        
        # K points mesh and weight with fractional coordinates
        kmesh = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        kweight = 1/np.prod(kpoints)*np.ones(np.prod(kpoints))
        return kmesh, kweight
    
    def kptsC(self, kpoints):
        kmesh, kweight = self.k_generator(kpoints)
        # kpoints in Cartesian coordinates
        return np.dot(kmesh, self.bcell), kweight
    
    def get_hk(self, kpt):
        eikr = np.exp(1j*np.dot(kpt, self.Rws.T)) / self.deg_ws
        hk = np.einsum("R, Rij -> ij", eikr, self.hmn_r, optimize=True)
        
        return hk
        
    def get_vk(self, kpt):
        eikr = np.exp(1j*np.dot(kpt, self.Rws.T)) / self.deg_ws
        Rws = self.Rws.T
        vkx = np.einsum("R, R, Rij -> ij", 1j*Rws[0], eikr, self.hmn_r, optimize=True)
        vky = np.einsum("R, R, Rij -> ij", 1j*Rws[1], eikr, self.hmn_r, optimize=True)
        vkz = np.einsum("R, R, Rij -> ij", 1j*Rws[2], eikr, self.hmn_r, optimize=True)
        
        return np.array([vkx, vky, vkz])
    
    def FermiDirac(self, kpt, mu, T):
        # Fermi Dirac distribution function
        energies = np.linalg.eigvalsh(self.get_hk(kpt=kpt))
        
        if T == 0:
            fd = np.array(energies < mu).astype(float)
        else:
            fd = 1/(np.exp((energies - mu)/(self.kb_eV*T))+1)
        return fd
        
    def FermiDirac_dE(self, kpt, mu, T):
        # derivation of Fermi Dirac distribution function
        energies = np.linalg.eigvalsh(self.get_hk(kpt=kpt))
        
        if T == 0:
            raise Exception("The temperature cannot be 0K.")
        else:
            fd_dE = -np.exp((energies - mu)/(self.kb_eV*T)) / ((self.kb_eV*T)*(np.exp((energies - mu)/(self.kb_eV*T))+1)**2)
        return fd_dE
        
    def Gab_calculator(self, kpt, deltaE=0.001):
        # Berry connection polarizability
        vk = self.get_vk(kpt=kpt)
        energies = np.linalg.eigvalsh(self.get_hk(kpt=kpt))
        invE = 1 / (energies[:, np.newaxis] - energies[np.newaxis, :] + 1j*deltaE)
        invE = invE - np.diag(np.diag(invE))
        #for i in range(invE.shape[0]):
            #np.fill_diagonal(invE[i, :, :], 0)
        invE3 = invE**3
        
        Gab = 2 * np.einsum("anm, bmn, nm -> abn", vk, vk, invE3, optimize=True).real
        return Gab
    
    
    def sigma_xyy(self, kpts, mu, T, kpart_size=100000, deltaE=0.001):
        # second order response coefficience
        # Parameters:
        # kpart_size: Size of split-k. As long as the memory is large enough, you can increase it indefinitely
        # MPI setting
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # k parts
        kpts_split = np.split(kpts, np.arange(0, kpts.shape[0], kpart_size)[1:])
        

        int_elements = np.zeros(len(kpts_split))

        for j in range(rank, len(kpts_split), size):
            ks = kpts_split[j]
            # Energies
            eikr = np.exp(1j*np.dot(ks, self.Rws.T)) / self.deg_ws
            hk = np.einsum("kR, Rij -> kij", eikr, self.hmn_r, optimize=True)
            energies = np.linalg.eigvalsh(hk)
            
            # 1-order deviration of Fermi-Dirac distribution
            fd_dE = -np.exp((energies - mu)/(self.kb_eV*T)) / ((self.kb_eV*T)*(np.exp((energies - mu)/(self.kb_eV*T))+1)**2)
        
            # velocity matrix
            vkx = np.einsum("R, kR, Rij -> kij", 1j*self.Rws[:,0], eikr, self.hmn_r, optimize=True)
            vky = np.einsum("R, kR, Rij -> kij", 1j*self.Rws[:,1], eikr, self.hmn_r, optimize=True)
            
            #BCP
            invE = 1 / (energies[:, :, np.newaxis] - energies[:, np.newaxis, :] + 1j*deltaE)
            for i in range(invE.shape[0]):
                np.fill_diagonal(invE[i, :, :], 0)
            invE3 = invE**3
            
            Gxy = 2 * np.einsum("knm, kmn, knm -> kn", vkx, vky, invE3, optimize=True).real
            Gyy = 2 * np.einsum("knm, kmn, knm -> kn", vky, vky, invE3, optimize=True).real
            
            int_elements[j] = np.sum((vkx.diagonal(axis1=1, axis2=2).real * Gyy- vky.diagonal(axis1=1, axis2=2).real * Gxy) * fd_dE)

        
        global_elements = np.zeros_like(int_elements)
        comm.Allreduce(int_elements, global_elements, op=MPI.SUM)
        
        if rank == 0:
            area = self.cell["a"] * self.cell["b"]
            sigma_xyy = 1 / area * np.sum(global_elements)
            return sigma_xyy

        
