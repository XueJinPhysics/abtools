import numpy as np
from pymatgen.io.vasp.inputs import Poscar
import time
#from mpi4py import MPI
import matplotlib.pyplot as plt

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
    
    
    

        
    def sigma_xyy(self, kpts, mu, T):
        # second order response coefficience
        int_element = np.zeros(kpts.shape[0])
        for j in range(kpts.shape[0]):
            kpt = kpts[j]
            fd_dE = self.FermiDirac_dE(kpt=kpt, mu=mu, T=T)
            Gab = self.Gab_calculator(kpt=kpt)
            vk = self.get_vk(kpt=kpt)
            vkn = np.diagonal(vk, axis1=1, axis2=2).real
            int_element[j] = np.sum(vkn[0]*Gab[1,1]*fd_dE - vkn[1]*Gab[0,1]*fd_dE)
            
        area = self.cell["a"] * self.cell["b"]
        sigma_xyy = 1 / area * np.sum(int_element)
        return sigma_xyy

        
if __name__ == "__main__":
    # start time
    start_time = time.time()

    wan90 = WannierMethod(poscar="POSCAR", hr_path="symmed_hr_BxBy=2.dat")
    kpoints = [100, 100, 1]
    
    #mu = 0
    T = 100
    kpts, kweight = wan90.kptsC(kpoints=kpoints)
    
    nef = 201
    mu = np.linspace(-2,2,nef)
    sigma = np.zeros(nef)
    for j in range(nef):
        sigma[j] = wan90.sigma_xyy(kpts=kpts, mu=mu[j], T=T)
        
    np.save("SOAHE.npy", np.array([mu, sigma]))

    




    # end time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序执行时间: {execution_time} 秒")
    