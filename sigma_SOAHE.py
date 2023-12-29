import numpy as np
from pymatgen.io.vasp.inputs import Poscar
import time
from mpi4py import MPI

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
    
    def FermiDirac(self, energies, mu, T):
        # Fermi Dirac distribution function
        if T == 0:
            fd = np.array(energies < mu).astype(float)
        else:
            fd = 1/(np.exp((energies - mu)/(self.kb_eV*T))+1)
        return fd
        
    def FermiDirac_grad(self, energies, mu, T):
        # derivation of Fermi Dirac distribution function
        if T == 0:
            raise Exception("The temperature cannot be 0K.")
        else:
            fd_grad = -np.exp((energies - mu)/(self.kb_eV*T)) / ((self.kb_eV*T)*(np.exp((energies - mu)/(self.kb_eV*T))+1)**2)
        return fd_grad
        
    def Gab_calculator(self, vk, deltaE=0.001):
        # Berry connection polarizability
        invE = 1 / (energies[:, np.newaxis] - energies[np.newaxis, :] + 1j*deltaE)
        invE = invE - np.diag(np.diag(invE))
        #for i in range(invE.shape[0]):
            #np.fill_diagonal(invE[i, :, :], 0)
        invE3 = invE**3
        
        Gab = 2 * np.einsum("anm, bmn, nm -> abn", vk, vk, invE3, optimize=True).real
        
        return Gab
        
    def sigma_xyy(self, vk, Gab, fd_grad):
        # second order response coefficience 
        int_element = 2 * (vk[0]*Gab[1,1]*fd_grad - vk[1]*Gab[0,1]*fd_grad)
        area = self.cell["a"] * self.cell["b"]
        sigma_xyy = 1 / area * np.sum
        
        
        
        
if __name__ == "__main__":
    # start time
    start_time = time.time()

    wan90 = WannierMethod(poscar="POSCAR", hr_path="symmed_hr_BxBy=2.dat")
    kpoints = [500, 500, 1]
    
    mu = 0
    T = 100
    kpts, kweight = wan90.kptsC(kpoints=kpoints)
    for kpt in kpts:
        hk = wan90.get_hk(kpt)
        energies, vectors = np.linalg.eigh(hk)
        vk = wan90.get_vk(kpt=kpt)
        fd = wan90.FermiDirac(energies=energies, mu=mu, T=T)
        fd_grad = wan90.FermiDirac_grad(energies=energies, mu=mu, T=T)
        Gab = wan90.Gab_calculator(vk=vk)

    ef_list = np.linspace(-2,2,101)




    # end time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序执行时间: {execution_time} 秒")
    