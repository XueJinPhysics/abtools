from sigma_SOAHE import WannierMethod
import time, datetime
import numpy as np


    # start time
start_time = time.time()
current_time = datetime.now()
print("开始时间:", current_time)


wan90 = WannierMethod(poscar="POSCAR", hr_path="symmed_hr_BxBy=2.dat")
kpoints = [100, 100, 1]
    
#mu = 0
T = 100
kpts, kweight = wan90.kptsC(kpoints=kpoints)
    

E_FERMI = -1.7310
    
MuNum =  201      
MuMin = E_FERMI - 0.18     
MuMax = E_FERMI - 0.14     
    
mu = np.linspace(MuMin, MuMax, MuNum)
sigma_xyy = np.zeros(MuNum)
for j in range(MuNum):
    sigma_xyy[j] = wan90.sigma_xyy(kpts=kpts, mu=mu[j], T=T)
        
np.save("SOAHE.npy", np.array([mu, sigma_xyy]))



    # end time
end_time = time.time()
execution_time = end_time - start_time
print(f"程序执行时间: {execution_time} 秒")
current_time = datetime.now()
print("结束时间:", current_time)
