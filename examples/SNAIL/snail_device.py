import numpy as np
from snail_hamiltonion import get_snail_couplings_at_zero_kerr, Rq

eta = 500e6 # Linear displacement driving strength assuming to be 500MHZ
DISPLACEMNT_PULSE_LENGTH = 10e-9 # 10ns
WAITING_TIME_FOR_KERR_DRIFT = 10e-9



def get_snail_parameters(f_infty=6.99,n=3):
    g3dc, g4dc, g5dc, g6dc, g3ac ,w0 = get_snail_couplings_at_zero_kerr(
    Ej0=245*2*np.pi, dl_Lj0=2.4133, f_infty=f_infty*1e9, Z=57.9/Rq, alpha=0.0971, n=n
    )
    dc = {3:g3dc, 4:g4dc, 5:g5dc, 6:g6dc}
    OMEGA_by_2pi = w0
    Kerr = 12*(dc[4] - 5*dc[3]**2/OMEGA_by_2pi) 
    print(f'w/2pi = {w0/1e9/2/np.pi:.3f} GHz')
    print(f'Kerr/2pi = {Kerr/1e6/2/np.pi:.3f} MHz ')
    print(f'g3dc/2pi = {g3dc/1e6/2/np.pi:.3f} MHz')
    print(f'g4dc/2pi = {g4dc/1e6/2/np.pi:.3f} MHz')
    print(f'g5dc/2pi = {g5dc/1e6/2/np.pi:.3f} MHz')
    print(f'g6dc/2pi = {g6dc/1e6/2/np.pi:.3f} MHz')
    print(f'g3ac/2pi = {g3ac/1e6/2/np.pi:.3f} MHz')
    return dc, g3ac, OMEGA_by_2pi


print('============N=3============')
get_snail_parameters(f_infty=6.95,n=3)
print('============N=4============')
get_snail_parameters(f_infty=6.90,n=4)
print('============N=5============')
get_snail_parameters(f_infty=6.95,n=5)