from GGMS.spd_generators import *
from GGMS.experimenting import *
from GGMS.metrics import *
from GGMS.solvers import *
import multiprocessing as mp
from time import time, localtime
import pickle

if __name__ == '__main__':  
    N = 20
    n = 100
    S_obs = 100
    S_exp = 500
    density_params = [0.94, 0.87, 0.85, 0.77, 0.74, 0.68, 0.64, 0.55, 0.4] # 0.1 to 0.9
    solvers = [SimInf(), Bonferroni(), Holm(), BenjaminiHochberg(), BenjaminiYekutieli()]
    metrics = [FP, FN, FDR, TPR, F1]

    with mp.Pool() as pool:
        results = [pool.apply_async(given_density_experiment, (n, N, dens, S_exp, S_obs, solvers, metrics)) for dens in density_params]
        start_time = localtime()
        print(f'Started at {start_time.tm_hour}:{start_time.tm_min}')
        start = time()
        density_frames = [result.get() for result in results]
        end = time()

        print('Computations took', end-start, 's')
        print('Mean time for 1 iteration was', (end-start) / (S_obs * S_exp * len(density_params)), 's')
        with open(f'data/{start_time.tm_hour}_{start_time.tm_min}_{N}_{n}.bin', 'wb') as f:
            pickle.dump(density_frames, f)