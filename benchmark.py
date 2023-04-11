from GGMS.spd_generators import *
from GGMS.experimenting import *
from GGMS.metrics import *
from GGMS.solvers import *
import multiprocessing as mp
from time import time, localtime
import pickle
import argparse
import logging

if __name__ == '__main__':
    logging.basicConfig(filename='logfile.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    logging.info("Computations started")
    parser = argparse.ArgumentParser()
    parser.add_argument('N', action='store', type=int)
    parser.add_argument('n', action='store', type=int)
    parser.add_argument('S_obs', action='store', type=int)
    parser.add_argument('S_exp', action='store', type=int)
    parser.add_argument('--chol_density', action='store_true')
    parser.add_argument('--output_filename', action='store', type=str, default='latest')
    args = parser.parse_args()
    N = args.N
    n = args.n
    S_obs = args.S_obs
    S_exp = args.S_exp
    density_params = [0.94, 0.87, 0.85, 0.77, 0.74, 0.68, 0.64, 0.55, 0.4] if args.chol_density else np.arange(0.1, 1, step=0.1)# 0.1 to 0.9
    solvers = [SimInf(), Bonferroni(), Holm(), BenjaminiHochberg(), BenjaminiYekutieli()]
    metrics = [FP, FN, FDR, TPR, F1]

    model_gen = generate_chol_model if args.chol_density else generate_peng_model
    
    logging.info(f'Working with N={N}, n={n}, S_obs={S_obs}, S_exp={S_exp}, densities={density_params}, func={model_gen} ofname={args.output_filename}')
    
    with mp.Pool() as pool:
        results = [pool.apply_async(given_density_experiment, (n, N, dens, S_exp,
                                                               S_obs, solvers, metrics,
                                                               model_gen)) for dens in density_params]
        start_time = localtime()
        logging.info(f'Started at {start_time.tm_hour}:{start_time.tm_min}')
        start = time()
        density_frames = [result.get() for result in results]
        end = time()

        logging.info(f'Computations took {end-start}s')
        logging.info(f'Mean time for 1 iteration was {(end-start) / (S_obs * S_exp * len(density_params))}s')
        with open(f'data/{args.output_filename}.bin', 'wb') as f:
            pickle.dump(density_frames, f)