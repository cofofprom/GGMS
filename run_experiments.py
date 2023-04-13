from GGMS.experimenting import *
from GGMS.metrics import *
import logging
import argparse
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('N', action='store', type=int)
    parser.add_argument('n', action='store', type=int)
    parser.add_argument('S_exp', action='store', type=int)
    parser.add_argument('S_obs', action='store', type=int)
    parser.add_argument('model', action='store', type=str)
    parser.add_argument('similarity', action='store', type=str)
    parser.add_argument('-o', action='store', type=str, default='latest.bin')
    logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    
    args = parser.parse_args()
    
    N = args.N
    n = args.n
    S_exp = args.S_exp
    S_obs = args.S_obs
    
    if args.model == 'chol':
        densities = [0.94, 0.87, 0.85, 0.77, 0.74, 0.68, 0.64, 0.55, 0.4]
        model_generator = generate_chol_model
    elif args.model == 'peng':
        densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        model_generator = generate_peng_model
        
    if args.similarity == 'corr':
        tester = perform_test_corr
    else:
        tester = perform_test_pcorr
        
    solvers = [SimInf(), Bonferroni(), Holm(), BenjaminiHochberg(), BenjaminiYekutieli()]
    metrics = [FP, FN, FDR, TPR, F1]
        
    logging.info(str([N, n, S_exp, S_obs, densities, model_generator]))
    
    density_frames = []
    
    logging.info('Started working')
    start = perf_counter()
    for dens in densities:
        frame = given_density_experiment_parallel(n, N, dens, S_exp, S_obs, solvers, metrics, model_generator, tester)
        density_frames.append(frame)
        logging.info('%s computed', dens)
        
    end = perf_counter()
    
    logging.info('Computations took %s s', end - start)
        
    with open(f'data/{args.o}', 'wb') as f:
        pickle.dump(density_frames, f)
        
    x = np.arange(0.1, 1, 0.1)

    for idx, metric in enumerate(density_frames[0].columns):
        density_df = pd.concat([density_frame[metric] for density_frame in density_frames], axis=1).T

        for method in density_frames[0].index:
            plt.plot(x, density_df[method], label=method)
        plt.legend()
        plt.title(metric)
        plt.xlabel('Density')
        plt.xlim(0.1, 0.9)
        if idx > 1:
            plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(f'data/{metric}_{args.model}_{N}_{n}_{args.similarity}.png')
        plt.clf()
    