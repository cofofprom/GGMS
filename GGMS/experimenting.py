import numpy as np
from GGMS.solvers import *
from GGMS.spd_generators import *
import pandas as pd
import multiprocessing as mp
from time import perf_counter
import os

def perform_experiments_with_given_model(n_samples, covariance, true_graph, num_experiments, solvers, metrics, test_func):
    dim = covariance.shape[0]
    solver_results = {solver.__class__.__name__: [[] for _ in metrics] for solver in solvers}
    for _ in range(num_experiments):
        samples = np.random.multivariate_normal(np.zeros(dim), covariance, size=n_samples)
        X = test_func(samples)
        for solver in solvers:
            solver.fit(X)
            sname = solver.__class__.__name__
            tp, tn, fp, fn = confusion(solver.graph, true_graph)
            for idx, metric in enumerate(metrics):
                solver_results[sname][idx].append(metric(tp, tn, fp, fn))

    solver_means = {}
    for solver in solver_results:
        solver_means[solver] = np.mean(np.array(solver_results[solver]), axis=1)
    
    return pd.DataFrame(solver_means, index=[metric.__name__ for metric in metrics])


def given_density_experiment(n_samples, N, density_param, S_exp, S_obs, solvers, metrics, model_generator, test_func):
    chol_models = []
    for _ in range(S_exp):
        prec, cov, pc, ed, G = model_generator(N, density_param)
        chol_models.append((prec, cov, pc, ed, G))
    final_exp_result = None
    for model in chol_models:
        prec, cov, pc, ed, G = model
        exp_result = perform_experiments_with_given_model(n_samples, cov, G, S_obs, solvers, metrics, test_func).T

        if final_exp_result is not None:
            final_exp_result += exp_result
        else:
            final_exp_result = exp_result


    final_exp_result /= S_exp

    return final_exp_result

def given_density_experiment_parallel(n_samples, N, density_param, S_exp, S_obs, solvers, metrics, model_generator, test_func):
    chol_models = []
    for _ in range(S_exp):
        prec, cov, pc, ed, G = model_generator(N, density_param)
        chol_models.append((prec, cov, pc, ed, G))
    with mp.Pool() as pool:
        res = [pool.apply_async(perform_experiments_with_given_model, (n_samples, cov, G, S_obs, solvers, metrics, test_func)) for prec, cov, pc, ed, G in chol_models]
        exp_results = sum([r.get().T for r in res])

        final_exp_result = exp_results / S_exp

    return final_exp_result