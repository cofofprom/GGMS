import numpy as np
from GGMS.solvers import *
import pandas as pd

def perform_experiments_with_given_model(n_samples, covariance, true_graph, num_experiments, solvers, metrics, ranger=range):
    dim = covariance.shape[0]
    solver_results = {solver.__class__.__name__: [[] for _ in metrics] for solver in solvers}
    for _ in ranger(num_experiments):
        samples = np.random.multivariate_normal(np.zeros(dim), covariance, size=n_samples)
        X = perform_test(samples)
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