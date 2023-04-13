import numpy as np
from GGMS.stat_funcs import pcorr, pcorr_to_edge_dict, test_edges_pcorr, test_edges_corr
import networkx as nx

def perform_test_pcorr(X, cov=np.cov, inv=np.linalg.inv):
    covariance = cov(X.T)
    precision = inv(covariance)
    partcorr = pcorr(precision)
    pcorr_edges = pcorr_to_edge_dict(partcorr)
    tests = test_edges_pcorr(pcorr_edges, X.shape[0], X.shape[1])
    n_tests = len(tests)

    return covariance, precision, partcorr, pcorr_edges, tests, n_tests


def perform_test_corr(X):
    corr = np.corrcoef(X.T)
    corr_edges = pcorr_to_edge_dict(corr)
    tests = test_edges_corr(corr_edges, X.shape[0])
    n_tests = len(tests)
    
    return corr, corr, corr, corr_edges, tests, n_tests


class MHT:
    """ABC for MHT solvers"""
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, X, y=None):
        pass

class SimInf(MHT):
    """Simultaneous Inference solver"""
    def __init__(self, alpha=0.05):
        super().__init__(alpha=alpha)

    def fit(self, X, y=None):
        self.covariance, self.precision, self.partcorr, self.pcorr_edges, self.tests, self.n_tests = X
        self.learned_edges = [edge for edge in self.tests if self.tests[edge] < self.alpha]
        dim = self.covariance.shape[0]
        self.graph = nx.empty_graph(dim)
        self.graph.add_edges_from(self.learned_edges)


class Bonferroni(MHT):
    """Bonferroni adjustment solver"""
    def __init__(self, alpha=0.05):
        super().__init__(alpha=alpha)

    
    def fit(self, X, y=None):
        self.covariance, self.precision, self.partcorr, self.pcorr_edges, self.tests, self.n_tests = X
        alpha_corrected = self.alpha / self.n_tests
        self.learned_edges = [edge for edge in self.tests if self.tests[edge] < alpha_corrected]
        dim = self.covariance.shape[0]
        self.graph = nx.empty_graph(dim)
        self.graph.add_edges_from(self.learned_edges)


class Holm(MHT):
    """Holm step-down procedure solver"""
    def __init__(self, alpha=0.05):
        super().__init__(alpha=alpha)

        
    def fit(self, X, y=None):
        self.covariance, self.precision, self.partcorr, self.pcorr_edges, self.tests, self.n_tests = X
        edges_fixed_order = list(self.tests.keys())
        pvalues_fixed_order = [self.tests[edge] for edge in edges_fixed_order]

        permutation = np.argsort(pvalues_fixed_order)

        edges_sorted = [edges_fixed_order[idx] for idx in permutation]
        pvalues_sorted = [pvalues_fixed_order[idx] for idx in permutation]
        
        self.learned_edges = []

        for k in range(1, self.n_tests + 1):
            curve_val = self.alpha / (self.n_tests + 1 - k)
            if pvalues_sorted[k - 1] > curve_val:
                break
            self.learned_edges.append(edges_sorted[k - 1])

        dim = self.covariance.shape[0]
        self.graph = nx.empty_graph(dim)
        self.graph.add_edges_from(self.learned_edges)

class BenjaminiHochberg(MHT):
    """BH solver"""
    def __init__(self, alpha=0.05):
        super().__init__(alpha=alpha)


    def fit(self, X, y=None):
        self.covariance, self.precision, self.partcorr, self.pcorr_edges, self.tests, self.n_tests = X
        edges_fixed_order = list(self.tests.keys())
        pvalues_fixed_order = [self.tests[edge] for edge in edges_fixed_order]

        permutation = np.argsort(pvalues_fixed_order)

        edges_sorted = [edges_fixed_order[idx] for idx in permutation]
        pvalues_sorted = [pvalues_fixed_order[idx] for idx in permutation]
        
        self.learned_edges = []

        for k in range(self.n_tests, 0, -1):
            curve_val = self.alpha * k / self.n_tests
            if pvalues_sorted[k - 1] <= curve_val:
                for i in range(k):
                    self.learned_edges.append(edges_sorted[i])
        
        dim = self.covariance.shape[0]
        self.graph = nx.empty_graph(dim)
        self.graph.add_edges_from(self.learned_edges)

class BenjaminiYekutieli(MHT):
    """BY solver"""
    def __init__(self, alpha=0.05):
        super().__init__(alpha=alpha)

    def fit(self, X, y=None):
        self.covariance, self.precision, self.partcorr, self.pcorr_edges, self.tests, self.n_tests = X
        edges_fixed_order = list(self.tests.keys())
        pvalues_fixed_order = [self.tests[edge] for edge in edges_fixed_order]

        permutation = np.argsort(pvalues_fixed_order)

        edges_sorted = [edges_fixed_order[idx] for idx in permutation]
        pvalues_sorted = [pvalues_fixed_order[idx] for idx in permutation]
        
        self.learned_edges = []

        for k in range(self.n_tests, 0, -1):
            harm = np.log(self.n_tests) + np.euler_gamma + 1 / (2*self.n_tests)
            curve_val = self.alpha * k / (self.n_tests * harm)
            if pvalues_sorted[k - 1] <= curve_val:
                for i in range(k):
                    self.learned_edges.append(edges_sorted[i])

        dim = self.covariance.shape[0]
        self.graph = nx.empty_graph(dim)
        self.graph.add_edges_from(self.learned_edges)


def confusion(pred, true):
    p = pred.edges
    t = true.edges
    full = nx.complete_graph(len(true.nodes)).edges
    
    TP = len(p & t)
    TN = len((full - p) & (full - t))
    FP = len(p & (full - t))
    FN = len((full - p) & t)
    
    return TP, TN, FP, FN