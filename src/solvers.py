import numpy as np
from stat_funcs import pcorr, pcorr_to_edge_dict, test_edges
import networkx as nx
from spd_generators import generate_chol_model

class MHT:
    """ABC for MHT solvers"""
    def __init__(self, alpha=0.05, cov_estimator=np.cov, invertor=np.linalg.inv):
        self.alpha = alpha
        self.cov = cov_estimator
        self.inv = invertor

    def perform_tests(self, X):
        self.covariance = self.cov(X.T)
        self.precision = self.inv(self.covariance)
        self.partcorr = pcorr(self.precision)
        self.pcorr_edges = pcorr_to_edge_dict(self.partcorr)
        self.tests = test_edges(self.pcorr_edges, X.shape[0], X.shape[1])
        self.n_tests = len(self.tests)

    def fit(self, X, y=None):
        pass

class SimInf(MHT):
    """Simultaneous Inference solver"""
    def __init__(self, alpha=0.05, cov_estimator=np.cov, invertor=np.linalg.inv):
        super().__init__(alpha=alpha, cov_estimator=cov_estimator, invertor=invertor)

    def fit(self, X, y=None):
        self.perform_tests(X)

        self.learned_edges = [edge for edge in self.tests if self.tests[edge] < self.alpha]
        self.graph = nx.empty_graph(X.shape[1])
        self.graph.add_edges_from(self.learned_edges)


class Bonferroni(MHT):
    """Bonferroni adjustment solver"""
    def __init__(self, alpha=0.05, cov_estimator=np.cov, invertor=np.linalg.inv):
        super().__init__(alpha=alpha, cov_estimator=cov_estimator, invertor=invertor)
    
    def fit(self, X, y=None):
        self.perform_tests(X)
        alpha_corrected = self.alpha / self.n_tests

        self.learned_edges = [edge for edge in self.tests if self.tests[edge] < alpha_corrected]
        self.graph = nx.empty_graph(X.shape[1])
        self.graph.add_edges_from(self.learned_edges)


class Holm(MHT):
    """Holm step-down procedure solver"""
    def __init__(self, alpha=0.05, cov_estimator=np.cov, invertor=np.linalg.inv):
        super().__init__(alpha=alpha, cov_estimator=cov_estimator, invertor=invertor)
        
    def fit(self, X, y=None):
        self.perform_tests(X)

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

        self.graph = nx.empty_graph(X.shape[1])
        self.graph.add_edges_from(self.learned_edges)

class BenjaminiHochberg(MHT):
    """BH solver"""
    def __init__(self, alpha=0.05, cov_estimator=np.cov, invertor=np.linalg.inv):
        super().__init__(alpha=alpha, cov_estimator=cov_estimator, invertor=invertor)

    def fit(self, X, y=None):
        self.perform_tests(X)

        edges_fixed_order = list(self.tests.keys())
        pvalues_fixed_order = [self.tests[edge] for edge in edges_fixed_order]

        permutation = np.argsort(pvalues_fixed_order)

        edges_sorted = [edges_fixed_order[idx] for idx in permutation]
        pvalues_sorted = [pvalues_fixed_order[idx] for idx in permutation]
        
        self.learned_edges = []

        for k in range(self.n_tests, 0, -1):
            curve_val = self.alpha * k / self.n_tests
            if pvalues_sorted[k - 1] <= curve_val:
                break

        for i in range(k):
            self.learned_edges.append(edges_sorted[i])

        self.graph = nx.empty_graph(X.shape[1])
        self.graph.add_edges_from(self.learned_edges)

class BenjaminiYekutieli(MHT):
    """BY solver"""
    def __init__(self, alpha=0.05, cov_estimator=np.cov, invertor=np.linalg.inv):
        super().__init__(alpha=alpha, cov_estimator=cov_estimator, invertor=invertor)

    def fit(self, X, y=None):
        self.perform_tests(X)

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
                break

        for i in range(k):
            self.learned_edges.append(edges_sorted[i])

        self.graph = nx.empty_graph(X.shape[1])
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

cov, prec, pc, ed, graph = generate_chol_model(5, 0.87, random_state=42)

si = SimInf()
bonf = Bonferroni()
holm = Holm()
bh = BenjaminiHochberg()
by = BenjaminiYekutieli()

samples = np.random.multivariate_normal(np.zeros(5), cov, size=22)
si.fit(samples)
bonf.fit(samples)
holm.fit(samples)
bh.fit(samples)
by.fit(samples)

print(confusion(si.graph, graph))
print(confusion(bonf.graph, graph))
print(confusion(holm.graph, graph))
print(confusion(bh.graph, graph))
print(confusion(by.graph, graph))