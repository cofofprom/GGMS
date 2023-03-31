import numpy as np
from sklearn.datasets import make_sparse_spd_matrix
import json
import networkx as nx
from stat_funcs import pcorr, pcorr_to_edge_dict

def write_chol_calibration(params, dim):
   obj = params

   with open(f'chol_calibration_values\\{dim}.json', 'w') as f:
    json.dump(obj, f)


def read_chol_calibration(dim):
    with open(f'chol_calibration_values\\{dim}.json', 'r') as f:
        obj = json.load(f)

    return obj

def calibrate_chol(dim):
    # TODO
    pass

def generate_chol_model(dim, param, zero_tol=1e-6, random_state=None, invertor=np.linalg.inv):
    """
    Generates graphical model with make_sparse_spd_matrix generator

    :param dim: dimension of precision matrix
    :param param: param alpha of make sparse spd
    :param zero_tol: tolerance of being zero
    :param random_state: rs
    :param invertor: inverse function

    :returns: (precision, covariance, partcorr, edge_dict, graph)
    """
    precision = make_sparse_spd_matrix(dim, alpha=param, norm_diag=True, random_state=random_state)
    covariance = invertor(precision)
    partcorr = pcorr(precision)
    edge_dict = pcorr_to_edge_dict(partcorr)
    graph = nx.empty_graph(dim)
    graph.add_edges_from([edge for edge in edge_dict if edge_dict[edge] - 0 > zero_tol])

    return precision, covariance, partcorr, edge_dict, graph