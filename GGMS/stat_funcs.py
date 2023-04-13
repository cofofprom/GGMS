import numpy as np
from scipy.stats import t
import networkx as nx

def pcorr(precision):
    """
    Calculates partial correlation by given precision matrix
    
    :param precision: precision matrix
    :returns: partial correlation matrix
    """
    D = np.diag(np.sqrt(1 / np.diag(precision)))

    pc = -(D @ precision @ D)
    np.fill_diagonal(pc, 1)

    return pc

def partcorr_test(r, n, k):
    """
    Perform single partial correlation test
    
    :param r: pcorr value
    :param n: num samples
    :param k: num of fixed variables
    :returns: pvalue of test
    """
    dof = n - 2 - k

    stat = r * np.sqrt(dof / (1 - np.power(r, 2)))
    pvalue = t.sf(np.abs(stat), dof)

    return pvalue

def corr_test(r, n):
    dof = n - 2
    
    stat = r * np.sqrt(dof / (1 - np.power(r, 2)))
    pvalue = t.sf(np.abs(stat), dof)
    
    return pvalue


def pcorr_to_edge_dict(pcorr):
    """
    Shrinks upper diagonal of partial correlation matrix in dict of (v, u): p_vu
    
    :param pcorr: partial correlation matrix
    :returns: edge dict (v, u): p_vu
    """
    edge_dict = {}
    for el in np.array(np.triu_indices_from(pcorr, k=1)).T:
        edge = tuple(el)
        edge_dict[edge] = pcorr[edge]

    return edge_dict

def test_edges_corr(edge_dict, n):
    pvalue_edge_dict = {}
    for edge in edge_dict:
        r = edge_dict[edge]
        pvalue = corr_test(r, n)
        pvalue_edge_dict[edge] = pvalue

    return pvalue_edge_dict

def test_edges_pcorr(edge_dict, n, dim):
    """
    Perform multiple pcorr tests for all edges
    
    :param edge_dict: edge dict with pcorr values
    :param n: n samples
    :param dim: dim of partcorr matrix
    :returns: edge dict with pvalues
    """
    pvalue_edge_dict = {}
    for edge in edge_dict:
        r = edge_dict[edge]
        pvalue = partcorr_test(r, n, dim - 2)
        pvalue_edge_dict[edge] = pvalue

    return pvalue_edge_dict

def edge_dict_to_graph(edge_dict, dim, zero_tol=1e-6):
    graph = nx.empty_graph(dim)
    graph.add_edges_from([edge for edge in edge_dict if np.abs(edge_dict[edge] - 0) > zero_tol])

    return graph