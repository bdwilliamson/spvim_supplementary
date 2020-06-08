# influence functions for shapley values

def shapley_influence_function(Z, z_counts, W, v, psi, G, c_n, ics, measure):
    """
    Compute influence function for the given predictiveness measure

    @param Z the subsets of the power set with estimates
    @param W the matrix of weights
    @param v the estimated predictivness
    @param psi the estimated Shapley values
    @param G the constrained ls matrix
    @param c_n the constraints
    @param ics a list of all ics
    @param measure the predictiveness measure
    """
    import numpy as np

    ## compute contribution from estimating V
    Z_W = Z.transpose().dot(W)
    A_m = Z_W.dot(Z)
    A_m_inv = np.linalg.pinv(A_m)
    phi_01 = A_m_inv.dot(Z_W).dot(ics)

    ## compute contribution from estimating Q
    qr_decomp = np.linalg.qr(G.transpose(), mode = 'complete')
    U_2 = qr_decomp[0][:, 3:(Z.shape[1])]
    V = U_2.transpose().dot(Z.transpose().dot(W).dot(Z)).dot(U_2)
    phi_02_shared_mat = (-1) * U_2.dot(np.linalg.pinv(V))
    phi_02_uniq_vectors = np.array([(Z[z, :].dot(psi) - v[z]) * (U_2.transpose().dot(Z[z, :])) for z in range(Z.shape[0])]).transpose()
    phi_02_uniq = phi_02_shared_mat.dot(phi_02_uniq_vectors)
    phi_02 = np.repeat(phi_02_uniq, z_counts, axis=1)

    return {'contrib_v': phi_01, 'contrib_s': phi_02}
