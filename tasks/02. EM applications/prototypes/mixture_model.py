import numpy as np
from numpy.linalg import slogdet, det, solve

class MixtureModel:
    def __init__(self, n_components, diag=False):
        """
        Parametrs:
        ---------------
        n_components: int
        The number of components in mixture model

        diag: bool
            If diag is True, covariance matrix is diagonal
        """
        self.n_components = n_components  
        # bonus part
        self.diag = diag
        
    def _E_step(self, data):
        """
        E-step of the algorithm
        
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
            Array of data points. Each row corresponds to a single data point.
        """    
        # set self.q_z
        if self.diag:
            # bonus part
            pass
        else:
            pass
                
    def _M_step(self, data):
        """
        M-step of the algorithm
        
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
            Array of data points. Each row corresponds to a single data point.
        """
        N, d = data.shape
        
        # set self.w, self.Mean, self.Sigma
        if self.diag:
            # bonus part
            pass
        else:
            pass
    
    def EM_fit(self, data, max_iter=10, tol=1e-3,
               w_init=None, m_init=None, s_init=None, trace=False):
        """
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.

        max_iter: int
        Maximum number of EM iterations

        tol: int
        The convergence threshold

        w_init: numpy array shape(n_components)
        Array of the each mixture component initial weight

        Mean_init: numpy array shape(n_components, n_features)
        Array of the each mixture component initial mean

        Sigma_init: numpy array shape(n_components, n_features, n_features)
        Array of the each mixture component initial covariance matrix
        
        trace: bool
        If True then return list of likelihoods
        """
        # parametrs initialization
        N, d = data.shape
        self.q_z = np.zeros((N, self.n_components))
        self.tol = tol
        
        # other initialization
        if w_init is None:
            pass
        else:
            self.w = w_init

        if m_init is None:
            pass
        else:
            self.Mean = m_init

        if s_init is None:
            pass
        else:
            self.Sigma = s_init
        
        log_likelihood_list = []
        
        # algo    
        for i in range(max_iter):
            # Perform E-step 
            pass
            # Compute loglikelihood
            pass
            # Perform M-step
            pass
        
        # Perform E-step
        # Compute loglikelihood
        
        if trace:
            return self.w, self.Mean, self.Sigma, log_likelihood_list
        else:
            return self.w, self.Mean, self.Sigma
    
    def EM_with_different_initials(self, data, n_starts, max_iter=10, tol=1e-3):
        """
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.

        n_starts: int
        The number of algorithm running with different initials

        max_iter: int
        Maximum number of EM iterations

        tol: int
        The convergence threshold

        Returns:
        --------
        Best values for w, Mean, Sigma parameters
        """
        best_w, best_Mean, best_Sigma, max_log_likelihood = None, None, None, -np.inf
        for i in range(max_iter):
            pass
        
        self.w = best_w
        self.Mean = best_Mean
        self.Sigma = best_Sigma
        
        return w, Mean, Sigma
    
    def compute_log_likelihood(self, data):
        """
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.
        """
        
        return log_likelihood