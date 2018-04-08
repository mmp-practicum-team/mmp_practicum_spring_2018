from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix

class LFM:    
    def __init__(self, n_components, lamb=1e-2, mu=1e-2, max_iter=10, tol=1e-4, verbose=False):
        """
        Parameters:
        -----------
            n_components : float, number of components in Latent Factor Model
            
            lamb : float, l2-regularization coef for users profiles
            
            mu : float, l2-regularization coef for items profiles
            
            max_iter: int, maximum number of iterations
            
            tol: float, tolerance of the algorithm
            (if \sum_u \sum_d p_{ud}^2 + \sum_i \sum_d q_{id}^2 < tol then break)
            
            verbose: bool, if true then print additional information during the optimization
        """
        self.n_components = n_components
        self.lamb = lamb
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
    def fit(self, X, P_init=None, Q_init=None, validation_triplets=None):
        """
        Fitting of Latent Factor Model using ALS method
        
        Parameters:
        -----------
            X : sparse matrix, users-items matrix
        """
        if P_init:
            P = P_init
        else:
            P = #init
            
        if Q_init:
            Q = Q_init
        else:
            Q = #init

        # need for faster optimization        
        XT = csr_matrix(X.T)
        
        
        
        for iteration in range(self.max_iter):
            norm_p = 0
            norm_q = 0
            
            # fix Q, recalculate P
            
            # fix P, recalculate Q
                
            if norm_p + norm_q <= self.tol:
                break
            
    
    def predict_for_pair(self, user, item):
        """
        Get the prediction
        
        Parameters:
        -----------
            user : non-negative int, user index
            
            item : non-negative int, item index
        """
        pass