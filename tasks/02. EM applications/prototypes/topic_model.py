import numpy as np
import scipy.sparse as sparse

class TopicModel:
    def __init__(self,
                 num_topics,
                 max_iter=30,
                 batch_size=100,
                 regularizers=tuple(),
                 modalities_coefs=(1.0, )):
        """
        Parameters:
        ---------------
        num_topics : int
            The number of topics in the algorithm
        
        max_iter: int
            Maximum number of EM iterations

        batch_size : int
            The number of objects in one batch
        
        regularizers : tuple of BaseRegularizer subclasses
            The tuple of model regularizers
                
        modalities_coefs : tuple of float
            The tuple of modalities coefs. Each coef corresponds to an element of list of data
        """
        self.num_topics = num_topics
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.regularizers = list(regularizers)
        self.modalities_coefs = modalities_coefs
    
    def _EM_step_for_batch(self, data_batch, Theta_part):
        """
        Iteration of the algorithm for one batch.
        It should include implementation of the E-step and M-step for the Theta matrix.
        
        Parametrs:
        ----------
        data: sparse array shape (n_documents, n_words) or
            list of sparse array shape (n_documents, n_modality_words) in modalities case
            Array of data points. Each row corresponds to a single document.
        
        Theta_part : numpy array (n_topics, batch_size)
            Part of Theta matrix
        
        Returns:
        --------
        Theta_part : numpy array (n_topics, batch_size)
            Part of Theta matrix (after M-step)
        
        n_wt : numpy array (n_words, n_topics)
            n_wt estimates
        """
        num_documents, num_words = data_batch.shape
                
        # your code is here
        # count n_wt for batch
        # set Theta_part
        
        return Theta_part, n_wt
            
    def _EM_iteration(self, data):
        """
        E-step of the algorithm. It should include 
        implementation of the E-step and M-step for the Theta matrix.
        
        Don't store ndwt in the memory simultaneously!
        
        Parametrs:
        ---------------
        data: sparse array shape (n_documents, n_words) or
            list of sparse array shape (n_documents, n_modality_words) in modalities case
            Array of data points. Each row corresponds to a single document.
        """ 
        # your code
        # set self._nwt shape of (num_words, num_topics)
        # set self._ntd shape of (num_topics, num_documents)
        
        num_documents, num_words = data.shape
        num_batches = int(np.ceil(num_documents / self.batch_size))
                
        for batch_number in range(num_batches):
            batch_start_border = batch_number * self.batch_size
            batch_end_border = (1 + batch_number) * self.batch_size
            
            Theta_part = None # your code is here
            
            Theta_part, n_wt_parts = self._EM_step_for_batch(data[batch_start_border:batch_end_border],
                                                            Theta_part)            
            # your code
            # Theta estimates
            # n_wt accumulation
        
        # your code
        # Phi estimates
        
    def EM_fit(self, data, phi_init=None, theta_init=None,
               trace=False, vocab=None, document_names=None):
        """
        Parameters:
        -----------
        data: sparse array shape (n_documents, n_words) or
            list of sparse array shape (n_documents, n_modality_words) in modalities case
            Array of data points. Each row corresponds to a single document.
        
        phi_init : numpy array (n_words, n_topics)
            Init values for phi matrix
            
        theta_init : numpy array (n_topics, n_documents)
            Init values for theta matrix
        
        trace: bool
            If True then return list of likelihoods
            
        vocab: list of words or list of list of words in modalities case
            vocab[i] - word that corresponds to an i column of data 
        
        document_names : list of str
            document_names[i] - name of the i-th document
        """
        num_documents, num_words = data.shape
        
        if phi_init:
            self.Phi = phi_init
        else:
            # use normalized random uniform dictribution
            # in bonus task for modalities Phi must be a list of numpy arrays
        
        if theta_init:
            self.Theta = theta_init
        else:
            # use the same number for all Theta values, 1 / num_topics
            self.Theta = None
            
        log_likelihood_list = []
        
        for i in range(self.max_iter):
            # your code is here
            
            if trace:
                log_likelihood_list += [self.compute_log_likelihood(data)]
        
        if trace:
            return self.Phi, self.Theta, log_likelihood_list
        else:
            return self.Phi, self.Theta
    
    def get_Theta_for_new_documents(self, data, num_inner_iter=10):
        """
        Parameters:
        -----------
        data: sparse array shape (n_new_documents, n_words) or
            list of sparse array shape (n_documents, n_modality_words) in modalities case
            Array of data points. Each row corresponds to a single document.
            
        num_inner_iter : int
            Number of e-step implementation
        """
        try:
            old_Theta = self.Theta
            
            # your code
            self.Theta = None
            
            # your code
            for i in range(num_inner_iter):
                pass
                
                
        finally:
            new_Theta = self.Theta
            self.Theta = old_Theta
        
        return new_Theta
    
    
    def compute_log_likelihood(self, data):
        """
        Parametrs:
        ---------------
        data: sparse array shape (n_documents, n_words)
            Array of data points. Each row corresponds to a single document.
        """
        return None
    
    def get_top_words(self, k):
        """
        Get list of k top words for each topic
        """
        # use argmax for Phi
        pass
        
    def get_top_docs(self, k):
        """
        Get list of k top documents for each topic
        """
        n_d = self._ntd.sum(axis=0)
        n_t = self._ntd.sum(axis=1)
        p_dt = None
        # use argmax for p_dt
        pass


class BaseRegularizer:
    def __init__(self, tau=1.0):
        """
        Parameters:
        ----------
        tau : float
            Regularization coef
        """
        self.tau = tau
        
    def grad(self, Phi, Theta, nwt, ntd):
        """
        Gradients for Phi and for Theta
        """
        raise NotImplementedError('must be implemented in subclass')

class KLWordPairsRegularizer(BaseRegularizer):
    def __init__(self, tau, word_pairs):
        """
        Parameters:
        ----------
        tau : float
            Regularization coef
            
        word_pairs : dict (str, list_of_str) or (int, list_of_ints)
            Dict of words and their translations. Implementation depends on you. 
        """
        super().__init__(self, tau)
        self.word_pairs = word_pairs
    
    def grad(self, Phi, Theta, nwt, ntd):
        """
        Gradients for Phi and for Theta
        """
        
        dR_dPhi = None
        dR_dTheta = np.zeros(Theta.shape)
        
        return dR_dPhi * self.tau, dR_dTheta

class KLDocumentPairsRegularizer(BaseRegularizer):
    def __init__(self, tau, document_pairs):
        """
        Parameters:
        ----------
        tau : float
            Regularization coef
            
        document_pairs : dict (int, list of ints)
            Dict of documents and their parallel variant
        """
        super().__init__(self, tau)
        self.document_pairs = document_pairs
    
    def grad(self, Phi, Theta, nwt, ntd):
        """
        Gradients for Phi and for Theta
        """
        
        dR_dPhi = np.zeros(Phi.shape)
        dR_dTheta = None
        
        return dR_dPhi, dR_dTheta * self.tau
