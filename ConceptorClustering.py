import numpy as np
from sklearn import cluster, manifold, metrics

class Reservoir(object):
    """ Reservoir Class
    
    Parameters
    ----------
    n_features: (default=1)
        number of input features, set in range (>=1)
    n_reservoir: (default=100)
        number of reservoir neurons, set in range(1, 10000)
    spectral_radius: (default=0.95)
        spectral radius of the absolute of recurrent weight matrix,
        set in range(0, 1)
        (see Notes for more information)
    connectivity: (default=0.05) 
        proportion of recurrent weights are retained, set in range(0, 1)
        (see Notes for more information)
    random_state: (default=None)
        integer seed, np.rand.RandomState object, or None to use numpy's builting RandomState.
    
    Attributes
    ----------
    W: (n_reservoir, n_reservoir) array
        recurrent weight matrix 
    W_in: (n_reservoir, n_features) array
        input weight matrix
    reservoir_state: (n_reservoir,) array
        state of the reservoir
    
    Notes
    -----
    The setting of parameters spectral_radius and connectivity affect the 
    dynamic of the reservoir. The proper setting can obtain reservoir with 
    long-term memory. Larger spectral_radius brings longer memory, proper 
    connectivity, e.g. 5 / n_reservoir, usually derives better dynamic for reservoir.
    
    References
    ----------
    Xu, M., P. Baraldi, and E. Zio. "Fault diagnostics by conceptors-aided clustering." 
    In 30th European Safety and Reliability Conference, ESREL 2020 and 15th Probabilistic
    Safety Assessment and Management Conference, PSAM 2020, pp. 3656-3663. 
    Research Publishing Services, 2020.
    
    Jaeger, Herbert. "Using conceptors to manage neural long-term memories for temporal patterns."
    The Journal of Machine Learning Research 18, no. 1 (2017): 387-429.
    
    Examples
    --------
    >>> # generate a set of multivariate time series dataset (n_samples, n_timestamps, n_features) array
    >>> # it combines 10 short time series with n_timestamps=100 and 10 long time series with n_timestamps=200
    >>> X_short = np.random.randn(10,100,2)
    >>> tail_arr = np.zeros((10,100,2)) + np.NaN
    >>> X_short = np.concatenate((X_short,tail_arr),axis=1)
    >>> X_long = np.random.randn(10,200,2)
    >>> X = np.concatenate((X_short,X_long),axis=0)
    
    >>> # build reservoir object
    >>> res = Reservoir(n_features=2)
    
    >>> # build reservoir object
    >>> # (assign arguments by users)
    >>> res = Reservoir(n_features=2, 
    ...                n_reservoir=100,
    ...                spectral_radius=0.95, 
    ...                connectivity=0.05,
    ...                random_state=2)
    
    >>> # convert these 20 multivariate time series into 20 Conceptors
    >>> X_C = res.transform_conceptor(X)
    
    >>> # show an example of Conceptor
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.imshow(X_C[0])
    >>> plt.show()
    
    """
        
    def __init__(self, 
                 n_features=1,
                 n_reservoir=100,
                 spectral_radius=0.95,
                 connectivity=0.05,
                 random_state=None):
        
        self.n_features = n_features
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand
        
        self.reservoir_state = np.zeros((n_reservoir,))
        
        self._init_weights()

    def _init_weights(self):
        """Initialize the weight of Reservoir network.
        
        Notes
        -----
        The weights are normalized in ``Uniform`` distribution and the 
        ``Spectral Radius`` < 1.
        

        """
        # initialize recurrent weights:
        # begin with a random matrix centered around zero:
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # delete the fraction of connections given by (self.connectivity):
        W[self.random_state_.rand(*W.shape) > self.connectivity] = 0
        # compute the spectral radius of these weights:
        radius = np.max(np.linalg.eigvals(np.abs(W)))
        radius = np.real(radius)
        # rescale them to reach the requested spectral radius:
        self.W = W * (self.spectral_radius / radius)

        # random input weights:
        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_features) * 2 - 1

    def _update_reservoir(self, x):
        """Performs one update step.
        
        Note
        ----
        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        
        Parameters
        ----------
        x: (n_features,) array
            the input pattern at a time step
        
        Returns
        -------
        reservoir_state: (n_reservoir,) array
            the updated state of the reservoir
        """
        self.reservoir_state = np.tanh(np.dot(self.W, self.reservoir_state) +
                                       np.dot(self.W_in, x))
        return self.reservoir_state
    
    
    def reset_reservoir(self, reservoir_state=None):
        """reset the reservoir state by reservoir_state argument or zeros

        Parameters
        ----------
        reservoir_state : (n_reservoir,) array, optional
            The initial state used to set the reservoir state. The default is None.

        Returns
        -------
        None.

        """
        if reservoir_state is None:
            self.reservoir_state = np.zeros((self.n_reservoir,))
        elif reservoir_state is not None:
            assert reservoir_state.shape == self.reservoir_state.shape, "reservoir_state shape doesn't match."
            self.reservoir_state = reservoir_state
        
            
    def transform_conceptor(self, X, alpha=1.):
        """Transform the input X into Conceptor matrix
        
        Parameters
        ----------
        X: (n_samples, n_timestamps, n_features) array
            Multivariate time series dataset. The time series may have different length,
            the short ones are padding with np.NaN type.
        alpha: float number, (default=1)
            A control parameter called aperture, which control the scaling of 
            singular values of correlation matrix of reservoir states, set in
            the range (1, 10, 1000, 10000) (see Notes)
        
        Returns
        -------
        X_C: (n_reservoir, n_reservoir) array
            Conceptor matrix transformed from multivariate time series dataset.
        
        Notes
        -----
        Alpha usually use an emperical value of 1.
        Each sample of multivariate time series can have different length.
        
        References
        -----------
        Xu, M., P. Baraldi, and E. Zio. "Fault diagnostics by conceptors-aided clustering." 
        In 30th European Safety and Reliability Conference, ESREL 2020 and 15th Probabilistic
        Safety Assessment and Management Conference, PSAM 2020, pp. 3656-3663. 
        Research Publishing Services, 2020.
        
        Examples
        --------
        >>> X = np.random.randn(10,100,2)
        >>> res = Reservoir(n_features=2, 
        ...                n_reservoir=100,
        ...                spectral_radius=0.95, 
        ...                connectivity=0.05,
        ...                random_state=2)
        >>> X_C = res.transform_conceptor(X, alpha=1.)
        
        """
        n_samples, n_timestamps, n_features = X.shape
        n_reservoir = self.reservoir_state.shape[0]
        X_C = np.zeros((n_samples, n_reservoir, n_reservoir))
        
        R = np.zeros((n_reservoir, n_reservoir))
        
        for i_s,u in enumerate(X):
            for i_t in range(1,n_timestamps):
                if not np.isnan(np.sum(u[i_t])):
                    # update R matrix
                    x = self._update_reservoir(u[i_t])
                    x = x[:,np.newaxis]
                    R = R*(i_t-1)/i_t + np.dot(x,x.T)/i_t
                    
                elif np.isnan(np.sum(u[i_t])):
                    break
                
            # compute C matrix
            R_inv = np.linalg.inv(R + alpha**(-2) * np.eye(n_reservoir))
            C = np.matmul(R, R_inv)
            X_C[i_s] = C
        
        return X_C

def conceptor_clustering(X_C, 
                         n_clusters=None,
                         n_components=None,
                         sigma=1.,
                         random_state=None):
    """Apply clustering to Conceptor matrixes.
    

    Parameters
    ----------
    X_C : (n_samples, n_reservoir, n_reservoir) array
        Conceptor matrixes transformed by using Reservoir.transform_conceptor(X).
    n_clusters : int, default=None
        Number of clusters to extract.
    n_components : int, default=n_clusters
        Number of eigenvectors to use for the spectral embedding.
    sigma : float, default=1
        Parameter in Radius Basis Function in similarity measure.
    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization of by 
        spectral clustering and K-Means initialization. Use an int to make
        the randomness deterministic.

    Returns
    -------
    labels : (n_samples,) array
        The labels of the clusters.
    silhouette_scores : dict
        The silhouette scores w.r.t. number of clusters.
        For example, silhouette_scores[3] denotes the score if n_clusters=3.
        note silhouette_scores[0], silhouette_scores[1] are -Inf. 
    affinity_matrix : (n_samples, n_samples) array
        The affinity matrix which measures the similarity of Conceptors between 
        each two multivariate time series.
            * affinity_matrix = np.exp(- D**2 / (2 * sigma**2)) *
        where D is Conceptor Distance matrix.
        
    Notes
    -----
    silhouette_scores may have different length for different cases.
    
    References
    ----------
    Xu, M., P. Baraldi, and E. Zio. "Fault diagnostics by conceptors-aided clustering." 
    In 30th European Safety and Reliability Conference, ESREL 2020 and 15th Probabilistic
    Safety Assessment and Management Conference, PSAM 2020, pp. 3656-3663. 
    Research Publishing Services, 2020.
    
    Examples
    --------
    >>> # generate a set of multivariate time series dataset (n_samples, n_timestamps, n_features) array
    >>> # it combines 10 short time series with n_timestamps=100 and 10 long time series with n_timestamps=200
    >>> X_short = np.random.randn(10,100,2)
    >>> tail_arr = np.zeros((10,100,2)) + np.NaN
    >>> X_short = np.concatenate((X_short,tail_arr),axis=1)
    >>> X_long = np.random.randn(10,200,2)
    >>> X = np.concatenate((X_short,X_long),axis=0)
    
    >>> # build reservoir object
    >>> res = Reservoir(n_features=2)
    
    >>> # build reservoir object
    >>> # (assign arguments by users)
    >>> res = Reservoir(n_features=2, 
    ...                n_reservoir=100,
    ...                spectral_radius=0.95, 
    ...                connectivity=0.05,
    ...                random_state=2)
    
    >>> # convert these 20 multivariate time series into 20 Conceptors
    >>> X_C = res.transform_conceptor(X)
    
    >>> # show an example of Conceptor
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.imshow(X_C[0])
    >>> plt.show()
    
    >>> # Clustering the Conceptors
    >>> # (automatically choose number of clusters, corresponding to the max silhouette score)
    >>> labels, silhouette_scores, affinity_matrix = conceptor_clustering(X_C)
    
    >>> # Clustering the Conceptors
    >>> # (assign arguments by users)
    >>> labels, silhouette_scores, affinity_matrix = conceptor_clustering(X_C, 
    ...                                                  n_clusters=3,
    ...                                                  n_components=3,
    ...                                                  sigma=1.,
    ...                                                  random_state=2)
    
    """
    
    n_samples = X_C.shape[0]
    assert n_samples > 2, "Too few time series samples (at least >= 3)."
    
    # obtain the affinity matrix of X_C
    D = np.zeros((n_samples,n_samples)) # distance matrix
    for i,c1 in enumerate(X_C):
        for j,c2 in enumerate(X_C):
            if j >= i:
                D[i,j] = np.linalg.norm((c1 - c2), 'fro')
            elif j < i:
                D[i,j] = D[j,i]
    
    A = np.exp(- D**2 / (2 * sigma**2)) # affinity matrix
    affinity_matrix = A
    
    # obtain the n_clusters and n_components if they are None
    max_n_clusters = min(max(8, n_samples//100), 100, n_samples-1)
    s_z = min(max_n_clusters*100, n_samples)
    silhouette_scores = {}
    if n_clusters is None:
        for n_c in range(2, max_n_clusters+1):
            # To obtain spectral embedding of Affinity matrix
            X_embed = manifold.spectral_embedding(A,
                                        n_components=n_c,
                                        random_state=random_state,
                                        eigen_tol=0.0,
                                        norm_laplacian=True)
            # Kmeans clustering
            kmeans = cluster.KMeans(n_clusters=n_c, random_state=random_state).fit(X_embed)
            labels = kmeans.labels_
            
            # To obtain silhouette score
            silhouette_scores[n_c] = metrics.silhouette_score(X_embed, labels,
                                                              metric='euclidean', 
                                                              sample_size=s_z,
                                                              random_state=random_state)
        # choose n_clusters with largest silhouette score
        n_clusters = max(silhouette_scores, key=silhouette_scores.get) 
    
    if n_components is None:
        n_components = n_clusters
    
    # call scikit spectral clustering algorithm
    labels = cluster.spectral_clustering(A, 
                                          n_clusters=n_clusters,
                                          n_components=n_components,
                                          random_state=random_state,
                                          assign_labels="kmeans")
    
    return labels, silhouette_scores, affinity_matrix
    

class ConceptorClustering(object):
    """The Conceptor Clustering Algorithm.
    
    Parameters
    ----------
    n_clusters : int, (default=None)
        Number of clusters to extract. 
        It can be automatically set by maximizing silhouette scores.
    n_reservoir: (default=100)
        number of reservoir neurons, set in range(1, 10000)
    spectral_radius: (default=0.95)
        spectral radius of the absolute of recurrent weight matrix,
        set in range(0, 1)
        (see Notes for more information)
    connectivity: (default=0.05) 
        proportion of recurrent weights are retained, set in range(0, 1)
        (see Notes for more information)
    n_components : int, (default=n_clusters)
        Number of eigenvectors to use for the spectral embedding.
    sigma : float, (default=1)
        Parameter in Radius Basis Function in similarity measure.
    random_state : int, RandomState instance, (default=None)
        A pseudo random number generator used for the initialization of by 
        spectral clustering and K-Means initialization. Use an int to make
        the randomness deterministic.
    
    
    Attributes
    ----------
    labels_ : numpy.ndarray
        Labels of each sample
    reservoir_ : Reservoir() object
        The reservoir instance of class Reservoir.
    affinity_matrix_ : numpy.mndarray
        The affinity matrix which measures the similarity of Conceptors between 
        each two multivariate time series.
            * affinity_matrix = np.exp(- D**2 / (2 * sigma**2)) *
        where D is Conceptor Distance matrix. 
    
    Notes
    -----
    The setting of parameters spectral_radius and connectivity affect the 
    dynamic of the reservoir. The proper setting can obtain reservoir with 
    long-term memory. Larger spectral_radius brings longer memory, proper 
    connectivity, e.g. 5 / n_reservoir, usually derives better dynamic for reservoir.
    
    
    References
    ----------
    Xu, M., P. Baraldi, and E. Zio. "Fault diagnostics by conceptors-aided clustering." 
    In 30th European Safety and Reliability Conference, ESREL 2020 and 15th Probabilistic
    Safety Assessment and Management Conference, PSAM 2020, pp. 3656-3663. 
    Research Publishing Services, 2020.
    
    Examples
    --------
    >>> # generate a set of multivariate time series dataset (n_samples, n_timestamps, n_features) array
    >>> # it combines 10 short time series with n_timestamps=100 and 10 long time series with n_timestamps=200
    >>> X_short = np.random.randn(10,100,2)
    >>> tail_arr = np.zeros((10,100,2)) + np.NaN
    >>> X_short = np.concatenate((X_short,tail_arr),axis=1)
    >>> X_long = np.random.randn(10,200,2)
    >>> X = np.concatenate((X_short,X_long),axis=0)
    
    >>> # create ConceptorClustering object 
    >>> # (automatically choose number of clusters, corresponding to the max silhouette score)
    >>> conceptor_cluster = ConceptorClustering()
    
    >>> # create ConceptorClustering object
    >>> # (assign arguments by users)
    >>> conceptor_cluster = ConceptorClustering(n_clusters=3,
    ...                                        n_reservoir=100, 
    ...                                        spectral_radius=0.95, 
    ...                                        connectivity=0.05, 
    ...                                        sigma=1.,
    ...                                        random_state=1)
    
    >>> # use fit() function to obtain prediction_labels, silhouette scores and affinity matrix
    >>> conceptor_cluster.fit(X)
    >>> print(conceptor_cluster.labels_)
    >>> print(conceptor_cluster.silhouette_scores_)
    >>> print(conceptor_cluster.affinity_matrix_)
    
    >>> labels = conceptor_cluster.fit_predict(X)
    >>> print(labels)
    
    """
    
    def __init__(self, n_clusters=None, 
                 n_reservoir=100, spectral_radius=0.95, connectivity=0.05,
                 n_components=None, sigma=1., random_state=None):
        self.n_clusters = n_clusters
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.n_components = n_components
        self.sigma = sigma
        self.random_state = random_state
    
    
    
    def fit(self, X, y=None, sample_weight=None):
        """Compute conceptor clustering.
        
        Parameters
        ----------
        X : array-like of shape=(n_samples, n_timestamps, n_features)
            Time series dataset.
        y
            Ignored
        """
        
        n_samples, n_timestamps, n_features = X.shape
        
        self.reservoir_ = Reservoir(n_features=n_features,
                                    n_reservoir=self.n_reservoir,
                                    spectral_radius=self.spectral_radius,
                                    connectivity=self.connectivity,
                                    random_state=self.random_state)

        X_C = self.reservoir_.transform_conceptor(X)
        self.labels_, self.silhouette_scores_, self.affinity_matrix_ = conceptor_clustering(
            X_C, 
            n_clusters=self.n_clusters,
            n_components=self.n_components,
            sigma=self.sigma,
            random_state=self.random_state)
        
        return self
    
    def fit_predict(self, X, y=None):
        """Fit conceptor clustering using X and then predict the closest cluster
        each time series in X belongs to.
        It is more efficient to use this method than to sequentially call fit
        and predict.
        
        Parameters
        ----------
        X : array-like of shape=(n_samples, n_timestamps, n_features)
            Time series dataset to predict.
        y
            Ignored
        Returns
        -------
        labels : array of shape=(n_samples, )
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_

    
