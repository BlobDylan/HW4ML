"""
Dylan Lewis, 209722610
Ziv Picciotto, 206919722
"""

import numpy as np
import pandas as pd


def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    X = x.values
    Y = y.values

    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    X = X - mean_x
    Y = Y - mean_y

    numerator = np.dot(X,Y)
    denomenator = np.sqrt(np.dot(X,X) * np.dot(Y,Y))
    r = numerator/denomenator
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    features = {}
    for feature in X.drop(['id', 'date'], axis=1).columns.to_list():
        corr = pearson_correlation(X[feature], y)
        features[feature] = np.abs(corr)
        # print(f'feature: {feature} correlations: {corr}')
    
    highest_items = sorted(features.items(), key=lambda item: item[1], reverse=True)
    best_features = [item[0] for item in highest_items][:n_features]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []
    
    def sigmoid(self, x):
        """
        Computes the sigmoid function (vectorized)
        """
        return 1 / (1 + np.exp(-x))
    
    def cost(self, X, y):
        """
        Computes the cost of a given theta 
        """
        
        sig = self.sigmoid(np.dot(X, self.theta))
        J = (-np.dot(y,np.log(sig))) - (np.dot((1-y), np.log(1- sig)))
        
        return J / len(X)

    def bias(self, X):
        """
        Apply bias trick
        """
        return np.insert(X, 0, 1, axis=1)
    
    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = X.copy()
        X = self.bias(X)
        i = 0
        self.theta = np.random.random(size=len(X[0]))
        self.Js.append(self.cost(X, y))
        while i < 2 or (i < self.n_iter and abs(self.Js[-2]- self.Js[-1]) >= self.eps):
            
            # Updating theta according to gradient descent
            pred_x = np.dot(X,self.theta)
            self.theta = self.theta - (self.eta * np.dot(X.T, self.sigmoid(pred_x) - y))

            self.Js.append(self.cost(X, y))
            self.thetas.append(self.theta)

            i += 1

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = X.copy()
        X = self.bias(X)

        preds = self.sigmoid(np.dot(X, self.theta)) > np.array([0.5] * len(X))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds.astype(int)

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrices

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    merged = np.column_stack((X,y))
    np.random.shuffle(merged)
    cv_accuracy = 0
    subsets = np.array_split(merged, folds)
    for i, validation_set in enumerate(subsets):
        # Combining all subsets except the i'th subset 
        stacks = [part for j, part in enumerate(subsets) if j != i]
        training = np.vstack(stacks)

        #Seperating back to X and y.
        training_x, training_y = training[:,:-1], training[:,-1] 
        
        # Trainiing the model
        algo.fit(training_x, training_y)

        # Predicting
        validation_x, validation_y = validation_set[:,:-1], validation_set[:,-1]
        predictions = algo.predict(validation_x)

        # Computes current accuracy
        cv_accuracy += np.mean(predictions == validation_y)
        
    cv_accuracy /= folds
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((data-mu) ** 2)/(2*(sigma ** 2)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.weights = np.ones(self.k) / self.k
        self.mus = np.random.choice(data.flatten(), self.k)
        self.sigmas = np.ones(self.k)
        self.costs = []
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.responsibilities = self.weights * norm_pdf(data,self.mus,self.sigmas)
        sum_responsibilities = np.sum(self.responsibilities, axis=1, keepdims=True)
        self.responsibilities /= sum_responsibilities

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        m = data.shape[0]
        
        self.weights = np.mean(self.responsibilities, axis=0)
        self.mus = np.sum(self.responsibilities * data.reshape(-1,1), axis=0) / np.sum(self.responsibilities, axis=0)
        self.sigmas = np.sqrt((np.mean(self.responsibilities * np.square(data.reshape(-1,1) - self.mus),axis=0))/self.weights)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def compute_cost(self,data):
      """
      Computes cost.
      """
      m = data.shape[0]
      cost = 0
      for i in range(m):
          p = 0
          for j in range(self.k):
              p += self.weights[j] * norm_pdf(data[i], self.mus[j], self.sigmas[j])
          cost -= np.log(p)
      self.costs.append(cost) 

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)
        self.compute_cost(data)  # Compute initial cost before entering the loop

        i = 0
        while i < self.n_iter:
            self.expectation(data)
            self.maximization(data)
            self.compute_cost(data)
            # Check for convergence if we have more than one cost value
            if i > 0 and abs(self.costs[-2] - self.costs[-1]) < self.eps:
                break
            i += 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pdf = 0
    for i in range(len(weights)):
        pdf += weights[i] * norm_pdf(data,mus[i],sigmas[i])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.class_0 = None
        self.class_1 = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        vals, counts = np.unique(y, return_counts=True)
        self.prior = {label: count / len(y) for label, count in dict(zip(vals,counts)).items()}

        self.class_0 = {feature: EM(k=self.k, random_state=self.random_state) for feature in range(X.shape[1])}
        self.class_1 = {feature: EM(k=self.k, random_state=self.random_state) for feature in range(X.shape[1])}
        
        X_Y_0 = X[y==0]
        X_Y_1 = X[y==1]

        for feature in range(X.shape[1]):
            self.class_0[feature].fit(X_Y_0[:, feature].reshape(-1,1))
            self.class_1[feature].fit(X_Y_1[:, feature].reshape(-1,1))

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        preds = []

        for x in X:
            likelihood_0 = self.prior[0]
            likelihood_1 = self.prior[1]

            # Calculate the likelihood for each class by multiplying the GMM PDFs of each feature
            for i in range(len(x)):
                likelihood_0 *= np.sum(gmm_pdf(x[i], *self.class_0[i].get_dist_params()))
                likelihood_1 *= np.sum(gmm_pdf(x[i], *self.class_1[i].get_dist_params()))
            preds.append(0 if likelihood_0 > likelihood_1 else 1)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return np.array(preds)

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    lor = LogisticRegressionGD(eta = best_eta, eps = best_eps)
    bayes = NaiveBayesGaussian(k=k)

    lor.fit(x_train,y_train)
    bayes.fit(x_train,y_train)

    lor_train_acc = np.mean(lor.predict(x_train) == y_train)
    lor_test_acc = np.mean(lor.predict(x_test) == y_test)
    bayes_train_acc = np.mean(bayes.predict(x_train) == y_train)
    bayes_test_acc = np.mean(bayes.predict(x_test) == y_test)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    size = 6000
    
    mus_a = [
        [0,0,0], 
        [5,5,5],
        [10,10,10]
    ]
    cov_mat_a = [
        [2,0,0],
        [0,2,0],
        [0,0,2]
    ]

    mus_b = [
        [0,4,0], 
        [0,7,0]
    ]
    cov_mat_b = [
        [4,4,4],
        [4,4,4],
        [4,4,5]
    ]
        

    rv_A = multivariate_normal(mus_a[0], cov_mat_a)
    rv_B = multivariate_normal(mus_a[1], cov_mat_a)
    rv_C = multivariate_normal(mus_a[2], cov_mat_a)
    
    dataset_a_features = np.vstack((rv_A.rvs(size=size // 3), rv_B.rvs(size=size//3), rv_C.rvs(size=size//3)))
    dataset_a_labels = np.vstack((np.ones((size // 3, 1)), np.zeros((size // 3, 1)), np.ones((size // 3 , 1)))).flatten()


    rv_D = multivariate_normal(mus_b[0], cov_mat_b, allow_singular=True)
    rv_E = multivariate_normal(mus_b[1], cov_mat_b, allow_singular=True)
    
    dataset_b_features = np.vstack((rv_D.rvs(size=size // 2), rv_E.rvs(size=size//2)))
    dataset_b_labels = np.vstack((np.ones((size // 2, 1)), np.zeros((size // 2, 1)))).flatten()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }

