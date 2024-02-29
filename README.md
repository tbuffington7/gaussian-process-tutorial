# Description
This is a repository to help others understand the intuition behind Gaussian processes and Gaussian process regression. 

The basic idea is that we start with univariate normal distributions, which most people are familiar with. The notebook lets users visualize the effects of changing
the mean and variance of this distribution. Then we look at bivariate normals, which are characterized by mean vectors and covariance matricies.
The produces randomly drawn vectors of size 2. Then we look at Gaussian processes, which are essentially infinite-dimensional multivariate normal distributions.
These stochastic objects are distributions of functions, which make them incredibly powerful for machine learning, especially when used in a Bayesian context. 

The interactive tutorial is contained in the GP_tutorial.ipynb notebook. Note that the predict and covariance functions are located in gp_methods.py. 

Although it is generally recommended to use an ML library such as scikit-learn to build GPs in production, I built the GPs in this repo using just scipy and numpy. 
This may be helpful to see for some as it explicitly shows the matrix algebra result implemented into code. 


# Requirements
- numpy
- pandas
- matplotlib
- scipy
- jupyter
