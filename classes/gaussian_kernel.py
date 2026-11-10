import numpy as np


class GaussianKernel:  # ChatGPT
    def __init__(self, center, sigma):
        """
        center: array-like -> mean of the Gaussian
        sigma: float -> standard deviation (same for both dimensions)
        """
        self.mu = np.array(center, dtype=float)
        self.sigma = float(sigma)
        self.dim = 1

        # Precompute constants
        self.norm_const = 1 / ((2 * np.pi * self.sigma) ** (self.dim / 2))

    def pdf(self, x):
        """
        Compute probability density at point x.
        x: array-like of shape (2,)
        """
        x = np.array(x, dtype=float)
        diff = x - self.mu
        exponent = -0.5 * np.dot(diff, diff) / (self.sigma**2)
        if np.exp(exponent) == 0:
            print("Exponent: ", exponent)
            print("Difference: ", diff)
            print("Dot product:", np.dot(diff, diff))
            raise ValueError("Zero here")
        return self.norm_const * np.exp(exponent)
