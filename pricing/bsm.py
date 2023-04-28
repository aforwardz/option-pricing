import numpy as np
from scipy.stats import norm

N = norm.cdf


class BlackScholesMerton:

    def call_price(self, S, K, T, r, sigma):
        """
        :param S: current price of the underlying asset
        :param K: Exercise price
        :param T: Remaining time until expiration in years
        :param r: Risk free rate of return
        :param sigma: volatility(σ)
        :return:
        """
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * N(d1) - K * np.exp(-r * T) * N(d2)

    def put_price(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * N(-d2) - S * N(-d1)

