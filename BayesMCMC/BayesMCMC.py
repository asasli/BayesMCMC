import scipy
from scipy.integrate import quad
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
'''Markov Chain Monte Carlo (MCMC), Bayesian Statistics & Distances to stars '''
from BayesMCMC import resources


N_stars = 500

dist_max = 100.0
# Generate random distances within nearest 1000 pc
dist = dist_max*np.random.rand(N_stars)**(1.0/3.0)

plx = 1.0/dist

# Generate mock observations
plx_err = 2.0*1.0e-3 * np.ones(len(plx))
plx_obs = plx + np.random.normal(0.0, plx_err, size=N_stars)

alpha_0 = 3.0
step_size = 0.3
chain = resources.metro_hastings(resources.ln_posterior, alpha_0, args=(dist_max, plx_obs, plx_err), 
                       step_size=step_size, N_steps=200)

n_burnin = 10   # change this value if needed

plt.figure()
plt.axvspan(0, n_burnin, color="k", alpha=0.3)
plt.plot(chain)
_, _, y_min, y_max = plt.axis()
plt.text(n_burnin + 3, (y_min + y_max) / 2.0, "Burn-in")
chain_converged = chain[n_burnin:]
plt.xlim(0,len(chain))
plt.show()

plt.figure()
plt.hist(chain_converged, bins=10, density=True)
_, _, _, y_max = plt.axis()
plt.ylim(ymax=y_max*1.4)

lo68, median, hi68 = np.percentile(chain_converged, [16,50,84])

plt.axvline(2.0, color="k", label="True value")
plt.axvline(median, color="r", linewidth=2, label="Median")
plt.axvspan(lo68, hi68, color="r", alpha=0.3, label="68% CI: 1$\sigma$")
plt.title(r"${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(median, hi68 - median, median - lo68))
plt.xlabel(r"$\alpha$")
plt.ylabel("Posterior density")
plt.legend(loc="upper center", ncol=3)
plt.show()