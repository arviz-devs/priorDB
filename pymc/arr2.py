import numpy as np
import pymc as pm
import pytensor.tensor as pt


# Specify the order of the autoregressive model and the observed data
p = 2
y_data = np.array([...])  # Replace with your actual time series data

# Hyperparameters for the ARR2 prior
cons = 0.1 * np.ones(p)
mean_R2 = 1 / 3
prec_R2 = 3

# Scale to make HalfNormal have unit variance
HALFNORMAL_SCALE = 1 / np.sqrt(1 - 2 / np.pi)

with pm.Model() as model:
    sigma = pm.HalfNormal("sigma", HALFNORMAL_SCALE)

    zb = pm.Normal("zb", mu=0.0, sigma=1.0, shape=p)
    psi = pm.Dirichlet("psi", a=cons)
    R2 = pm.Beta("R2", mu=mean_R2, nu=prec_R2)

    tau2 = R2 / (1 - R2)
    phi = pm.Deterministic("phi", zb * (sigma / y_data.std()) * pt.sqrt(tau2 * psi))

    pm.AR(
        "obs",
        phi,
        np.sqrt(2) * sigma,
        init_dist=pm.Normal.dist(0, 1, shape=p),
        observed=y_data,
    )
