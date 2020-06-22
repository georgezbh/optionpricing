The project is to set up a library to price options and structure products.
It starts with my python learning incentive and the goal is to have a free, user friendly, comprehensive tool to price variours derivatives products with proper performance. The library should be able to calibrarte the curve and smile using raw data and show price.

Currently it can price:

Vanilla options with Black-Scholes, binomial tree, PDE and Monte Carlo simulation

Signle barrier (up & down) & (in & out) option with PDE and Monte Carlo

Target redemption forward with Monte Carlo simulation

Autocallable with Monte Carlo simulation

All with constant volatility (which is obviously not true in reality)

Next step:

1. Develop a volatility smile management tool with Local vol, Henston model and SABR model.
2. Refactor code to replace array manipulation using Numpy to impove performance
3. CUDA with numba implementation to utilize GPU
