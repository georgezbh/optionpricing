The project is to set up a library to price options and structure products.
It starts with my python learning incentive and the goal is to have a user friendly, comprehensive tool to price variours derivatives products. The library should be able to calibrarte the curve and smile using raw data and show price.

Currently it can price:

Vanilla options with Black-Scholes, binomial tree, PDE and Monte carlor simulation

Target redemption forward with Monte Carlo simulation


All with constant volatility (which is obviously not true in reality)

Next step:

1. Add Autocallable in the library
2. Start volatility smile management tool with Local vol and Henston model.
