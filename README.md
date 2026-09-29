# condobuyUQ2026

A simple uncertainty quantification (UQ) model for the decision to buy a condo in Keystone, CO in 2026 vs. investing the cash elsewhere and renting instead.

Some note to incorporate later when we fill this out:
Mean growth models assume strictly Gaussian distributions at every time step.  The process, called OU, is a mean reversion process.  It takes the current mean growth rate and reverts it to the long term average growth rate.  The long term stable growth rate is in % annual, so the monthly plots are showing monthly growth.

All of the growth models are scraped data based and in units of change fraction.  They are independent of any input deck settings.

All derived values, are weighted combinations of growth models.  They are in the same units as growth models.  They are independent of any input deck settings, and sceneriors, except the weights for the growth models (composition).

Priors are various models which pull from various growth and or derived value models.  They are in absolute value units.  They can be input deck and scenerio setting dependent.
