# Satellite-based fire detection using data assimilation concepts

This package includes the MATLAB source codes for a satellite-based fire detection method.  The background temperature of a pixel is estimated using the data assimilation scheme and the ensemble forecasting mechanism. The threshold on the difference between the observed brightness temperature and the expected background temperature is derived under Constant False Alarm Rate (CFAR) framework. 


The package implements three data assimilation methods:

 •	The Ensemble Kalman Filter (EnKF)

 •	The Sampling Importance Resampling (SIR)

 •	The weak-constraint Four-Dimensional Variational Assimilation (4D-Var)


## fire-dataAssimilation
FNDATAREGIONFILE.mat at https://www.dropbox.com/s/ibhdy51j91hdxwr/FNDATAREGIONFILE.mat?dl=0


## Reference

The method is described in the article:

Udahemuka, G.; van Wyk, B.J.; Hamam, Y. Characterization of Background Temperature Dynamics of a Multitemporal Satellite Scene through Data Assimilation for Wildfire Detection. Remote Sens. 2020, 12, 1661. doi: [10.3390/rs12101661](https://doi.org/10.3390/rs12101661)

