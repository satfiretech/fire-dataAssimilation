# Satellite-based fire detection using data assimilation concepts

This package includes the MATLAB source codes for a satellite-based fire detection method.  The background temperature of a pixel is estimated using the data assimilation scheme and the ensemble forecasting mechanism. The threshold on the difference between the observed brightness temperature and the expected background temperature is derived under Constant False Alarm Rate (CFAR) framework. Data acquired from the Meteosat Second Generation (MSG)-SEVIRI sensor are used in this project.  


The package implements three data assimilation methods:

 •	The Ensemble Kalman Filter (EnKF)

 •	The Sampling Importance Resampling (SIR)

 •	The weak-constraint Four-Dimensional Variational Assimilation (4D-Var)


## Data

MSG Level 1.5 image dataset can be acquired from the [EUMETSAT data centre] (https://www.eumetsat.int/website/home/Data/DataDelivery/EUMETSATDataCentre/index.html). 

Brighteness temperatures are extracted using the [GDAL library](https://gdal.org/).

The following file contains the structured format of the IR 3.9 brightness temperature from (20 S, 23 E) to (33 S, 38 E): 
FNDATAREGIONFILE.mat at https://www.dropbox.com/s/ibhdy51j91hdxwr/FNDATAREGIONFILE.mat?dl=0

## Running the tests

```
>> dtc_v_fire_dynamic_main_fireExample
```

## Reference

The method is described in the article:

Udahemuka, G.; van Wyk, B.J.; Hamam, Y. Characterization of Background Temperature Dynamics of a Multitemporal Satellite Scene through Data Assimilation for Wildfire Detection. Remote Sens. 2020, 12, 1661. doi: [10.3390/rs12101661](https://doi.org/10.3390/rs12101661)

