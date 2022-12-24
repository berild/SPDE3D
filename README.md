# Code for "Spatially varying anisotropy for Gaussian random field in three-dimensions"
This code is for the journal article "Spatially varying anisotropy for Gaussian random field in three-dimensions". Original implementation is found at [https://github.com/berild/GMRFpy](https://github.com/berild/GMRFpy) but without comments and not specified for this publication.

* **spde.py**: Collection class for all models:
  * **StatIso.py**: Class for Stationary Isotropic model (SI).
  * **StatAnIso.py**: Class for Stationary Anisotropic model (SA). 
  * **NonStatIso.py**: Class for Non-Stationary Isotropic model (NI). (Not used)
  * **NonStatAnIso.py**: New class for Non-Stationary Anisotropic model (NA)(190 params).
* **grid.py**: Constructs the grid class for the **spde.py**. Call '_.setGrid()' to set new grid.
* **mission.py**: Specifically handles the nidelva mission 27th of may 2021. Can be used to fit the different models or evaluate the result.
* **auv.py**: class that could operate onboard autonomous underwater vehicles. 
* **rqinv.R**: R implementation that handles the partial inverse of a covariance matrix. Uses the R-INLA package.
* **AH3D2.cpp**: Handles the construction of the **A_H** matrix in the precision matrix of the spatial field. **ah3d2.py** is used to communicate between C++ and python. 


## Simulation Study
* **./simulation_study_models/**: holds the parameter values of the different models, and the inital values of the optimization. 
* **runsim.py**: simulates all the dataset for all models used in the simulation study and stores them in the **simulations** folder. 
* **runfit.py**: fits all the models to all the datasets. run 'python3 runfit.py *model* *number*' where *model* is 1,2 or 4 for SI, SA, or NA, and *number* defines the trial from which to start to fit from (between 1-100). Fits are saved in the folder **./fits/**. This runs in parallel, but still will take a very long time (~weeks). The resulting model estimates are included in this repo.
* **runres.py**: runs the statistics for all the model estimates for the different models and prints it. Run 'python3 runres.py'.

## Application as emulator of numerical ocean model
* **emulator.npz**: the simulated dataset from the numerical ocean model SINMOD on the 27th may 2021. 
* **mission.csv**: the assimilated measurements collected with an autonomous underwater vehicles on the 27th of may 2021. 
* **fit_nidelva.py**: fits a specified model to the simulated dataset from SINMOD. Run 'python3 fit_nidelva.py 2' for stationary anisotropic fit and 'python3 fit_nidelva.py 4' for non-stationary aniostropic fit. (this will take some time)
* **create_fugyres.py**: makes a specified figure in the paper. Figure (4,6,7,8, or 9). Figure 9 takes a while to create. 
