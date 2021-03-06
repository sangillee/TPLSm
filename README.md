# TPLSm 1.0.2

Updated March 6th 2021

MATLAB package for performing Thresholded Partial Least Squares (T-PLS).  
T-PLS combines partial least squares with back-projection and thresholding to fit a large scale regularized linear model.  
Due to its mostly analytical solutions and 'fit once tune later' approach to cross-validation, it can fit cross-validated  
model on big data (both large n & p) in very short time.

The folder 'TPLSm' contains the actual functions necessary to run. This is the only folder that's technically needed.  
Add this folder to the MATLAB path, and you should be good to go.

The folder 'examples' contains example MATLAB script and data to show how to use the functions in the folder 'TPLSm'.  
This is for your aid only, and is not a necessary part of the package.

Citation:
Lee, S., Bradlow, E. T., & Kable, J. W. (2021). Thresholded Partial Least Squares: Fast Construction of Interpretable Whole-brain Decoders. BioRXiv.