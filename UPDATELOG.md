# TPLSm 1.0.3

Updated June 6th 2022

Finalized update including citations to the publication.

# TPLSm 1.0.2

Updated March 6th 2021

Separated input checking function into a separate file for generalized treatment.

# TPLSm 1.0.1

Updated March 5th 2021

General:  
Added examples.  
Added citation in readme.  
Added input variable checking for safety.  

TPLS:  
Removed support for single floating point precision operation in TPLS.  
Disabled saving of weights in TPLS model as it doesn't seem to do anything.  
Added loop abortion based on norm of covariance. Will stop from generating meaningless components based on noise.

TPLS_cv:  
Added support for more sophisticated cross-validation support in TPLS_cv.  
Added support for fold-specific weights in TPLS_cv.  

evalTuningParam:  
Added a Accuracy, negative MSE, and negative RMSE as a performance metric.  
Fixed a bug of AUC that occurs where there's only one observation of Y = 1.  


# TPLSm 1.0.0

Initial release.