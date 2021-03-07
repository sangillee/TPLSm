% TPLS_example.m
% Written by Arthur 3/3/2021
% this script shows how one can use T-PLS to build a whole-brain decoder
% be sure to run line-by-line to follow along
clear
clc

%%%%%%%%%%%%% data explanation %%%%%%%%%%%%%%
% be sure to run this script in the examples directory where there's example data
% also make sure to have added the TPLSm folder to your path
load('TPLSdat.mat')

% this should load X, Y, run, subj, mask
% mask is the 3d brain image mask 
% see how many voxels there are inside the mask
% this should say that there are 3714 voxels
sum(mask(:))

% X is the single trial betas. It has 3714 columns, each of which corresponds to a voxel
% Y is binary variable to be predicted. In this case, the Y was whether the participant chose left or right button
% hopefully, when we create whole-brain predictor, we should be able to see left and right motor areas

% subj is a numerical variable that tells us the subject number that each observation belongs to.
% In this dataset, there are only 3 datasets.
% run is a numerical variable that tells us the scanner run that each observation belongs to.
% In this dataset, each of the 3 subjects had 8 scan runs.



%%%%%%%%%%%%%%%%%%% CV %%%%%%%%%%%%%%%%%%%%%
% Let's first do leave-one-subject out cross-validation to find the best tuning parameters
% We will give X and Y as variables and subj to be used as cross-validation folds
% The rest of the inputs are omitted to default.
cvmdl = TPLS_cv(X,Y,subj);

% That should have been pretty quick.
% Now we need to evaluate prediction performance across the three folds
% We will use AUC of ROC as the prediction performance metric
% By default we trained TPLS model with 25 components so we will try out 1 to 25 components in cross validation
compvec = 1:25;
% For thresholding, let's try 0 to 1 in 0.05 increments
threshvec = 0:.05:1;
% subfold is not always necessary but you can use it if you want your performance in each fold to be calculated by an average of subfolds
% rather than in whole. For example, in this case, instead of estimating the AUC across all 8 runs of each subject, we can estimate the AUC
% within each of the 8 runs and then average them to obtain subject-level performance metric. You may want to do this because there are
% often spurious baseline shifts in estimated activity across runs that can make their alignment poor.
subfold = run;

% now let's evaluate
cvstats = evalTuningParam(cvmdl,'AUC',X,Y,compvec,threshvec,subfold);

% we can now plot to look at the cross-validation performance as a function of number of components (1:25) and threshold (0:.05:1)
% the 3d plot is interactive so feel free to move it around
plot(cvstats)

% so if you're seeing what I'm seeing, the best performance, as indicated by blue dot (Max Perf) should be at 
% threshold 0.1 (10% of voxels left) and at 8 components
% this information is also available in the cvstats structure if you just type it in console
cvstats



%%%%%%%%%%%%%%%%%%% Final Model %%%%%%%%%%%%%%%%%%%%%
% now that we know the best tuning parameter, let's fit the final model using this tuning parameter
% fitting up to 25 components for simplicity. You can specify it to fit up to 8 components if you're short on time.
mdl = TPLS(X,Y);

% see how much covariance between X and Y each PLS component explains (left)
% also see the correlation between each PLS component and Y (right)
subplot(1,2,1); plot(mdl.pctVar); subplot(1,2,2); plot(mdl.scoreCorr)

% now let's extract the whole-brain map
compval = cvstats.compval_best;
threshval = cvstats.threshval_best;
betamap = makePredictor(mdl,compval,threshval);

% voila! you now have a whole-brain predictor
% you can check how many voxels have non-zero coefficients by doing this
sum(betamap~=0)
% to me, it shows 372, which is about 1/10th of all the voxels (since that was our thresholding tuning parameter)

% you can easily use this to predict trials by just multiplying this betamap to each single-trial beta images
% for example:
prediction = X * betamap;
corr(prediction,Y) % 0.63 correlation! In-sample though so not that impressive
scatter(prediction,Y)

% now let's look at the resulting whole-brain predictor
mymap = mask;
mymap(mask(:)==1) = betamap;

% if you have a nifti toolbox, you can save this out into a nifti file and look at it.
% for now, let's look at the slice where there should be motor activity
subplot(1,2,1)
heatmap(flipud(squeeze(mask(:,15,:))'))
subplot(1,2,2)
heatmap(flipud(squeeze(mymap(:,15,:))'))

% it isn't much, but you're looking at a coronal slice of the brain right about where the motor cortex is.
% You can see the left motor cortex has positive coefficients whole the right cortex has negative coefficients.
% That's because our Y variable was whether participant chose the right button (hence left motor cortex)



%%%%%%%%%%%%%%%%%%% short version %%%%%%%%%%%%%%%%%%%%%%%%
% in case you want a version of example that you can easily copy paste to your pipeline, here it is. 4 lines!
clear
clc
load('TPLSdat.mat')
cvmdl = TPLS_cv(X,Y,subj);
cvstats = evalTuningParam(cvmdl,'AUC',X,Y,1:25,0:.05:1,run);
mdl = TPLS(X,Y);
betamap = makePredictor(mdl,cvstats.compval_best,cvstats.threshval_best);