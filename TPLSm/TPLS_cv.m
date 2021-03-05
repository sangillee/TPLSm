classdef TPLS_cv %   Thresholded Partial Least Squares.
    properties
        NComp, numfold, CVfold, cvMdls
    end
    methods
        function TPLScvmdl = TPLS_cv(X,Y,CVfold,NComp,W,nmc)
            % Constructor method for fitting a cross-validation T-PLS model
            %   'X'     : Numerical matrix of predictors. Typically single-trial betas where each column is a voxel and row is observation
            %   'Y'     : Variable to predict. Binary 0 and 1 in case of classification, continuous variable in case of regression
            %   'CVfold': Cross-validation testing fold information. Can either be a vector or a matrix, the latter being more general.
            %             Vector: n-by-1 vector. Each element is a number ranging from 1 ~ numfold to identify which testing fold eachobservation belongs to
            %             Matrix: n-by-numfold matrix. Each column indicates the testing data with 1 and training data as 0.
            %             Example: For leave-one-out CV, Vector would be 1:n, Matrix form would be eye(n)
            %             Matrix form is more general as it can have same trial be in multiple test folds
            %   'NComp' : (Optional) Number of PLS components to compute. Default is 25.
            %   'W'     : (Optional) Observation weights. Optional input. By default, all observations have equal weight.
            %             Can either be a n-by-1 vector or a n-by-nfold matrix where each column is observation weights in that CV fold
            %   'nmc'   : (Optional) 'no mean centering'. See TPLS for more detail.
            
            % input checking
            if nargin<6, nmc = 0; end
            if nargin<5, W = ones(size(Y)); end % by default all observation have equal weights
            if nargin<4, NComp = 25; end % default value
            [TPLScvmdl.CVfold,TPLScvmdl.numfold,W] = TPLScvinputchecking(X,Y,CVfold,NComp,W,nmc);
            
            TPLScvmdl.NComp = NComp;
            TPLScvmdl.cvMdls = cell(TPLScvmdl.numfold,1);
            for i = 1:TPLScvmdl.numfold
                disp(['Fold #',num2str(i)])
                train = TPLScvmdl.CVfold(:,i) == 0;
                TPLScvmdl.cvMdls{i} = TPLS(X(train,:),Y(train),NComp,W(train,i),nmc);
            end
        end
    end
end

% checking input parameters to TPLS_cv to ensure smooth running
function [CVfold,numfold,W] = TPLScvinputchecking(X,Y,CVfold,NComp,W,nmc)
% 1. type checking
assert(all(isnumeric(X)),'X should be numerical matrix');
assert(all(isnumeric(Y)),'Y should be numerical vector');
assert(all(isnumeric(CVfold)),'CVfold should be numeric');
assert(all(isnumeric(NComp)),'NComp should be numeric');
assert(all(isnumeric(W)),'W should be numerical vector');
assert(all(isnumeric(nmc)),'nmc should either be 0 or 1');

% 2. size checking
[n,v] = size(X);
assert(v > 2,'X should have at least 3 columns');
assert(n > 2,'X should have at least 3 observations');
[nY,vY] = size(Y);
assert(n==nY,'X and Y should have same number of rows');
assert(vY==1,'Y should be a column vector');
nCVfold = size(CVfold,1);
assert(n==nCVfold,'X and CVfold should have same number of rows');
[nC,vC] = size(NComp);
assert(nC==1 && vC == 1,'NComp should be a scalar number');
nW = size(W,1);
assert(nW==nY,'W and Y should have same number of rows');
[nn,vn] = size(nmc);
assert(nn==1 && vn == 1,'NComp should be a scalar number');

% 3. nan checking
assert(~any(isnan(X(:))),'NaN found in X');
assert(~any(isnan(Y)),'NaN found in Y');
assert(~any(isnan(CVfold(:))),'NaN found in CVfold');
assert(~isnan(NComp),'NComp is NaN');
assert(~isnan(nmc),'nmc is NaN');

% 4. logic checking
assert( floor(NComp)==NComp && ceil(NComp)==NComp,'NComp should be an integer')
assert( nmc==1 || nmc==0 ,'nmc switch should be either 0 or 1');
[CVfold,numfold] = prepCVfold(CVfold); % convert CVfold into matrix form, if not already
if size(W,2) == 1 % vector form weight
    W = repmat(W,1,numfold);
end
end

% prepare CV fold data into a matrix form, which is more generalizable
function [CVfold,nfold] = prepCVfold(inCVfold)
if size(inCVfold,2) == 1 % vector
    uniqfold = unique(inCVfold);
    nfold = length(uniqfold);
    CVfold = zeros(length(inCVfold),nfold);
    for i = 1:nfold
        CVfold(:,i) = 1.*(inCVfold == uniqfold(i));
    end
elseif size(inCVfold,2) > 1 % matrix
    nfold = size(inCVfold,2);
    CVfold = inCVfold;
    if any(CVfold(:)~=0 & CVfold(:)~=1)
        error('Non-binary element is matrix form CVfold. Perhaps you meant to use vector form?')
    end
else
    error('unexpected size of CVfold')
end
end