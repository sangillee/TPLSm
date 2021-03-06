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
            if nargin<5, W = ones(size(Y)); end
            if nargin<4, NComp = 25; end
            TPLSinputchecker(X,'X','mat',[],[],1)
            TPLSinputchecker(Y,'Y','colvec',[],[],1)
            TPLSinputchecker(CVfold,'CVfold')
            TPLSinputchecker(NComp,'NComp','scalar',[],1,0,1)
            TPLSinputchecker(W,'W','colvec',Inf,0)
            TPLSinputchecker(nmc,'nmc','scalar')
            assert(size(X,1)==length(Y) && size(X,1)==size(CVfold,1) && size(X,1) == length(W),'X, Y, W, and CV fold should have same number of rows');
            [TPLScvmdl.CVfold,TPLScvmdl.numfold] = prepCVfold(CVfold); % convert CVfold into matrix form, if not already
            if size(W,2) == 1, W = repmat(W,1,numfold); end % convert into matrix form, if not already
            
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

% prepare CV fold data into a matrix form, which is more generalizable
function [CVfold,nfold] = prepCVfold(inCVfold)
if size(inCVfold,2) == 1 % vector
    uniqfold = unique(inCVfold); nfold = length(uniqfold);
    CVfold = zeros(length(inCVfold),nfold);
    for i = 1:nfold
        CVfold(:,i) = 1.*(inCVfold == uniqfold(i));
    end
elseif size(inCVfold,2) > 1 % matrix
    nfold = size(inCVfold,2); CVfold = inCVfold;
    if any(CVfold(:)~=0 & CVfold(:)~=1)
        error('Non-binary element is matrix form CVfold. Perhaps you meant to use vector form?')
    end
else
    error('unexpected size of CVfold')
end
end