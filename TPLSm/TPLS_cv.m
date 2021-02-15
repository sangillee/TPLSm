classdef TPLS_cv %   Thresholded Partial Least Squares.
    properties
        NComp, numfold, testfold, cvMdls
    end
    methods
        function TPLScvmdl = TPLS_cv(X,Y,CVfold,NComp,W,nmc)
            if nargin<6, nmc = 0; end
            if nargin<5, W = ones(size(Y)); end % by default all observation have equal weights
            if nargin<4, NComp = 50; end % default value
            
            uniqfold = unique(CVfold);
            TPLScvmdl.NComp = NComp; 
            TPLScvmdl.numfold = length(uniqfold);
            TPLScvmdl.testfold = nan(size(CVfold));
            TPLScvmdl.cvMdls = cell(TPLScvmdl.numfold,1);
            for i = 1:TPLScvmdl.numfold
                disp(['Fold #',num2str(i)])
                train = CVfold ~= uniqfold(i);
                TPLScvmdl.testfold(~train) = i;
                TPLScvmdl.cvMdls{i} = TPLS(X(train,:),Y(train),NComp,W(train),nmc);
            end
        end
    end
end