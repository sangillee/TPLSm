classdef TPLS
    properties
        NComp, MX, MY, scoreCorr, pctVar, betamap, threshmap
    end
    methods
        function TPLSmdl = TPLS(X,Y,NComp,W,nmc)
            % Constructor method for fitting a T-PLS model with given data X and Y.
            %   'X'    : Numerical matrix of predictors. Typically single-trial betas where each column is a voxel and row is observation
            %   'Y'    : Variable to predict. Binary 0 and 1 in case of classification, continuous variable in case of regression
            %   'NComp': (Optional) Number of PLS components to compute. Default is 25.
            %   'W'    : (Optional) Observation weights. Optional input. By default, all observations have equal weight.
            %   'nmc'  : (Optional) 'no mean centering'. Default is 0. If 1, T-PLS will skip mean-centering.
            %            This option is only provided in case you already mean-centered the data and want to save some memory usage.
            
            % input checking
            if nargin<3, NComp = 25; end
            if nargin<4, W = ones(size(Y)); end
            if nargin<5, nmc = 0; end
            [n,v,W] = TPLSinputchecking(X,Y,NComp,W,nmc);
            
            % Mean-Center variables as needed by SIMPLS algorithm
            TPLSmdl.NComp = NComp; TPLSmdl.MX = W'*X; TPLSmdl.MY = W'*Y; % calculating weighted means of X and Y
            if nmc == 0 % do mean centering
                X = bsxfun(@minus, X, TPLSmdl.MX); Y = Y - TPLSmdl.MY; % subtract means
            elseif any(abs(TPLSmdl.MX) > 1e-04)
                warning('X does not seem to be mean-centered. Results may not be valid')
            end
            
            % allocate memories
            TPLSmdl.scoreCorr = nan(NComp,1); TPLSmdl.betamap = nan(v,NComp); TPLSmdl.threshmap = [0.5 .* ones(v,1), nan(v,NComp-1)]; % output variables
            B = nan(NComp,1); P2 = nan(n,NComp); C = nan(v,NComp); sumC2 = zeros(v,1); r = Y; V = nan(v,NComp); % interim variables
            WYT = (W.*Y)'; WTY2 = W'*Y.^2; WT = W'; W2 = W.^2; % often-used variables in calculation
            
            % perform Arthur-modified SIMPLS algorithm
            Cov = (WYT*X)'; % weighted covariance
            for i = 1:NComp
                disp(['Calculating Comp #',num2str(i)])
                P = X*Cov; % this is the component, before normalization
                norm_P = sqrt(WT*P.^2); % weighted standard deviation of component as normalization constant
                P = P ./ norm_P; B(i) = (norm(Cov)^2)/norm_P; C(:,i) = Cov ./ norm_P; % normalize component, beta, and back-projection coefficient
                
                % Update the orthonormal basis with modified Gram Schmidt
                vi = ((W.*P)'*X)'; % weighted covariance between X and current component
                vi = vi - V(:,1:i-1)*(V(:,1:i-1)'*vi); % orthogonalize with regards to previous components
                vi = vi ./ norm(vi); V(:,i) = vi; % normalize and add to orthonormal basis matrix
                Cov = Cov - vi*(vi'*Cov); Cov = Cov - V(:,1:i)*(V(:,1:i)'*Cov); % Deflate Covariance using the orthonormal basis matrix
                
                % back-projection
                TPLSmdl.betamap(:,i) = C(:,1:i)*B(1:i); % back-projection of coefficients
                sumC2 = sumC2+C(:,i).^2; P2(:,i) = P.^2; r = r - P*B(i); % some variables that will facilitate computation later
                if i > 1 % no need to calculate threshold for first component
                    se = sqrt(P2(:,1:i)'*(W2.*(r.^2))); %Huber-White Sandwich estimator (assume no small T bias)
                    TPLSmdl.threshmap(:,i) = abs((C(:,1:i)*(B(1:i)./se))./sqrt(sumC2)); % absolute value of back-projected z-statistics
                end
            end
            TPLSmdl.threshmap(:,2:end) = (v-tiedrank(TPLSmdl.threshmap(:,2:end)))./v; % convert into thresholds between 0 and 1
            TPLSmdl.pctVar = B.^2 ./ WTY2; % Compute the percent of variance of Y each component explains
            TPLSmdl.scoreCorr = sqrt(TPLSmdl.pctVar); % weighted correlation between Y and current component
        end
        
        function [betamap,bias] = makePredictor(TPLSmdl,compval,threshval)
            % Method for extracting the T-PLS predictor at a given compval and threshval
            %   'betamap'  : T-PLS predictor coefficient
            %   'bias'     : Intercept for T-PLS model.
            %   'TPLSmdl'  : Object created by TPLS constructor function
            %   'compval'  : Vector of number of components to use in final predictor
            %                (e.g., [3,5] will give you two betamaps, one with 3 components and one with 5 components
            %   'threshval': Single number of thresholding value to use in final predictor.
            %                (e.g., 0.1 will yield betamap where only 10% of coefficients will be non-zero)
            assert(all(isnumeric(compval)) && length(compval)==length(compval(:)),'compval should be a numerical vector');
            assert( max(compval)<=TPLSmdl.NComp, 'compval should only include values small than number of components used for TPLS model');
            assert(all(isnumeric(threshval)) && length(threshval)==1 && 0<=threshval && threshval<=1,'threshval should be a single number between 0 and 1');
            if threshval == 0
                betamap = TPLSmdl.betamap(:,compval) .*0;
            else
                betamap = TPLSmdl.betamap(:,compval) .* (TPLSmdl.threshmap(:,compval)<=threshval); % finalized predictor map at select components
            end
            bias = TPLSmdl.MY-TPLSmdl.MX*betamap; % post-fitting of bias
        end
        
        function score = predict(TPLSmdl,compval,threshval,testX)
            % Method for making predictions on a testing dataset testX
            %   'score'    : Prediction scores on a testing dataset
            %   'TPLSmdl'  : Object created by TPLS constructor function
            %   'compval'  : Vector of number of components to use in final predictor
            %   'threshval': Single number of thresholding value to use in final predictor.
            %   'testX'    : Data to be predicted
            assert(all(isnumeric(testX)),'testX should be numerical matrix');
            assert(~all(isnan(testX(:))),'NaN found in X');
            [threshbetamap,bias] = makePredictor(TPLSmdl,compval,threshval);
            score = bsxfun(@plus,bias,testX*threshbetamap);
        end
    end
end

% checking input parameters to TPLS to ensure smooth running
function [n,v,W] = TPLSinputchecking(X,Y,NComp,W,nmc)
% 1. type checking
assert(all(isnumeric(X)),'X should be numerical matrix');
assert(all(isnumeric(Y)),'Y should be numerical vector');
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
[nC,vC] = size(NComp);
assert(nC==1 && vC == 1,'NComp should be a scalar number');
[nW,vW] = size(W);
assert(nW==nY,'W and Y should have same number of rows');
assert(vW==1,'W should be a column vector');
[nn,vn] = size(nmc);
assert(nn==1 && vn == 1,'NComp should be a scalar number');

% 3. nan checking
assert(~any(isnan(X(:))),'NaN found in X');
assert(~any(isnan(Y)),'NaN found in Y');
assert(~isnan(NComp),'NComp is NaN');
assert(~any(isnan(W)),'NaN found in W');
assert(~isnan(nmc),'nmc is NaN');

% 4. logic checking
assert( floor(NComp)==NComp && ceil(NComp)==NComp,'NComp should be an integer')
assert( all(W>0) ,'weights should be non-negative');
assert( nmc==1 || nmc==0 ,'nmc switch should be either 0 or 1');
W = W./sum(W); % normalize weight sum to 1
end