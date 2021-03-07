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
            TPLSinputchecker(X,'X','mat',[],[],1); [n,v] = size(X);
            TPLSinputchecker(Y,'Y','colvec',[],[],1)
            TPLSinputchecker(NComp,'NComp','scalar',[],1,0,1)
            TPLSinputchecker(W,'W','colvec',[],0)
            TPLSinputchecker(nmc,'nmc','scalar')
            assert(n==length(Y) && n==length(W),'X, Y, and W should have equal number of rows')
            W = W./sum(W); % normalize weight sum to 1
            
            % Mean-Center variables as needed by SIMPLS algorithm
            TPLSmdl.NComp = NComp; TPLSmdl.MX = W'*X; TPLSmdl.MY = W'*Y; % calculating weighted means of X and Y
            if nmc == 0 % do mean centering
                X = bsxfun(@minus, X, TPLSmdl.MX); Y = Y - TPLSmdl.MY; % subtract means
            elseif any(abs(TPLSmdl.MX) > 1e-04)
                warning('Skipped mean centering, but X does not seem to be mean-centered. Results may be invalid')
            end
            
            % allocate memories
            TPLSmdl.pctVar = nan(NComp,1); TPLSmdl.scoreCorr = nan(NComp,1); % percent of variance of Y each component explains, weighted correlation between Y and current component
            TPLSmdl.betamap = zeros(v,NComp); TPLSmdl.threshmap = 0.5.*ones(v,NComp); % output variables
            B = nan(NComp,1); P2 = nan(n,NComp); C = nan(v,NComp); sumC2 = zeros(v,1); r = Y; V = nan(v,NComp); % interim variables
            WYT = (W.*Y)'; WTY2 = W'*Y.^2; WT = W'; W2 = W.^2; % often-used variables in calculation
            
            % perform Arthur-modified SIMPLS algorithm
            Cov = (WYT*X)'; normCov = norm(Cov); % weighted covariance
            for i = 1:NComp
                disp(['Calculating Comp #',num2str(i)])
                P = X*Cov; norm_P = sqrt(WT*P.^2); % this is the component and its weighted stdev
                P = P ./ norm_P; B(i) = (normCov^2)/norm_P; C(:,i) = Cov ./ norm_P; % normalize component, beta, and back-projection coefficient
                TPLSmdl.pctVar(i) = (B(i)^2)/WTY2; TPLSmdl.scoreCorr(i) = sqrt(TPLSmdl.pctVar(i));
                
                % Update the orthonormal basis with modified Gram Schmidt
                vi = ((W.*P)'*X)'; % weighted covariance between X and current component
                vi = vi - V(:,1:i-1)*(V(:,1:i-1)'*vi); % orthogonalize with regards to previous components
                vi = vi ./ norm(vi); V(:,i) = vi; % normalize and add to orthonormal basis matrix
                Cov = Cov - vi*(vi'*Cov); Cov = Cov - V(:,1:i)*(V(:,1:i)'*Cov); normCov = norm(Cov); % Deflate Covariance using the orthonormal basis matrix
                
                % back-projection
                TPLSmdl.betamap(:,i) = C(:,1:i)*B(1:i); % back-projection of coefficients
                sumC2 = sumC2+C(:,i).^2; P2(:,i) = P.^2; r = r - P*B(i); % some variables that will facilitate computation later
                if i > 1 % no need to calculate threshold for first component
                    se = sqrt(P2(:,1:i)'*(W2.*(r.^2))); %Huber-White Sandwich estimator (assume no small T bias)
                    abszstat = abs((C(:,1:i)*(B(1:i)./se))./sqrt(sumC2)); % absolute value of back-projected z-statistics
                    TPLSmdl.threshmap(:,i) = (v-tiedrank(abszstat))./v; % convert into thresholds between 0 and 1
                end
                
                % check if there's enough covariance to milk
                if normCov < 10*eps
                    disp('All Covariance between X and Y has been explained. Stopping...'); break;
                elseif TPLSmdl.pctVar(i) < 10*eps % Proportion of Y variance explained is small
                    disp('New PLS component does not explain more covariance. Stopping...'); break;
                end
            end
        end
        
        function [betamap,bias] = makePredictor(TPLSmdl,compval,threshval)
            % Method for extracting the T-PLS predictor at a given compval and threshval
            %   'betamap'  : T-PLS predictor coefficient
            %   'bias'     : Intercept for T-PLS model.
            %   'TPLSmdl'  : Object created by TPLS constructor function
            %   'compval'  : Vector of number of components to use in final predictor
            %                (e.g., [3,5] will give you two betamaps, one with 3 components and one with 5 components
            %   'threshval': Scalar thresholding value to use in final predictor.
            %                (e.g., 0.1 will yield betamap where only 10% of coefficients will be non-zero)
            TPLSinputchecker(compval,'compval','vec',TPLSmdl.NComp,1,0,1)
            TPLSinputchecker(threshval,'threshval','scalar',1,0)
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
            %   'testX'    : Data to be predicted. In same orientation as X
            TPLSinputchecker(testX,'testX')
            [threshbetamap,bias] = makePredictor(TPLSmdl,compval,threshval);
            score = bsxfun(@plus,bias,testX*threshbetamap);
        end
    end
end