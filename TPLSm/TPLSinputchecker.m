function TPLSinputchecker(input,name,type,maxval,minval,variation,integercheck)
if nargin < 7, integercheck = 0; end
if nargin < 6, variation = 0; end
if nargin < 5, minval = []; end
if nargin < 4, maxval = []; end
if nargin < 3, type = []; end

assert(all(isnumeric(input)),[name,' should be numeric']); % numeric check
assert(~any(isnan(input(:))),['NaN found in ',name]); % nan check
assert(~any(~isfinite(input(:))),['Non finite value found in ',name]); % inf check

if ~isempty(type)
    [n,v] = size(input);% size check
    switch type
        case 'mat'
            assert(v > 2,[name,' should have at least 3 columns']);
            assert(n > 2,[name,' should have at least 3 observations']);
        case 'vec'
            assert(length(input)==length(input(:)),[name,' should be a vector'])
        case 'colvec'
            assert(v==1,[name,' should be a column vector']);
        case 'scalar'
            assert(n==1 && v==1,[name,' should be a scalar']);
        otherwise
            error('unexpected input type checking requested')
    end
end

if ~isempty(maxval)
    assert(all(input(:)<=maxval),[name,' should be less than or equal to ',num2str(maxval)])
end
if ~isempty(minval)
    assert(all(input(:)>=minval),[name,' should be greater than or equal to ',num2str(minval)])
end
if variation == 1
    assert( ~any(std(input)==0),['There is no variation in ',name])
end
if integercheck == 1
    input = input(:);
    for i = 1:length(input)
        assert(floor(input(i))==ceil(input(i)),[name,' should be integer'])
    end
end
end