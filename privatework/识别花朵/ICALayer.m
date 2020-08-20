function layer = ICALayer(varargin)

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

% Create an internal representation of a ICA layer.
internalLayer = nnet.internal.cnn.layer.ICA(inputArguments.Name);

% Pass the internal layer to a  function to construct a user visible ReLU
% layer.
layer = nnet.cnn.layer.ICALayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser);
end

function p = iCreateParser()
p = inputParser;
defaultName = '';
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.Name = char(p.Results.Name); % make sure strings get converted to char vectors
end
