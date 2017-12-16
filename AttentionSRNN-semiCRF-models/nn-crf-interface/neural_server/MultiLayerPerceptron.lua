local MultiLayerPerceptron, parent = torch.class('MultiLayerPerceptron', 'AbstractNeuralNetwork')

function MultiLayerPerceptron:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function MultiLayerPerceptron:initialize(javadata, ...)
    local gpuid = self.gpuid
    self.data = {}
    local data = self.data
    data.numLabels = javadata:get("numLabels")
    data.rawInputs = listToTable(javadata:get("nnInputs"))
    local isTraining = javadata:get("isTraining")
    self.embeddingSize = 100
    if isTraining then
        self.x = self:prepare_input()
    else
        self.testInput = self:prepare_input()
    end
    if self.net == nil and isTraining then
        -- means is initialized process and we don't have the input yet.
        self:createNetwork()
        print(self.net)
    end

    local outputAndGradOutputPtr = {... }
    if #outputAndGradOutputPtr > 0 then
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
        return self:obtainParams()
    end
end

function MultiLayerPerceptron:createNetwork()
    local data = self.data
    local gpuid = self.gpuid
    self.numLabels = data.numLabels
    local mlp = nn.Sequential()
    local hiddenSize = 50
    local lt = nn.LookupTable(self.vocabSize, self.embeddingSize)
    mlp:add(lt)
    mlp:add(nn.Linear(self.embeddingSize, hiddenSize))
    mlp:add(nn.Tanh())
    mlp:add(nn.Linear(hiddenSize, self.numLabels))
    if gpuid >= 0 then
        mlp:cuda()
    end
    self.net = mlp
end


function MultiLayerPerceptron:obtainParams()
    --make sure we will not replace this variable
    self.params, self.gradParams = self.net:getParameters()
    print("Number of parameters: " .. self.params:nElement())
    if self.doOptimization then
        self:createOptimizer()
        -- no return array if optim is done here
    else
        if self.gpuid >= 0 then
            -- since the the network is gpu network.
            self.paramsDouble = self.params:double()
            self.paramsDouble:retain()
            self.paramsPtr = torch.pointer(self.paramsDouble)
            self.gradParamsDouble = self.gradParams:double()
            self.gradParamsDouble:retain()
            self.gradParamsPtr = torch.pointer(self.gradParamsDouble)
            return self.paramsPtr, self.gradParamsPtr
        else
            self.params:retain()
            self.paramsPtr = torch.pointer(self.params)
            self.gradParams:retain()
            self.gradParamsPtr = torch.pointer(self.gradParams)
            return self.paramsPtr, self.gradParamsPtr
        end
    end
end

function MultiLayerPerceptron:createOptimizer()
    local data = self.data

    -- set optimizer. If nil, optimization is done by caller.
    print(string.format("Optimizer: %s", data.optimizer))
    self.doOptimization = data.optimizer ~= nil and data.optimizer ~= 'none'
    if self.doOptimization == true then
        if data.optimizer == 'sgd' then
            self.optimizer = optim.sgd
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adagrad' then
            self.optimizer = optim.adagrad
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adam' then
            self.optimizer = optim.adam
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adadelta' then
            self.optimizer = optim.adadelta
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'lbfgs' then
            self.optimizer = optim.lbfgs
            self.optimState = {tolFun=10e-10, tolX=10e-16}
        end
    end
end

function MultiLayerPerceptron:prepare_input()
    local gpuid = self.gpuid
    local data = self.data

    local rawInputs = data.rawInputs
    local results = torch.IntTensor(#rawInputs)
    self:buildVocab(rawInputs)
    for i=1,#rawInputs do
        local tok = rawInputs[i]
        local tok_id = self.word2idx[tok]
        if tok_id == nil then
            tok_id = self.word2idx['<UNK>']  
        end
        results[i] = tok_id
    end
    if gpuid >= 0 then result:cuda() end
    return results
end

function printTable(table)
    local count = 0
    for i,k in pairs(table) do print(i .. " " .. k) end
    return count
end

function MultiLayerPerceptron:buildVocab(wordInputs)
    if self.word2idx ~= nil then
        --means the vocabulary is already created
        return 
    end
    self.word2idx = {}
    self.word2idx['<UNK>'] = 1
    self.vocabSize = 2
    for i=1,#wordInputs do
        local token = wordInputs[i]
        local tok_id = self.word2idx[token]
        if tok_id == nil then
            self.word2idx[token] = self.vocabSize
            self.vocabSize = self.vocabSize + 1
        end
    end
    print("number of unique words:" .. self.vocabSize)
end

function MultiLayerPerceptron:forward(isTraining, batchInputIds)
    if self.gpuid >= 0 and not self.doOptimization and isTraining then
        self.params:copy(self.paramsDouble:cuda())
    end
    local input_x = self:getForwardInput(isTraining, batchInputIds)
    local output = self.net:forward(input_x)
    if not self.outputPtr:isSameSizeAs(output) then
        self.outputPtr:resizeAs(output)
    end
    self.outputPtr:copy(output)
end

function MultiLayerPerceptron:getForwardInput(isTraining, batchInputIds)
    if isTraining then
        if batchInputIds ~= nil then
            batchInputIds:add(1) -- because the sentence is 0 indexed\
            self.batchInput = self.x:index(1, batchInputIds)
            return self.batchInput
        else
            return self.x
        end
    else
        return self.testInput
    end
end

function MultiLayerPerceptron:backward()
    self.gradParams:zero()
    local gradOutputTensor = self.gradOutputPtr
    if self.gpuid >= 0 then
        gradOutputTensor = gradOutputTensor:cuda()
    end
    self.net:backward(self.x, gradOutputTensor)
    if self.doOptimization then
        self.optimizer(self.feval, self.params, self.optimState)
    end
end

