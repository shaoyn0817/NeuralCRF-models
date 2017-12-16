 
--include 'GPUUtils.lua'
local SimpleBiLSTM, parent = torch.class('SimpleBiLSTMSRNN', 'AbstractNeuralNetwork')

function SimpleBiLSTM:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function SimpleBiLSTM:initialize(javadata, ...)
    self.data = {}
    local data = self.data
    data.sentences = listToTable(javadata:get("nnInputs"))
    data.hiddenSize = javadata:get("hiddenSize")
    data.optimizer = javadata:get("optimizer")
    data.learningRate = javadata:get("learningRate")
    data.clipping = javadata:get("clipping")
    self.numLabels = javadata:get("numLabels")
    data.embedding = javadata:get("embedding")
    local modelPath = javadata:get("nnModelFile")
    local isTraining = javadata:get("isTraining")
    data.isTraining = isTraining

    if isTraining then
        self.x = self:prepare_input()
        self.numSent = #data.sentences
    end

    if self.net == nil and isTraining then
        -- means is initialized process and we don't have the input yet.
        self:createNetwork()
        print(self.net)
    end
    if self.net == nil then 
        self:load_model(modelPath)
    end


    if not isTraining then 
        self.testInput = self:prepare_input()
    end
    self.output = torch.Tensor()
    self.x1Tab = {}
    self.x1 = torch.LongTensor()
    self.x2Tab = {}
    self.x2 = torch.LongTensor()
    if self.gpuid >= 0 then
        self.x1 = self.x1:cuda()
        self.x2 = self.x2:cuda()
    end
    self.gradOutput = {}
    local outputAndGradOutputPtr = {... }
    if #outputAndGradOutputPtr > 0 then
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
        return self:obtainParams()
    end
end

--The network is only created once is used.
function SimpleBiLSTM:createNetwork()
    local data = self.data

    local hiddenSize = data.hiddenSize

    local sharedLookupTable
    if data.embedding ~= nil then
        if data.embedding == 'glove' then
            sharedLookupTable = loadGlove(self.idx2word, hiddenSize, true)
        else -- unknown/no embedding, defaults to random init
            print ("Not using any embedding..")
            sharedLookupTable = nn.LookupTableMaskZero(self.vocabSize, hiddenSize)
        end
    else
        print ("Not using any embedding..")
        sharedLookupTable = nn.LookupTableMaskZero(self.vocabSize, hiddenSize)
    end


fwd1 = nn.Sequential()
       :add(sharedLookupTable)
       :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))

fwdSeq1 = nn.Sequencer(fwd1)

local bwd1, bwdSeq1
if bidirection then
    bwd1 = nn.Sequential()
       :add(sharedLookupTable:sharedClone())
       :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))
           
    bwdSeq1 = nn.Sequential()
        :add(nn.Sequencer(bwd1))
        :add(nn.ReverseTable())
end

parallel1 = nn.ParallelTable()
parallel1:add(fwdSeq1)
    
if bidirection then
    parallel1:add(bwdSeq1)
end

brnn1 = nn.Sequential()
   :add(parallel1)
   :add(nn.JoinTable(3))

mergeHiddenSize = hiddenSize
if bidirection then
    mergeHiddenSize = 2 * hiddenSize
end

net = nn.Sequential():add(brnn1)



fwd2 = nn.Sequencer(nn.FastLSTM(mergeHiddenSize, hiddenSize):maskZero(1))
local bwd2
if bidirection then
    bwd2 = nn.Sequential()
            :add(nn.Sequencer(nn.FastLSTM(mergeHiddenSize, hiddenSize):maskZero(1)))
            :add(nn.ReverseTable())
    end

 
map = nn.MapTable()
parallel2 = nn.ConcatTable():add(fwd2)


    
if bidirection then
    parallel2:add(bwd2)
end
map:add(parallel2)
lstm2 = nn.Sequential():add(map):add(nn.MapTable():add(nn.JoinTable(3)))

brnn2 = nn.Sequential()
   :add(lstm2)



attenmlp =nn.Sequencer(nn.Sequential():add(nn.Linear(mergeHiddenSize, 1)):add(nn.ReLU()))
atten = nn.Sequential():add(nn.Sequential():add(attenmlp):add(nn.SoftMax()))

atl = nn.MapTable():add(nn.ConcatTable():add(nn.Sequential():add(atten):add(nn.Squeeze(3))):add(nn.Identity()))
final = nn.MapTable():add(nn.Sequential():add(nn.MapTable():add(nn.Transpose({1,2})))
                                         :add(nn.MixtureTable())
                                         :add(nn.Linear(mergeHiddenSize,2))
                                         :add(nn.ReLU()))

--net1  compute span length 1
index1 = nn.ConcatTable()
for i = 1, 100, 1 do   -----------------------
    l = torch.LongTensor({i})
    index1:add(nn.Index({1, l}))
end          
self.net1 = nn.Sequential():add(net):add(index1):add(brnn2):add(atl):add(final)

--numsen timestep hiddensize
--seqlen x batchsize x featsize
index2 = nn.ConcatTable()
for i = 1, 99, 1 do
    l = torch.LongTensor({i, i+1})
    index2:add(nn.Index({1, l}))
end          
self.net2 = nn.Sequential():add(net):add(index2):add(brnn2):add(atl):add(final)

index3 = nn.ConcatTable()
for i = 1, 98, 1 do
    l = torch.LongTensor({i, i+1, i+2})
    index3:add(nn.Index({1, l}))
end          
self.net3 = nn.Sequential():add(net):add(index3):add(brnn2):add(atl):add(final)

index4 = nn.ConcatTable()
for i = 1, 97, 1 do
    l = torch.LongTensor({i, i+1, i+2, i+3})
    index4:add(nn.Index({1, l}))
end          
self.net4 = nn.Sequential():add(net):add(index4):add(brnn2):add(atl):add(final)

index5 = nn.ConcatTable()
for i = 1, 96, 1 do
    l = torch.LongTensor({i, i+1, i+2, i+3, i+4})
    index5:add(nn.Index({1, l}))
end
self.net5 = nn.Sequential():add(net):add(index5):add(brnn2):add(atl):add(final)

index6 = nn.ConcatTable()
for i = 1, 95, 1 do
    l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5})
    index6:add(nn.Index({1, l}))
end          
self.net6 = nn.Sequential():add(net):add(index6):add(brnn2):add(atl):add(final)

index7 = nn.ConcatTable()
for i = 1, 94, 1 do
    l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6})
    index7:add(nn.Index({1, l}))
end
self.net7 = nn.Sequential():add(net):add(index7):add(brnn2):add(atl):add(final)

index8 = nn.ConcatTable()
for i = 1, 93, 1 do
    l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6, i+7})
    index8:add(nn.Index({1, l}))
end          
self.net8 = nn.Sequential():add(net):add(index8):add(brnn2):add(atl):add(final)

index9 = nn.ConcatTable()
for i = 1, 92, 1 do
    l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8})
    index9:add(nn.Index({1, l}))
end
self.net9 = nn.Sequential():add(net):add(index9):add(brnn2):add(atl):add(final)

index10 = nn.ConcatTable()
for i = 1, 91, 1 do
    l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9})
    index10:add(nn.Index({1, l}))
end          
self.net10 = nn.Sequential():add(net):add(index10):add(brnn2):add(atl):add(final)

    
if self.gpuid >=0 then 
    self.net1:cuda() 
    self.net2:cuda() 
    self.net3:cuda() 
    self.net4:cuda() 
    self.net5:cuda() 
    self.net6:cuda() 
    self.net7:cuda() 
    self.net8:cuda() 
    self.net9:cuda() 
    self.net10:cuda() 
end




end

function SimpleBiLSTM:obtainParams()
    --make sure we will not replace this variable
    self.params, self.gradParams = self.net1:getParameters()
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

function SimpleBiLSTM:createOptimizer()
    local data = self.data
    -- set optimizer. If nil, optimization is done by caller.
    print(string.format("Optimizer: %s", data.optimizer))
    self.doOptimization = data.optimizer ~= nil and data.optimizer ~= 'none'
    if self.doOptimization == true then
        if data.optimizer == 'sgd_normal' then
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
        elseif data.optimizer == 'sgd' then
            --- with gradient clipping
            self.optimizer = sgdgc
            self.optimState = {learningRate=data.learningRate, clipping=data.clipping}
        end
    end
end

function SimpleBiLSTM:forward(isTraining, batchInputIds)
    if self.gpuid >= 0 and not self.doOptimization and isTraining then
        self.params:copy(self.paramsDouble:cuda())
    end
    local nnInput = self:getForwardInput(isTraining, batchInputIds)
    local output_table1 = self.net1:forward(nnInput)
    local output_table2 = self.net2:forward(nnInput)
    local output_table3 = self.net3:forward(nnInput)
    local output_table4 = self.net4:forward(nnInput)
    local output_table5 = self.net5:forward(nnInput)
    local output_table6 = self.net6:forward(nnInput)
    local output_table7 = self.net7:forward(nnInput)
    local output_table8 = self.net8:forward(nnInput)
    local output_table9 = self.net9:forward(nnInput)
    local output_table10 = self.net10:forward(nnInput)
    if self.gpuid >= 0 then
        nn.utils.recursiveType(output_table1, 'torch.DoubleTensor')
        nn.utils.recursiveType(output_table2, 'torch.DoubleTensor')
        nn.utils.recursiveType(output_table3, 'torch.DoubleTensor')
        nn.utils.recursiveType(output_table4, 'torch.DoubleTensor')
        nn.utils.recursiveType(output_table5, 'torch.DoubleTensor')
        nn.utils.recursiveType(output_table6, 'torch.DoubleTensor')
        nn.utils.recursiveType(output_table7, 'torch.DoubleTensor')
        nn.utils.recursiveType(output_table8, 'torch.DoubleTensor')
        nn.utils.recursiveType(output_table9, 'torch.DoubleTensor')
        nn.utils.recursiveType(output_table10,'torch.DoubleTensor')
    end 
    --- this is to converting the table into tensor.
    out1 = torch.cat(torch.Tensor(), output_table1, 1)
    out2 = torch.cat(torch.Tensor(), output_table2, 1)
    out3 = torch.cat(torch.Tensor(), output_table3, 1)
    out4 = torch.cat(torch.Tensor(), output_table4, 1)
    out5 = torch.cat(torch.Tensor(), output_table5, 1)
    out6 = torch.cat(torch.Tensor(), output_table6, 1)
    out7 = torch.cat(torch.Tensor(), output_table7, 1)
    out8 = torch.cat(torch.Tensor(), output_table8, 1)
    out9 = torch.cat(torch.Tensor(), output_table9, 1)
    out10 = torch.cat(torch.Tensor(), output_table10, 1)
    
    --original shape is a table 99 tensor of shape  batchsize*numoutput
    self.output = torch.cat({out1, out2, out3, out4, out5, out6, out7, out8, out9, out10})

    
    if not self.outputPtr:isSameSizeAs(self.output) then
        self.outputPtr:resizeAs(self.output)
    end
    self.outputPtr:copy(self.output)
end

function SimpleBiLSTM:getForwardInput(isTraining, batchInputIds)
    if isTraining then
        if batchInputIds ~= nil then
            batchInputIds:add(1) -- because the sentence is 0 indexed.
            self.batchInputIds = batchInputIds
            self.x1 = torch.cat(self.x1, self.x[1], 2):index(1, batchInputIds)
            self.x1 = self.x1:view(self.x1:size(2), self.x1:size(1)):view(-1)
            torch.split(self.x1Tab, self.x1, batchInputIds:size(1), 1)
            self.x2 = torch.cat(self.x2, self.x[2], 2):index(1, batchInputIds)
            self.x2 = self.x2:resize(self.x2:size(2), self.x2:size(1)):view(-1)
            torch.split(self.x2Tab, self.x2, batchInputIds:size(1), 1)

            self.batchInput = {self.x1Tab, self.x2Tab}
            return self.batchInput
        else
            return self.x
        end
    else
        return self.testInput
    end
end

function SimpleBiLSTM:getBackwardInput()
    if self.batchInputIds ~= nil then
        return self.batchInput
    else
        return self.x
    end
end

function SimpleBiLSTM:getBackwardSentNum()
    if self.batchInputIds ~= nil then
        return self.batchInputIds:size(1)
    else
        return self.numSent
    end
end

function SimpleBiLSTM:backward()
    self.gradParams:zero()
    local gradOutputTensor = self.gradOutputPtr
    local backwardInput = self:getBackwardInput()  --since backward only happen in training
    local backwardSentNum = self:getBackwardSentNum()
    --print(gradOutputTensor)
    
    
    o1 = gradOutputTensor:narrow(1, 1, 100)
    o2 = gradOutputTensor:narrow(1, 101, 199)
    o3 = gradOutputTensor:narrow(1, 200, 297)
    o4 = gradOutputTensor:narrow(1, 298, 394)
    o5 = gradOutputTensor:narrow(1, 395, 490)
    o6 = gradOutputTensor:narrow(1, 491, 585)
    o7 = gradOutputTensor:narrow(1, 586, 679)
    o8 = gradOutputTensor:narrow(1, 680, 772)
    o9 = gradOutputTensor:narrow(1, 773, 864)
    o10 = gradOutputTensor:narrow(1, 865, 955)

    torch.split(self.gradOutput1, o1, backwardSentNum, 1)
    torch.split(self.gradOutput2, o2, backwardSentNum, 1)
    torch.split(self.gradOutput3, o3, backwardSentNum, 1)
    torch.split(self.gradOutput4, o4, backwardSentNum, 1)
    torch.split(self.gradOutput5, o5, backwardSentNum, 1)
    torch.split(self.gradOutput6, o6, backwardSentNum, 1)
    torch.split(self.gradOutput7, o7, backwardSentNum, 1)
    torch.split(self.gradOutput8, o8, backwardSentNum, 1)
    torch.split(self.gradOutput9, o9, backwardSentNum, 1)
    torch.split(self.gradOutput10, o10, backwardSentNum, 1)

    if self.gpuid >= 0 then
        nn.utils.recursiveType(self.gradOutput1, 'torch.CudaTensor')
        nn.utils.recursiveType(self.gradOutput2, 'torch.CudaTensor')
        nn.utils.recursiveType(self.gradOutput3, 'torch.CudaTensor')
        nn.utils.recursiveType(self.gradOutput4, 'torch.CudaTensor')
        nn.utils.recursiveType(self.gradOutput5, 'torch.CudaTensor')
        nn.utils.recursiveType(self.gradOutput6, 'torch.CudaTensor')
        nn.utils.recursiveType(self.gradOutput7, 'torch.CudaTensor')
        nn.utils.recursiveType(self.gradOutput8, 'torch.CudaTensor')
        nn.utils.recursiveType(self.gradOutput9, 'torch.CudaTensor')
        nn.utils.recursiveType(self.gradOutput10, 'torch.CudaTensor')
    end
    self.net:backward(backwardInput, self.gradOutput)
    if self.doOptimization then
        self.optimizer(self.feval, self.params, self.optimState)
    else
        if self.gpuid >= 0 then
            self.gradParamsDouble:copy(self.gradParams:double())
        end
    end
    
end

function SimpleBiLSTM:prepare_input()
    local data = self.data

    local sentences = data.sentences
    local sentence_toks = {}
    local maxLen = 0
    for i=1,#sentences do
        local tokens = stringx.split(sentences[i]," ")
        table.insert(sentence_toks, tokens)
        if #tokens > maxLen then
            maxLen = #tokens
        end
    end

    --note that inside if the vocab is already created
    --just directly return
    self:buildVocab(sentences, sentence_toks)    

    local inputs = {}
    local inputs_rev = {}
    for step=1,maxLen do
        inputs[step] = torch.LongTensor(#sentences)
        for j=1,#sentences do
            local tokens = sentence_toks[j]
            if step > #tokens then
                inputs[step][j] = 0 --padding token
            else
                local tok = sentence_toks[j][step]
                local tok_id = self.word2idx[tok]
                if tok_id == nil then
                    tok_id = self.word2idx['<UNK>']
                end
                inputs[step][j] = tok_id
            end
        end
        if self.gpuid >= 0 then inputs[step] = inputs[step]:cuda() end
    end
    print("max sentencen length:"..maxLen)
    for step=1,maxLen do
        inputs_rev[step] = torch.LongTensor(#sentences)
        for j=1,#sentences do
            local tokens = sentence_toks[j]
            inputs_rev[step][j] = inputs[maxLen-step+1][j]
        end
        if self.gpuid >= 0 then inputs_rev[step] = inputs_rev[step]:cuda() end
    end
    self.maxLen = maxLen
    return {inputs, inputs_rev}
end

function SimpleBiLSTM:buildVocab(sentences, sentence_toks)
    if self.idx2word ~= nil then
        --means the vocabulary is already created
        return 
    end
    self.idx2word = {}
    self.word2idx = {}
    self.word2idx['<PAD>'] = 0
    self.idx2word[0] = '<PAD>'
    self.word2idx['<UNK>'] = 1
    self.idx2word[1] = '<UNK>'
    self.vocabSize = 2
    for i=1,#sentences do
        local tokens = sentence_toks[i]
        for j=1,#tokens do
            local tok = tokens[j]
            local tok_id = self.word2idx[tok]
            if tok_id == nil then
                self.vocabSize = self.vocabSize+1
                self.word2idx[tok] = self.vocabSize
                self.idx2word[self.vocabSize] = tok
            end
        end
    end
    print("number of unique words:" .. #self.idx2word)
end

function SimpleBiLSTM:save_model(path)
    --need to save the vocabulary as well.
    torch.save(path, {self.net, self.idx2word, self.word2idx})
end

function SimpleBiLSTM:load_model(path)
    local object = torch.load(path)
    self.net = object[1]
    self.idx2word = object[2]
    self.word2idx = object[3]
end
