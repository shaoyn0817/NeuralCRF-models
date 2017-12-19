local AttentionSRNN, parent = torch.class('AttentionSRNN', 'AbstractNeuralNetwork')

function AttentionSRNN:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function AttentionSRNN:initialize(javadata, ...)
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
    self.bidirection = javadata:get("bidirection")
 
 

    if isTraining then
        self.x = self:prepare_input()
        self.numSent = #data.sentences
    end

    if self.network == nil and isTraining then
        -- means is initialized process and we don't have the input yet.
        self:createNetwork()
        --print(self.network)
    end
    if self.network == nil then 
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
function AttentionSRNN:createNetwork()
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
    if self.bidirection then
        bwd1 = nn.Sequential()
           :add(sharedLookupTable:sharedClone())
           :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))
           
        bwdSeq1 = nn.Sequential()
            :add(nn.Sequencer(bwd1))
            --:add(nn.ReverseTable())
    end


    parallel1 = nn.ParallelTable()
    parallel1:add(fwdSeq1)
    
    if self.bidirection then
        parallel1:add(bwdSeq1)
    end

    brnn1 = nn.Sequential()
       :add(parallel1)
       :add(nn.JoinTable(3))


    mergeHiddenSize = hiddenSize
    if self.bidirection then
        mergeHiddenSize = 2 * hiddenSize
    end

    net = nn.Sequential():add(brnn1):add(nn.ConcatTable():add(nn.Identity()))

    fwd2 = nn.Sequencer(nn.FastLSTM(mergeHiddenSize, hiddenSize):maskZero(1))
    local bwd2
    if self.bidirection then
        bwd2 = nn.Sequential()
            :add(nn.Sequencer(nn.FastLSTM(mergeHiddenSize, hiddenSize):maskZero(1)))
            --:add(nn.ReverseTable())
    end

    map = nn.MapTable()
    parallel2 = nn.ConcatTable():add(fwd2)

    if self.bidirection then
        parallel2:add(bwd2)
    end
    map:add(parallel2)
    lstm2 = nn.Sequential():add(map):add(nn.MapTable():add(nn.JoinTable(3)))

    brnn2 = nn.Sequential()
        :add(lstm2)

    attenmlp =nn.Sequencer(nn.Sequential():add(nn.Linear(mergeHiddenSize, 1)):add(nn.ReLU()))
    atten = nn.Sequential():add(nn.Sequential():add(attenmlp):add(nn.SoftMax()))

    attenlayer = nn.MapTable():add(nn.ConcatTable():add(nn.Sequential():add(atten):add(nn.Squeeze(3))):add(nn.Identity()))
    final = nn.MapTable():add(nn.Sequential():add(nn.MapTable():add(nn.Transpose({1,2})))
                                         :add(nn.MixtureTable())
                                         :add(nn.Linear(mergeHiddenSize, self.numLabels))
                                         :add(nn.ReLU()))
 
    padlen = 60
    index1 = nn.ConcatTable()
    for i = 1, padlen, 1 do   -----------------------
        l = torch.LongTensor({i})
        index1:add(nn.Index({1, l}))
    end   
    i1 = nn.Sequential():add(index1):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())  

    --numsen timestep hiddensize
    --seqlen x batchsize x featsize
    index2 = nn.ConcatTable()

    for i = 1, padlen-1, 1 do
        l = torch.LongTensor({i, i+1})
        index2:add(nn.Index({1, l}))
    end          
    i2= nn.Sequential():add(index2):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())  


    index3 = nn.ConcatTable()
    for i = 1, padlen-2, 1 do
        l = torch.LongTensor({i, i+1, i+2})
        index3:add(nn.Index({1, l}))
    end          
    i3= nn.Sequential():add(index3):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())  


    index4 = nn.ConcatTable()
    for i = 1, padlen-3, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3})
        index4:add(nn.Index({1, l}))
    end
    i4= nn.Sequential():add(index4):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())            


    index5 = nn.ConcatTable()
    for i = 1, padlen-4, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4})
        index5:add(nn.Index({1, l}))
    end
    i5= nn.Sequential():add(index5):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())  


    index6 = nn.ConcatTable()
    for i = 1, padlen-5, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5})
        index6:add(nn.Index({1, l}))
    end
    i6= nn.Sequential():add(index6):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())            


    index7 = nn.ConcatTable()
    for i = 1, padlen-6, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6})
        index7:add(nn.Index({1, l}))
    end
    i7= nn.Sequential():add(index7):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())  


    index8 = nn.ConcatTable()
    for i = 1, padlen-7, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6, i+7})
        index8:add(nn.Index({1, l}))
    end
    i8= nn.Sequential():add(index8):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())            


    index9 = nn.ConcatTable()
    for i = 1, padlen-8, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8})
        index9:add(nn.Index({1, l}))
    end
    i9= nn.Sequential():add(index9):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())  


    index10 = nn.ConcatTable()
    for i = 1, padlen-9, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9})
        index10:add(nn.Index({1, l}))
    end 
    i10= nn.Sequential():add(index10):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())           


    indexlayer = nn.ConcatTable():add(i1):add(i2):add(i3):add(i4):add(i5):add(i6):add(i7):add(i8):add(i9):add(i10)
    self.network = nn.Sequential():add(net):add(indexlayer):add(nn.FlattenTable())
    
if self.gpuid >=0 then 
    self.network:cuda() 
end

end

function AttentionSRNN:obtainParams()
    --make sure we will not replace this variable
    self.params, self.gradParams = self.network:getParameters()
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

function AttentionSRNN:createOptimizer()
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

function AttentionSRNN:forward(isTraining, batchInputIds)
 
    if self.gpuid >= 0 and not self.doOptimization and isTraining then
        self.params:copy(self.paramsDouble:cuda())
    end
    local nnInput = self:getForwardInput(isTraining, batchInputIds)
    out = torch.CudaTensor()
    if isTraining then
        if not self.bidirection then
            out = self.network:forward({nnInput[1]})
            out = torch.cat(torch.CudaTensor(), out, 1)
        else
            out = self.network:forward(nnInput)
            out = torch.cat(torch.CudaTensor(), out, 1) 
        end             
    else
        batchsize = 3
        batchnumber = math.ceil(nnInput[1]:size(2)/batchsize)
        if self.bidirection then
            for i = 0, batchnumber-1, 1 do
                start = i*batchsize+1
                tail = (i+1)*batchsize
                bs = batchsize

                if tail > nnInput[1]:size(2) then
                    bs = nnInput[1]:size(2)-start+1
                end

                temp_output = self.network:forward({nnInput[1]:narrow(2, start, bs),nnInput[2]:narraow(2, start, bs)})
                temp_tensor = torch.cat(torch.CudaTensor(), temp_output, 1) 
                out = torch.cat(out, temp_tensor, 1)
            end
        else
            for i = 0, batchnumber-1, 1 do
                start = i*batchsize+1
                tail = (i+1)*batchsize
                bs = batchsize

                if tail > nnInput[1]:size(2) then
                    bs = nnInput[1]:size(2)-start+1
                end
                temp_output = self.network:forward({nnInput[1]:narrow(2, start, bs)})
                temp_tensor = torch.cat(torch.CudaTensor(), temp_output, 1) 
                out = torch.cat(out, temp_tensor, 1)
            end
        end
    end
    self.output = out:double()
    --if self.gpuid >= 0 then
      --  nn.utils.recursiveType(self.output, 'torch.DoubleTensor')
    --end  
    if not self.outputPtr:isSameSizeAs(self.output) then
        self.outputPtr:resizeAs(self.output)
    end
    self.outputPtr:copy(self.output)
end

function AttentionSRNN:getForwardInput(isTraining, batchInputIds)
    if isTraining then
        if batchInputIds ~= nil then
            --self.x shape is 100* allsentencenumber    
            batchInputIds:add(1) -- because the sentence is 0 indexed.
            self.batchInputIds = batchInputIds
            --self.x1 = torch.cat(torch.CudaTensor(), self.x[1], 1):index(2, batchInputIds)
            --self.x1 = self.x1:view(self.x1:size(2), self.x1:size(1)):view(-1)
            self.x1 = self.x[1]:index(2, batchInputIds) 
           
            --torch.split(self.x1Tab, self.x1, batchInputIds:size(1), 1)
            --self.x2 = torch.cat(self.x2, self.x[2], 2):index(1, batchInputIds)
            --self.x2 = self.x2:resize(self.x2:size(2), self.x2:size(1)):view(-1)
            --torch.split(self.x2Tab, self.x2, batchInputIds:size(1), 1)
            self.x2 = self.x[2]:index(2, batchInputIds) 
            self.batchInput = {self.x1, self.x2}
            
            return self.batchInput
        else
            return self.x
        end
    else
        return self.testInput
    end
end

function AttentionSRNN:getBackwardInput()
    if self.batchInputIds ~= nil then
        return self.batchInput
    else
        return self.x
    end
end

function AttentionSRNN:getBackwardSentNum()
    if self.batchInputIds ~= nil then
        return self.batchInputIds:size(1)
    else
        return self.numSent
    end
end

function AttentionSRNN:backward()
    self.gradParams:zero()
    local gradOutputTensor = self.gradOutputPtr
    local backwardInput = self:getBackwardInput()  --since backward only happen in training
    local backwardSentNum = self:getBackwardSentNum()
    torch.split(self.gradOutput, gradOutputTensor, backwardSentNum, 1)
    if self.gpuid >= 0 then
        nn.utils.recursiveType(self.gradOutput, 'torch.CudaTensor')
    end
    self.network:backward(backwardInput, self.gradOutput)
    if self.doOptimization then
        self.optimizer(self.feval, self.params, self.optimState)
    else
        if self.gpuid >= 0 then
            self.gradParamsDouble:copy(self.gradParams:double())
        end
    end
    
end

function AttentionSRNN:prepare_input()
    local data = self.data

    local sentences = data.sentences
    local sentence_toks = {}
    local maxLen = 60
    for i=1,#sentences do
        local tokens = stringx.split(sentences[i]," ")  --tokens is a table  {i, am , a, student}
        table.insert(sentence_toks, tokens)
 
        if #tokens > maxLen then
            --maxLen = #tokens
            print('maxlen is too short')
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
    inputs = torch.cat(torch.CudaTensor(), inputs, 3)
    inputs_rev = torch.cat(torch.CudaTensor(), inputs_rev, 3)
    i1 = inputs:transpose(1, 3)
    i2 = i1:transpose(2, 3)
    r1 = inputs_rev:transpose(1, 3)
    r2 = r1:transpose(2, 3)
    return {i2:squeeze(), r2:squeeze()}
end

function AttentionSRNN:buildVocab(sentences, sentence_toks)
    if self.idx2word ~= nil then
        --means the vocabulary is already created
        print('already created')
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

function AttentionSRNN:save_model(path)
    --need to save the vocabulary as well.
    torch.save(path, {self.network, self.idx2word, self.word2idx})
end

function AttentionSRNN:load_model(path)
    local object = torch.load(path)
    self.network = object[1]
    self.idx2word = object[2]
    self.word2idx = object[3]
end
