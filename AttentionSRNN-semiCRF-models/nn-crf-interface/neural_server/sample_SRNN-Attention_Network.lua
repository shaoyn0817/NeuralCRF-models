require 'nn'
require 'rnn'
--require 'Index'
bidirection = true
numsen = 2
numLabels = 5
hiddenSize=4 
maxlan = 40
    --sample input x,  shape is [timestep, batchsize, featuresize]
    x = torch.randn(maxlan, numsen, hiddenSize)

    fwd1 = nn.Sequential()
       --:add(sharedLookupTable)
       :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))

    fwdSeq1 = nn.Sequencer(fwd1)

    local bwd1, bwdSeq1
    if bidirection then
        bwd1 = nn.Sequential()
           --:add(sharedLookupTable:sharedClone())
           :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))
           
        bwdSeq1 = nn.Sequential()
            :add(nn.Sequencer(bwd1))
            --:add(nn.ReverseTable())
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

    net = nn.Sequential():add(brnn1):add(nn.ConcatTable():add(nn.Identity()))

    fwd2 = nn.Sequencer(nn.FastLSTM(mergeHiddenSize, hiddenSize):maskZero(1))
    local bwd2
    if bidirection then
        bwd2 = nn.Sequential()
            :add(nn.Sequencer(nn.FastLSTM(mergeHiddenSize, hiddenSize):maskZero(1)))
            --:add(nn.ReverseTable())
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

    attenlayer = nn.MapTable():add(nn.ConcatTable():add(nn.Sequential():add(atten):add(nn.Squeeze(3))):add(nn.Identity()))
    final = nn.MapTable():add(nn.Sequential():add(nn.MapTable():add(nn.Transpose({1,2})))
                                         :add(nn.MixtureTable())
                                         :add(nn.Linear(mergeHiddenSize, numLabels))
                                         :add(nn.ReLU()))
 
    regularmlp = nn.MapTable():add()
    index1 = nn.ConcatTable()
    
    for i = 1, maxlan, 1 do   -----------------------
        l = torch.LongTensor({i})
        index1:add(nn.Index({1, l}))
    end   
    i1 = nn.Sequential():add(index1):add(brnn2:sharedClone())--:add(attenlayer:sharedClone()):add(final:sharedClone())  

    --numsen timestep hiddensize
    --seqlen x batchsize x featsize
    index2 = nn.ConcatTable()
    for i = 1, maxlan-1, 1 do
        l = torch.LongTensor({i, i+1})
        index2:add(nn.Index({1, l}))
    end          
    i2= nn.Sequential():add(index2):add(brnn2:sharedClone())--:add(attenlayer:sharedClone()):add(final:sharedClone())  


    index3 = nn.ConcatTable()
    for i = 1, maxlan-2, 1 do
        l = torch.LongTensor({i, i+1, i+2})
        index3:add(nn.Index({1, l}))
    end          
    i3= nn.Sequential():add(index3):add(brnn2:sharedClone())--:add(attenlayer:sharedClone()):add(final:sharedClone())  


    index4 = nn.ConcatTable()
    for i = 1, maxlan-3, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3})
        index4:add(nn.Index({1, l}))
    end
    i4= nn.Sequential():add(index4):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())            


    index5 = nn.ConcatTable()
    for i = 1, maxlan-4, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4})
        index5:add(nn.Index({1, l}))
    end
    i5= nn.Sequential():add(index5):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())  


    index6 = nn.ConcatTable()
    for i = 1, maxlan-5, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5})
        index6:add(nn.Index({1, l}))
    end
    i6= nn.Sequential():add(index6):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())            


    index7 = nn.ConcatTable()
    for i = 1, maxlan-6, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6})
        index7:add(nn.Index({1, l}))
    end
    i7= nn.Sequential():add(index7):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())  


    index8 = nn.ConcatTable()
    for i = 1, maxlan-7, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6, i+7})
        index8:add(nn.Index({1, l}))
    end
    i8= nn.Sequential():add(index8):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())            


    index9 = nn.ConcatTable()
    for i = 1, maxlan-8, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8})
        index9:add(nn.Index({1, l}))
    end
    i9= nn.Sequential():add(index9):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())  


    index10 = nn.ConcatTable()
    for i = 1, maxlan-9, 1 do
        l = torch.LongTensor({i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9})
        index10:add(nn.Index({1, l}))
    end 
    i10= nn.Sequential():add(index10):add(brnn2:sharedClone()):add(attenlayer:sharedClone()):add(final:sharedClone())           


    --indexlayer = nn.ConcatTable():add(i1):add(i2):add(i3):add(i4):add(i5):add(i6):add(i7):add(i8):add(i9):add(i10)
    indexlayer = nn.ConcatTable():add(i1):add(i2):add(i3)
    network = nn.Sequential():add(net):add(indexlayer):add(nn.FlattenTable())
    

    --forward the input 
    r1 = network:forward({x,x})
    print(r1)

    --just use the output to test the network backpropagation
    r2 = network:backward({x,x}, r1)
    print(r2)




