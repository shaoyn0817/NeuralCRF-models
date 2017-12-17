local Index, parent = torch.class('nn.Index', 'nn.Module')

function Index:__init(dimension)
    parent.__init(self)
    self.dimension = dimension[1]
    self.index = dimension[2]
    self.gradInput = {self.gradInput} --, self.gradInput.new()
end

function Index:updateOutput(input)
    local t = input[1]
    self.output:index(t, self.dimension, self.index)
    return self.output
end

function Index:updateGradInput(input, gradOutput)
    local t = input[1]

    --self.gradInput[2]:resize(self.index:size()):zero()

    local gradInput = self.gradInput[1] -- no gradient for the index variable
    gradInput:resizeAs(t):zero()
    gradInput:indexAdd(self.dimension, self.index, gradOutput)
    return self.gradInput[1]
end

function Index:clearState()
    self.gradInput[1]:set()
    self.gradInput[2]:set()
    self.output:set()
    return self
end
