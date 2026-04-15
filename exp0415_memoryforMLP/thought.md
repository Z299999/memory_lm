This folder carries out smoke test for how MLP carry memory through online learning.

To be specific, we ask a simple shallow MLP to do forward propagation using constant input, let us say, 1 (will be <BOS> for LLMs we will implement in very future)

the outputs are required to be 0 1 0 1 0 1 0 1 ,..... along the time series

if there is no backpropagation then the MLP would be a constant function, yielding constant output.

the thing we want to do is to do online learning, by frequently doing back propagation to change the MLP thus change the function it represents.

the training is:

each time we forward propagate constant 1, and the output is y. we compute error of y from y hat, and do back propagation.

for example, if this time the correct answer is 1 but the MLP outputs 0.5, then there is strong gradient that make the MLP tends to output 0 next time but not 1. 

this is by training, the MLP is trained to approximate next correct answer but not this correct answer.

meanwhile, the training parameters and the network structure is well-designed such that online learning would not consume too much computational resources.

a sample parameter set could be:

batchsize = 1
for each epoch, number of batch = 1
the batch data contains current time step, only one single pair, e.g (1,0) or (1,1) or something.
and the MLP can be simple like [1 4 4 1]

the goal we want to see:

online training allows the MLP to predict next output if input is constant 1 all the time, while the change comes from the change in MLP parameter by iteslf through back propagation.