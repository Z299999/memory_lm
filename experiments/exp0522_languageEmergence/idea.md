这个实验我们做关于语言的产生。

我先引用我之前一片report里面的一段话，是关于agent之间communication的描述。

In real life, communication is somehow by mediums in the natural environment, e.g. air conveying voices, and paper carrying words. Spreading message between individuals rely almost on interaction with environment or cell products. In our model, we enable individuals to communicate through environment. Each individual is given $k$ extra output nodes, and use them to generate bull shit to the environment and leave it at the location. These bull shit contains information from biological body. Although we are not able to decode these bull shit, another individual with $k$ extra input nodes are able to pick up the bull shit as signal and decode it in its body.

这里我们强调的是语言的涌现，而不是语言的训练。

这个agent我们假设它是一个mlp，有m个输入头和n个输出头，在m个头和n个头中，分别有k个语言头，听语言的感受器和说话的效应器。我们观察语言的涌现和演化，所以语言不参与反向传播。

其余的m-k个输入头和n-k个输出头都是正常的用来完成任务的头，输出头需要用来反向传播，他们的反向传播会影响所有的edge，所以下一次说话时，他们的语言功能也会慢慢的受影响。

我们需要设计一个实验，来观察语言的涌现。我们或许可以先从单agent自言自语开始，之后再推广到多agent。然后关于task，我们也可以先从简单的开始，比如拟合函数之类的，你觉得呢？或者是做生存实验也可以，比如agent在2d空间里捕食和生存，输入是闻到的气味，输出是运动控制，另外他也会自言自语，输出头说出的话下一时间和和下一时刻输入一起进入输入层。

再未来，我们甚至可以考虑和0513的多巴胺学习和0422-0501的强化学习结合起来，不过这些是未来的事情了，现在先不管，但是或许可以写入计划。

我们一起来讨论。