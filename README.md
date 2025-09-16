## Titans: Learning to memorise at test time


This is my implementation of the paper "Titans: Learning to memorise at test time" from Google Research. The paper aims to address limitations in how current AI models handle long-term memory with a neural memory module that learns and adapts during inference, inspired by human memory systems. 

My implementation is focused on the neural long-term memory component of the _Titans_ architecture. Put very simply, this can be as straightforward as an MLP that does online regression when new information to be remembered is fed to the LLM. 

What does it regress onto what? Well, it learns two different linear transformations of the input tokens $x_t$ to convert it into a key-value pair. 

_Aside: My understanding is that the weights in the matrices that do this are learned along with the training of the transformer, as they are fixed during the online training of the neural memory module. Therefore, since I did not want to train a transformer in my implementation, I fixed these._

Then, the 'key' is passed forward through the MLP and the loss for regression is the MSE between the result of this and the 'value'. 

So by training the small MLP at test time, the neural network can flexibly adapt to learn and memorise new information. The paper also introduces lots of interesting interpretations of standard optimisation procedures:
- Surprise-based memory update. They define surprise as the gradient of the memory module parameters, so this is equivalent to gradient descent.
- Incorporating momentum in the gradient descent optimiser is akin to including a window of 'past surprise' as well as current surprise. 
- Forgetting mechanism via a parameter ($\alpha$) that controls how much weight should be given to incorporatng new information vs keeping existing information. This becomes important when the memory module approaches its memory capacity (which intuitively is proportional to the overall 'capacity', in the deep learning sense, of the network). It is equivalent to weight decay and is closely related to the gating mechanism in modern RNNs. 


### My implementation

As mentioned above, I shied away from training a transformer for this. This is because I had limited time to work on it, and wanted to be able to iterate quickly without worrying about the big heavy task of training a transformer, which is not really the focus of this exercise anyway. The point of the paper is to be able to memorise at test time, so I integrated my memory module with GPT-5 mini via the OpenAI API to show that it can be used with off-the-shelf LLMs. 

A further simplification I made was to focus on a simple, somewhat contrived sub-problem. This was necessary in order to fix the matrices that linearly transform the new information tokens into the key-value pair for memory training. The problem class I chose was memorising fixed-length sequences of numbers, specifically sequences of 4 integers between 0 and 1000. This works by splitting the sequence into the first three numbers (we'll call this the key) and the last number (the value). It should now be obvious what the matrices are - they are just selector matrices that pick out the respective sub-sequences for the key and value. Then, at memory-test-time, I give the memory module the first three numbers of a sequence it should have memorised and hope that it outputs the fourth. 

It's crucial at this point to mention that the sequences are totally random, so we are truly testing memory. Usually, neural networks aim to learn some general input-output mapping by approximating a function that works for all the samples in the training dataset. There is no such function here; there is no underlying low-dimensional statistical similarity across training data that can be leveraged or extracted by the network. We are literally treating it as a storage device. 

While this may seem like a major reduction in scope, I felt that this was a good way to get a minimal proof-of-concept up and running quickly. It's also straightforward to evaluate, since there is exactly one correct answer for a given 3-number sequence stem at test time, provided the model has seen the full sequence before. I could then use this for more interesting things like testing memory capacity (read on for those results).


### A tricky detail

One thing that tripped me up initially was that the network will naturally try to learn some mathematical relationship between the numbers in the input sequence that maps them to the target value. Since the numbers are random, this won't work so we need to treat the numbers as token IDs, like in a language model. This means we need an embedding layer, which can project the input sequence into an embedding space. In the case of language, the vectors in this embedding space would have some semantic meaning - similar bits of language would get mapped to similar vectors. This won't be the case here; it's actually just a hack to force the network to treat the data as categorical rather than numerical/continuous. 

The exact way in which we 'tokenise' and embed the input is a design choice. The first thing I tried was using a standard off-the-shelf tokeniser and applying this to my integer sequence data. The tokeniser I used was `cl100k_base`, which is the one used by GPT-4 and GPT-3.5 Turbo. As the name suggests, it has a vocabulary of 100,000 tokens, and it can handle multiple languages. However, I quickly realised this is total overkill for this application. The reason this matters is that the number of parameters in the embedding layer is proportional to the vocabulary size, since each unique token in the vocabulary must be mapped to a unique vector representation. With this tokeniser, I can handle all of language (and for a general-purpose memory module, you would want this), but I only want to be able to embed the numbers from 0 to 1000. So I simplified and just use the numbers themselves as token IDs. This keeps the number of parameters in the MLP to an absolute minimum. 


### Initial tests

I have split up each phase of exploration into a different Python script within the `tests` folder. 
Chronologically, the first of these was `memorise_one_seq.py`. Having simplified the problem as above, the first step was to see if I could, at LLM inference time, learn a new sequence using the neural memory module, feed it back to the LLM as context and then let the LLM attention decide whether it now knows the answer. The script queries the LLM twice, once before training the memory MLP and once after. In terms of simplicity, this is most similar to the "Memory as a Layer" architecture from the paper, since the memory module preprocesses the input before passing it to the attention module of the LLM, and it does not use the outputs of attention to update the memory module (which is what happens in "Memory as Context").
This basic proof-of-concept was extended in `memorise_multi_seq.py`, where the goal was to memorise multiple sequences. This required a lot more care around neural network training dynamics as exploding gradients were a common issue. Despite the very small amount of training data, this is in some sense a difficult learning problem as there is actually nothing to _learn_ as such, and one training sample may suggest a totally different gradient to another. 

_Aside: I probably could have avoided some of these training issues by using an optimiser like Adam which does parameter-wise learning rate scheduling. However, I stuck with (stochastic) gradient descent to stay faithful to the paper and so that I could sequentially add in things like momentum and weight decay to see how that affected training dynamics._

Once I established that I could memorise 50+ sequences, I wrote a quick script (`simulation.py`) to simulate doing this "in the wild", where the memory module randomly switches between training and inference. This is more like the setting in which such a system might be used in real life. There is nothing technically new in this script, but it shows a more realistic use case for the memory module, as opposed to having separate training and testing phases. 


### Testing capacity

Now we can get to the interesting bit - testing the capacity of the memory module, i.e. how many sequences can be memorised with a given MLP architecture? This is what I test in the `capacity_test.py` script. Once we know the capacity of a given architecture, we can also bring in and test the adaptive forgetting mechanism.
