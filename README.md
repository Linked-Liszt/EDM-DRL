# EARL: Toward Stable Reinforcement Learning through Evolutionary Algorithms

## Abstract

Deep reinforcement learning (RL) has experienced tremendous growth in the past few years. However, training stability of agents continues to be an open research question. Here, the authors present a closely entwined hybrid method which combines concepts from the fields of evolutionary computing (EC) and deep RL as a means of mitigating training instability in RL agents. We are able to show that our method trains more consistently than RL and EC baselines alone. We also show that by employing ensemble methods, performance of RL agents can be improved during test time while simultaneously preserving the training stability of the agent. Finally, we conduct an ablation study to identify components within the EARL agent responsible for highest contribution. Though this work is in its early phases, it stands to benefit the stability of RL agents during training by combining multiple AI disciplines.

### Requirements

* Python 3.7
* Libraries
  * numpy
  * gym
  * matplotlib
  * scipy
  * torch

### Results

The graphs which correspond to the results of this work are documented below.

<img src="graphics/baseline.png"/>
As shown, though the A2C algorithm is able to solve the environment faster on average, the EARL agent
is able to solve the environment with a tighter standard deviation than both the A2C and CGP baselines.

<img src="graphics/ensemble.png"/>
This demonstrates that the weight voting mechanism is able to out perform the 1-elite strategy. In
constrast, the softmax ensemble approach does not surpass the 1-elite strategy in final performance.

<img src="graphics/ablation.png"/>
Here we see that the ablations of various mutation and recombination methods do not have significant
effect on the final performance or stability of agents during training. In particular, the mean
recombination strategy has high overlap with the unmodified base agent.
