# super-weight-circuit-patching
TL;DR: I reproduced the "Superweight" failure mode in OLMo-1B (where deleting one weight causes catastrophic collapse). Then, I attempted to repair the model using a tiny, rank-1 row patch trained on a CPU. The patch recovered around 93% of the lost performance, but interestingly, it did not just relearn the original weight! Instead, it learned a new, distributed circuit orthogonal to the original.

Results (NLL, PPL, & KL)
![](results.png)

Training Loss
![](training_loss_plot.png)
