# HMM model for POS tagging

## Mandatory part

For the Viterbi algorithm we've taken some decisions. For words missing in the word2pos distribution or , we just set a uniform probability for each tag.
For missing transition probabilities in data estimation, as we are working with logs and we can't do the log(0), we just set a very small negative value 1e-10.

## Optional part

The following file contains the output of the execution for the optional bonus part. The confusion matrix can be found at `cm.png`.

```text
Epoch 1:
211727/211727 [==================================================] - 144s 681us/step - accuracy: 0.8757 - batch loss: 0.0029 - epoch loss: 0.0178
Test accuracy: 0.8881875872612

Epoch 2:
211727/211727 [==================================================] - 144s 681us/step - accuracy: 0.8986 - batch loss: 0.0055 - epoch loss: 0.0143
Test accuracy: 0.8933107256889343

Epoch 3:
211727/211727 [==================================================] - 143s 677us/step - accuracy: 0.9052 - batch loss: 0.0044 - epoch loss: 0.0134
Test accuracy: 0.8790940046310425

Epoch 4:
211727/211727 [==================================================] - 143s 678us/step - accuracy: 0.9072 - batch loss: 0.0049 - epoch loss: 0.0130
Test accuracy: 0.9022572636604309

Epoch 5:
211727/211727 [==================================================] - 144s 680us/step - accuracy: 0.9093 - batch loss: 0.0061 - epoch loss: 0.0127
Test accuracy: 0.9003041386604309

Epoch 6:
211727/211727 [==================================================] - 143s 677us/step - accuracy: 0.9108 - batch loss: 0.0071 - epoch loss: 0.0126
Test accuracy: 0.9004721641540527
Best accuracy obtained: 0.9022572636604309
```