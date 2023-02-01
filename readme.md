# intro

This repo contains code that can use SAI human pipeline dataset (png and info.json) to predict focal length.

# approach

I used the following code as reference:

https://www.kaggle.com/code/ivankunyankin/resnet18-from-scratch-using-pytorch/notebook

I made a few changes to the training loop, including modifying the loss function to fit a regression problem instead of classification. I also added Weights and Biases.

# results

Results so far have shown that resnet18 can achieve an average L1 loss of logs of predicted and target of ~0.23 on validation set.

This section will be updated with new results as they're available.

# next steps

- [ ] Make sure no issues with train loop by overfitting to small train set.
- [ ] Look at outliers
- [ ] Look at heat maps of conv layers to understand what's being learned so far.
- [ ] Look at signed difference in logs.
