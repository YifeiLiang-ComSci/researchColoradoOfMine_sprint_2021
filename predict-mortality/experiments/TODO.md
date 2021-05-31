## COVID
[x] p-th norm loss, usually p < 1.
[x] Toy dataset test.
[ ] Different factor for static and dynamic features in reconstruction error.
[o] learning_rate= 0.00001 was stable, all losses decreased consistently.
[x] BatchNormalization works better than Dropout.
[x] Separate static features from dynamic features.
[o] Do NOT use drop out in RNN model, such as '"dropout": 0.3, "recurrent_dropout": 0.1' parameters. We SHOULD set zero drop out, such as dropout = 0.0 and recurrent_dropout = 0.0, and the prediction accuracy was improved from 53% to 79%.
[o] NOT using regularizer also slightly improves the accuracy.
[ ] assistant: Better classification baseline models : especially for MLP.
[ ] assistant: Hyper-parameter tuning of SLAE.
[x] Precision, Recall fix.
[ ] Gradient based feature identification.
[ ] Reconstruction Visualization.
[ ] assistant : Merge Methods section, To study : progit, stash, .gitignore
[ ] assistant: Python, class syntax, packing/unpacking parameters.
[ ] Fix random seed number when release code.
[ ] The feature value distribution may not be uniform, e.g. binary, so pick most close value when calculate the perturbation gradients.
[ ] Plot Reconstruction VS Original Records in 2D image: x axis of 2D image is grid of time points in range[t_0, t_T], and y axis is features with 75 pixels (excluding RE_DATE, age, admission time, discharge time). 
[ ] Feature Importance : Makes one feature all 'unobserved' (mask is 0), and observe the prediction change between that feature is observed and unobserved.
[ ] LSTM Encoder is generator, stack of encoded representations is pass to the discriminator (another LSTM, outputs the certainty vector) with concatenated time stamps.

## Alzheimer's Disease
[ ] Group norm regularization on SNPs input group.

## IB
[ ] Update utilsforminds.
[ ] Clone and Install Alz.
[ ] Put the Alz dataset.
[ ] Pull and merge the cvpr2021 branch to your branch.
[ ] Study model, dataset, output.
[ ] During vscode live share, when you want to compile your edits, Hoon will give you terminal access. At my terminal (accessible via bottom panel of vscode window), type:
"latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf cvpr.tex".

## IDEA
[x] Try the different predictors (end point of SAE), such as SVM, instead of Dense.
[ ] Visualize the enriched vectors to show temporal consistency can be learned.
    [ ] Pick most dense samples (participants) to visualize.
[ ] Visualize the original vectors/images v.s. reconstructed vectors/images to show enriched vector contains enough information to summarize the original data.
[ ] Plot importance distribution on images.
[ ] Rotate feature importance graph.
[ ] Plot survival curve v.s. temporal graph of top important features.