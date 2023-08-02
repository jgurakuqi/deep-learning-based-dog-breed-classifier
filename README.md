# deep-learning-based-dog-breed-classifier


## Table of Contents

- [Description](#Description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Description

The goal of the project is to show how some changes in [this kaggle project](https://www.kaggle.com/code/gabrielloye/dogs-inception-pytorch-implementation/notebook) allowed me
to improve the test accuracy of the model from 79% to 87% and the entire processing pipeline. The introduced changes are:
- More functions for higher modularity and better memory management.
- Multiprocessing to handle the preprocessing.
- Using SGD instead of Adam.
- Using a different training algorithm based on Constraint propagation.


## Installation

The requirements for this project are:
- (Python 3)[https://www.python.org/downloads/]
- (Pytorch)[https://pytorch.org/]
  
```bash
pip install numpy
pip install matplotlib
pip install pandas
pip install pillow
pip install torchvision
pip install -U albumentations
```

## Usage

To run the project it's required an IDE or editor which supports jupyer notebooks.

## Contributing

This project aimed at showing how much a model training could be improved changing several aspects of it, like:
- Data augmentation.
- Training the whole CNN.
- Change optimizer.
- Change learning rate (through schedulers or other methods).

```bash
git clone https://github.com/jgurakuqi/deep-learning-based-dog-breed-classifier
```

Possible changes:
- Replace the multiprocessing pool with a multithreading pool. The project was built upon Python 3.8, hence multithreading has severly improved in terms of performance since then.
- The project was made upon an old version of Pytorch (< 2.0), which did not include the new features (e.g., torch compile). Using newer versions could allow to improve the model training.
- For my tests I used Adam, AdamW and SGD, and I ended up using SGD which showed better results, at the cost of much higher training times. Other optimizers might be considered to improve further the test accuracy and training times.
- The current training batches are devised for my GPU memory (8GB). If your GPU has higher memory amounts, increasing the batch size can improve the execution. BE CAREFUL: might influence also the training results.
- To determine the best Learning-rate paths I tested, I used several Learning rate schedulers, combined with several other configurations of my training context (i.e., with different optimizers, different weight decays, etc...), and I even defined some custom learning rate schedulers, which were fundamental, as they led me to my final Constraint propagation based trainer. Defining new schedulers, and testing the existing ones with other optimizers and further configurations might improve further the training execution time and the model's result.
- My trainer applies a Constraint propagation over the training, by training in the same epoch the model multiple times with a range of possible learning rates (with configurable granularity), and proceedes in the training of the model which achieves the best validation scores in that epoch ("pruning" those models which did not achieve the best result). This idea could be further improved introducing new a new logic for the update of the learning rates. 
- The model currently uses early stopping. During my tests the chosen stopping threshold has shown to achieve the best results, but using this training model over other datasets and problems might require different thresholds.
- Other augmentations might be used to improve the model's generalization.

## License

MIT License

Copyright (c) 2023 Jurgen Gurakuqi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
