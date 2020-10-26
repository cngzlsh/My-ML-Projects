# My-ML-Projects
Machine Learning projects I have done throughout learning.

### Cat vs Dog
A convolutional neural network trained on my local machine that identifies a pet as either a cat or a dog.
To run the IPython Notebook, please ```pip install``` packages: torch, matplotlib, numpy, tqdm, PIL. Alternatively, run ```pip install -r requirements.txt```.
#### Main ConvNet Structures
- Convolutional layers: 6 layers, each mapping previous inputs to 16, 32, 64, 128, 256, 32 features. Each convolutional layer is followed by a batch normalisation layer and a ReLU layer. Dropout layers (p=0.25) and max pooling layers are inserted in an attempt to prevent overfitting.
- Fully Connected layers: 2 layers, mapping the previous features into 2 categories. Log sigmoid function is used to output a probability of each category (Cat or Dog).
#### Data
16000 cats and 16000 dogs images, collected from Kaggle (https://www.kaggle.com/chetankv/dogs-cats-images). Can also be downloaded from my OneDrive shared file: https://liveuclac-my.sharepoint.com/:u:/g/personal/zcqssli_ucl_ac_uk/EeDt9KuvkFdMuGlaZdKHbrsB-S48sMgVElgiCqYjlYG28g?e=h0Oof7 (expires June 2021).
Random transformation, resized crop is performed prior to feeding into the dataloader. Images are shuffled and converted into torch tensor.
#### Hyperparameters
My laptop does have a dGPU with CUDA support but is however entry-level. Exact specification is GeForce MX150 with 2GB VRAM.
- Batch size is set to 4 (I have tried 8 but there is not enough CUDA memory)
- Loss function is set to Cross Entropy Loss.
- Optimiser is set to Adam.
- Learning rate is set to 0.00001.
- Training epoch is set to 12.
#### Training
Training 8000 batches for 12 epoches took approximately 16 hours on my laptop. (Time to invest I guess!)
When plotted epoch loss, it can be seen to decrease smoothly from 5200 to 4000.
![Epoch_Loss](/images/loss_over_epochs_cat_v_dog.png)
The trained model is saved as a ```.pth``` file (21MB) which can be loaded.
#### Testing
When tested on a test set of 500 cats and 500 dogs images, the testing accuracy reached 75. This is certainly not the best but it can be seen that the model is working!
#### Predictions
Here is an example of the model predicting on an unseen image (not from training or test sets).
![Prediction_Example](/images/prediction_examples_cat_v_dog.png)
(Sihao, May 2020)
