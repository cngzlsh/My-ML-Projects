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
#### Evaluation
When tested on a test set of 500 cats and 500 dogs images, the testing accuracy reached 75. This is certainly not the best but it can be seen that the model is working!
#### Predictions
Here is an example of the model predicting on an unseen image (not from training or test sets).
![Prediction_Example](/images/prediction_examples_cat_v_dog.png)
(Sihao, May 2020)


### German-English Translator
This is a Seq2Seq neural machine translation model trained on the Multi30k dataset. I was heavily inspired by https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html and the AI Core tutorial https://www.youtube.com/watch?v=UnVYPOGiK3E.
#### Seq2seq structures
The model consists of two recurrent neural netwroks: a German Encoder and an English Decoder with Attention applied. Both RNNs are LSTMs.
Encoder: Consists of an embedding layer, a dropout layer, a hidden LSTM layer and a feed-forward linear output layer. The LSTM is bidirectional so the outputs have doubled the size. When source sentence (in German) is passed into the encoder, it returns the encoding output of the sentence and a tupple (hidden state, cell state).
Decoder: Consists of an embedding layer, a two-layer linear attention model, a hidden LSTM layer and a feed-forward linear output layer. When the encoding of the German sentence (from the encoder) is passed into the decoder, attention is calculated using encoder output and the hidden states at each time step. Attention score is concatenated with the embedding of the encoder output and then passed into the LSTM layer and output layer. It returns the predicted English translation, the LSTM hidden states and the attention scores.
#### Data
Parallel corpus: 30000 German sentences and their English translation. Preprocessing is done using SpaCy including word tokenisation, adding start of sequence and end of sequence tokens and padding sentences. Split into 29000 training sentences and 1000 validation sentences for calculating BLEU score.
An example is shown here:
![Multi30k_Example](/images/multi30k_example.png)
#### Hyperparameters
Due to hardware restrictions, batch size is set to 50.
- Input and Output dimensions are the length of vocabulary of source and target fields. (as instructed on the torchtext.Multi30K page)
- Embedding layer size for both Encoder and Decoder: 256
- Hidden LSTM layer size for both Encoder and Decoder: 512
- Training epochs: 50
- Learning rate: 0.002
- Optimiser: Adam
- Loss Function: Cross Entropy Loss
#### Training
#### Evaluation (BLEU score)
#### Test Translation
Some examples are shown here:
![Translation_Example](/images/translation_example.png)
Problems:
- Named identity is not recognised e.g. deutschland
- Some nouns are not recognised e.g. apfel even though this exists in the training set
- Some grammar are not correct e.g. sie kommt = she comes; sie kommen = they come.
Rooms for improvement:
- Modify models with more layers and dimensions
- Increase batch size
- Implement beam search algorithm
- Implement Transformer model
(Sihao, July 2020)
