# caption-generation

#DATA
Flickr8k dataset
Flickr8k dataset is a dataset of of 8k image with captions.Below are some features of this dataset:
•	It is small in size. So, the model can be trained easily on low-end laptops/desktops.
•	Data is properly labelled.
•	For each image 5 captions are provided.
•	The dataset is available for free.

Data pre-processing and cleaning is an important part of the whole model building process. Understanding the data helps us to build more accurate models.
After extracting zip files you will find below folders
Flickr8k_Dataset: Contains a total of 8092 images in JPEG format with different shapes and sizes. Of which 6000 are used for training, 1000 for test and 1000 for development.
Flickr8k_text : Contains text files describing train_set ,test_set. Flickr8k.token.txt contains 5 captions for each image i.e. total 40460 captions.

Glove.6B embeddings
GloVe stands for Global Vectors for Word Representation. It was developed by Jeffrey Pennington,   Richard Socher,   Christopher D. Manning at nlp department stanford.


Highlights
1.   Nearest neighbors
The Euclidean distance (or cosine similarity) between two word vectors provides an effective method for measuring the linguistic or semantic similarity of the corresponding words. Sometimes, the nearest neighbors according to this metric reveal rare but relevant words that lie outside an average human's vocabulary. For example, here are the closest words to the target word frog:

•	frog
•	frogs
•	toad
•	litoria
•	leptodactylidae
•	rana
•	lizard
•	eleutherodactylus


2.   Linear substructures
The similarity metrics used for nearest neighbor evaluations produce a single scalar that quantifies the relatedness of two words. This simplicity can be problematic since two given words almost always exhibit more intricate relationships than can be captured by a single number. For example, man may be regarded as similar to woman in that both words describe human beings; on the other hand, the two words are often considered opposites since they highlight a primary axis along which humans differ from one another.

In order to capture in a quantitative way the nuance necessary to distinguish man from woman, it is necessary for a model to associate more than a single number to the word pair. A natural and simple candidate for an enlarged set of discriminative numbers is the vector difference between the two word vectors. GloVe is designed in order that such vector differences capture as much as possible the meaning specified by the juxtaposition of two words.

  
man - woman                                       company - ceo
  
city - zip code                                comparative - superlative

The underlying concept that distinguishes man from woman, i.e. sex or gender, may be equivalently specified by various other word pairs, such as king and queen or brother and sister. To state this observation mathematically, we might expect that the vector differences man - woman, king - queen, and brother - sister might all be roughly equal. This property and other interesting patterns can be observed in the above set of visualizations.






#generation of caption from image
Model made and the collab file both are attached.The project tries to guess the caption of images given in the specifed url or uploade
Image captioning is a problem that belong to two domains: NLP(Natural Language Processing)
and CV (Computer Vision).In our project we have combined LSTM text generation with the computer vision powers of a convolutional neural network. 
We have used transfer learning to utilize these two projects:
•	InceptionV3
•	Glove
We use inception to extract features from the images. Glove is a set of Natural Language Processing (NLP) vectors for common words.
we begin by importing needed libraires and certain functions from libraries into our notebook.Following is the list of libraries we have used.

1.	os
2.	string
3.	glob
4.	tqdm
5.	time
6.	pickle
7.	numpy
8.	PIL (pillow)
9.	Matplotlib
10.	Keras with tensorflow 2.x backend

Then we download the required files. We download 3 files out of which two files are zip files of images and captions while the third file are glove embeddings. Since these are all available publically they can be directly downloaded to notebooks using unix command !wget. All these files are zip files so the next step is to unzip these files using th unix command !unzip (Note ! is used in google colab to invooke unix commands actual unix command is unzip).

Next we clean up the captions by removing using follwing steps:
1.	Converting every word into lower case.
2.	Removing all the punctuations.
3.	Removing all the one letter words.
4.	Removing all the characters that are not alphabets.

These cleaned captions are then mapped to images in a lookup table. Another list is made for all the words that are present in the captions. All the elements of the list are unique.

After cleaning the captions we add start and stop token to the beggining and end of the caption respectively.This will help us to inititate and terminate caption generation. We also split the cleaned data into training and testing data.We have total of 8k images we take 6k as training images while the rest 2k are reserved for testing puposes.

The next step is to define the image encoder for our network we use inceptionv3 as our encoder wih imagenet weights. The architecture performs pretty well on imagenet dataset and hence the the encoding produced by this network would be great representaion of the image. Encoder is made by remomving the last layer of the netwok which output layer that categorises the image. Instead we use the second last layer that has 2048 units as output. Output we get after passing through the modified network is used as encoding of the image. 

Since, the images we have can of different sizes but our network architecture can only take fixed size image as input, we need to reize the image before passing it through the encoder. In case of inceptionv3 the input must be (224,224). So, we resize the image to the same.

To reduce the training time we can compute all the encodings in advance. So, we compute all the encodings and save it to a pickle file which can be loaded when training.This is done to both training and testing datasets.

Next we need to remove the less frequent words from the captions as they can be misleading while training network.we set the word count threshold to 10 words, i.e. any words that occurs less than ten times is removed from captions as well as dictionary. Also we need to assign an index to every word in vocabulary. This will help us to identify the word while training.


Since loading all the data may fill up all the ram in low end systems we use progressive loading to avoid system crash while training. In progressive loading a set is loaded from memory while another set is being trained and is removed from memory as soon as they are not required. At each step the generator yields two outputs one contanis the images and the input sequence while the other contains output sequence. The nuber of images yeilded in one step depends on batch size that is passed as argument to the generator.

Next step is to create our main model as described in model architecutre section .the glove embeddings that we have loaded before is coonverted into suitaable  matrix and then inserted as wieght to embedding layer. The embedding layer is made untrainable so that the embeddings do not change while training.we compile the model with adam as our optimizing algorithm while categorical cross entropy as loss metric. We start training our model With learning rate 1e-4 and batch size 5. Training runs for 20 epoch and takes approximately 2.5 hours to complete

