import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import re #regular expressions library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.applications import efficientnet
''' convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolut
from keras.layers import TextVectorization
keras.utils.set_random_seed(111)
#wget is a tool that sustains file downloads in unstable and slow network connections
!wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
!wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
!unzip -qq Flickr8k_Dataset.zip
!unzip -qq Flickr8k_text.zip
!rm Flickr8k_Dataset.zip Flickr8k_text.zip

IMAGES_PATH = "Flicker8k_Dataset"
IMAGE_SIZE = (224, 224)
VOCAB_SIZE = 10000
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE
#AUTOTUNE , which will prompt the tf.data runtime to tune the value dynamically at runtime
def load_captions_data(filename):
with open(filename) as caption_file:
caption_data = caption_file.readlines()
caption_mapping = {}
text_data = []
images_to_skip = set() #skipping unsable data
for line in caption_data:
line = line.rstrip("\n")
# Image name and captions are separated using a tab
img_name, caption = line.split("\t")
# Each image is repeated five times for the five different captions.
# Each image name has a suffix `#(caption_number)`
img_name = img_name.split("#")[0]
img_name = os.path.join(IMAGES_PATH, img_name.strip())
# We will remove caption that are either too short to too long
tokens = caption.strip().split()
if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
images_to_skip.add(img_name)
continue
if img_name.endswith("jpg") and img_name not in images_to_skip:
# We will add a start and an end token to each caption
caption = "<start> " + caption.strip() + " <end>"
text_data.append(caption)
if img_name in caption_mapping:
caption_mapping[img_name].append(caption)
else:
caption_mapping[img_name] = [caption]

for img_name in images_to_skip:
if img_name in caption_mapping:
del caption mapping[img name]

del caption_mapping[img_name]
return caption_mapping, text_data
def train_val_split(caption_data, train_size=0.8, shuffle=True):
# 1. Get the list of all image names
all_images = list(caption_data.keys())
# 2. Shuffle if necessary
if shuffle:
np.random.shuffle(all_images)
# 3. Split into training and validation sets
train_size = int(len(caption_data) * train_size)
training_data = {
img_name: caption_data[img_name] for img_name in all_images[:train_size]
}
validation_data = {
img_name: caption_data[img_name] for img_name in all_images[train_size:]
}
# 4. Return the splits
return training_data, validation_data
# Load the dataset
captions_mapping, text_data = load_captions_data("Flickr8k.token.txt")
# Split the dataset into training and validation sets
train_data, valid_data = train_val_split(captions_mapping)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))
 Number of training samples: 6114
Number of validation samples: 1529
os # handling the files
pickle # storing numpy features
numpy as np
dm.notebook import tqdm # how much data is process till now
nsorflow.keras.applications.vgg16 import VGG16 , preprocess_input # extract features from image data.
nsorflow.keras.preprocessing.image import load_img , img_to_array
nsorflow.keras.preprocessing.text import Tokenizer
nsorflow.keras.preprocessing.sequence import pad_sequences #ensure same length
nsorflow.keras.models import Model
nsorflow.keras.utils import to_categorical, plot_model
nsorflow.keras.layers import Input , Dense , LSTM , Embedding , Dropout , add
vgg16 Model
VGG16()
ucture model
Model(inputs = model.inputs , outputs = model.layers[-2].output)
rize
odel.summary())
-trainable Parameters:
: These parameters are fixed and do not change during training. They are often used for tasks like data preprocessing or feat
: In some pre-trained models, like VGG16 or ResNet, the convolutional layers have non-trainable parameters because they've a
ly connected layer of the VGG16 model is not needed, just the previous layers to extract feature results.

Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vg
553467096/553467096 [==============================] - 4s 0us/step
Model: "model"
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
input_1 (InputLayer) [(None, 224, 224, 3)] 0
block1_conv1 (Conv2D) (None, 224, 224, 64) 1792
block1_conv2 (Conv2D) (None, 224, 224, 64) 36928
block1_pool (MaxPooling2D) (None, 112, 112, 64) 0
block2_conv1 (Conv2D) (None, 112, 112, 128) 73856
block2_conv2 (Conv2D) (None, 112, 112, 128) 147584
block2_pool (MaxPooling2D) (None, 56, 56, 128) 0
block3_conv1 (Conv2D) (None, 56, 56, 256) 295168
block3_conv2 (Conv2D) (None, 56, 56, 256) 590080
block3_conv3 (Conv2D) (None, 56, 56, 256) 590080
block3_pool (MaxPooling2D) (None, 28, 28, 256) 0
block4_conv1 (Conv2D) (None, 28, 28, 512) 1180160
block4_conv2 (Conv2D) (None, 28, 28, 512) 2359808
block4_conv3 (Conv2D) (None, 28, 28, 512) 2359808
block4_pool (MaxPooling2D) (None, 14, 14, 512) 0
block5_conv1 (Conv2D) (None, 14, 14, 512) 2359808
block5_conv2 (Conv2D) (None, 14, 14, 512) 2359808
block5_conv3 (Conv2D) (None, 14, 14, 512) 2359808
block5_pool (MaxPooling2D) (None, 7, 7, 512) 0
flatten (Flatten) (None, 25088) 0
fc1 (Dense) (None, 4096) 102764544
fc2 (Dense) (None, 4096) 16781312
=================================================================
Total params: 134260544 (512.16 MB)
Trainable params: 134260544 (512.16 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
None
'
Fully connected layer of the VGG16 model is not needed, just the previous layers t

features={}

import os
# extract features from image
features = {}
directory = os.path.join('/content/Flicker8k_Dataset')
for img_name in tqdm(os.listdir(directory)):
# load the image from file
img_path = directory + '/' + img_name
image = load_img(img_path, target_size=(224, 224))
# convert image pixels to numpy array
image = img_to_array(image)
# reshape data for model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# preprocess image for vgg
image = preprocess_input(image)
# extract features
feature = model.predict(image, verbose=0)
# get image ID
image_id = img_name.split('.')[0]
# store feature
features[image_id] = feature

from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
# store features in pickle
'''features_path = '/content/drive/My Drive/features.pkl'
with open(features_path, 'wb') as f:
pickle.dump(features, f)'''
'''Extracted features are not stored in the disk, so re-extraction of features can extend running time Dumps and store your
import pickle
def load_features_from_pickle(file_path):
with open(file_path, 'rb') as f:
features = pickle.load(f)
return features
# Example usage:
file_path = '/content/drive/My Drive/features.pkl'
features = load_features_from_pickle(file_path)
with open(('/content/Flickr8k.token.txt'), 'r') as f:
next(f)
captions_doc = f.read()

airs to her playhouse .\n1000268201_693b08cb0e.jpg#4\tA little girl in a pink dress
going into a wooden cabin .\n1001773457_577c3a7d70.jpg#0\tA black dog and a spotted
dog are fighting\n1001773457_577c3a7d70.jpg#1\tA black dog and a tri-colored dog pla
ying with ea
'

'
captions_doc[200:458]

100% 40460/40460 [00:00<00:00, 177182.71it/s]
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
# split the line by tab
tokens = line.split('\t')
if len(tokens) < 2:
continue
image_id, caption = tokens[0], " ".join(tokens[1:]).strip()
# remove extension from image ID
image_id = image_id.split('.')[0]
# create list if needed
if image_id not in mapping:
mapping[image_id] = []
# store the caption
mapping[image_id].append(caption)

for key,value in mapping.items():
print(key," ",value)
break
1000268201_693b08cb0e ['A girl going into a wooden building .', 'A little girl climbing into a wooden playhouse .',

from IPython.display import Image
Image('/content/Flicker8k_Dataset/1000268201_693b08cb0e.jpg')

len(mapping)
8092

def clean(mapping):
for key, captions in mapping.items():
for i in range(len(captions)):
# Take one caption at a time
caption = captions[i]
print(caption)
# Preprocessing steps
# Convert to lowercase
caption = caption.lower()
# Delete digits, special chars, etc.,
caption = re.sub(r'[^a-zA-Z]', ' ', caption)
# Delete additional spaces
caption = re.sub(r'\s+', ' ', caption).strip()
# Add start and end tags to the caption
caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
captions[i] = caption

clean(mapping)
two teenage girls walking down a mall breezeway .
two women walk on the sidewalk .
A group of six children sit at a wooden table .
A group of young asian children pose for a picture at a table .
Six children , all with black hair , sit at a table for a picture .
Six children at a table .
Six small children sitting at a desk .
Two brown and white dogs lift their ears and look through a fence .
Two little dogs looking out of their cage .
Two small dogs are in a cage outside .
Two small dogs in a cage .
Two white and brown dogs up against a fence .
A boy goes down an inflatable slide .
A boy in red slides down an inflatable ride .
A boy is sliding down in a red shirt .
A child going down an inflatable slide .
A young boy sliding down an inflatable , is looking off camera .
one brown and white dog chasing a black and white dog through the grass
The two dogs are running through the grass .
Two cocker spaniels running through the grass .
Two dogs are running through a green yard
Two dogs are running through the grass near a house and trees .
A girl in a pool wearing goggles and surrounded by other children
A girl in green goggles in a pool with three other children .
A red haired girl making a peace sign is wearing neon green glasses and floaties and playing in the pool with other
A red-headed girl offers the peace sign as she swims in the pool with floaties .
A young girl with goggles and floaties poses for the camera as she plays in a pool .
A black dog has a dumbbell in his mouth .
A black dog has a dumbbell in his mouth looking at the person wearing blue .
A black dog holding a weight in its mouth stands next to a person .
A black dog holds a small white dumbbell in its mouth .
The black dog has a toy in its mouth and a person stands nearby .
A man does a wheelie on his bicycle on the sidewalk .
A man is doing a wheelie on a mountain bike .
A man on a bicycle is on only the back wheel .
Asian man in orange hat is popping a wheelie on his bike .
Man on a bicycle riding on only one wheel .
A group is sitting around a snowy crevasse .
A group of people sit atop a snowy mountain .
A group of people sit in the snow overlooking a mountain scene .
Five children getting ready to sled .
Five people are sitting together in the snow .
A grey bird stands majestically on a beach while waves roll in .
A large bird stands in the water on the beach .
A tall bird is standing on the sand beside the ocean .
A water bird standing at the ocean 's edge .
A white crane stands tall as it looks out upon the ocean .
A person stands near golden walls .
a woman behind a scrolled wall is writing
A woman standing near a decorated wall writes .
The walls are covered in gold and patterns .
Woman writing on a pad in room with gold , decorated walls .
A man in a pink shirt climbs a rock face
A man is rock climbing high in the air .
A person in a red shirt climbing up a rock face covered in assist handles .
A rock climber in a red shirt .
A rock climber practices on a rock climbing wall .

from IPython.display import Image
Image('/content/Flicker8k_Dataset/1015118661_980735411b.jpg')

mapping['1015118661_980735411b']
['startseq boy smiles in front of stony wall in city endseq',
'startseq little boy is standing on the street while man in overalls is working on stone wall endseq',
'startseq young boy runs aross the street endseq',
'startseq young child is walking on stone paved street with metal pole and man behind him endseq',
'startseq smiling boy in white shirt and blue jeans in front of rock wall with man in overalls behind him endseq']
#testing a few image mappings
i = 0
for key, captions in mapping.items():
print(key+": ",captions)
i += 1
if(i==5): break
1000268201_693b08cb0e: ['startseq girl going into wooden building endseq', 'startseq little girl climbing into wooden
1001773457_577c3a7d70: ['startseq black dog and spotted dog are fighting endseq', 'startseq black dog and tri colored
1002674143_1b742ab4b8: ['startseq little girl covered in paint sits in front of painted rainbow with her hands in bow
1003163366_44323f5815: ['startseq man lays on bench while his dog sits by him endseq', 'startseq man lays on the benc
1007129816_e794419615: ['startseq man in an orange hat starring at something endseq', 'startseq man wears an orange h

all_captions = []
for key in mapping:
for caption in mapping[key]:
all_captions.append(caption)
len(all_captions)
#unique captions
40459
all_captions[:10]
['startseq girl going into wooden building endseq',
'startseq little girl climbing into wooden playhouse endseq',
'startseq little girl climbing the stairs to her playhouse endseq',
'startseq little girl in pink dress going into wooden cabin endseq',
'startseq black dog and spotted dog are fighting endseq',
'startseq black dog and tri colored dog playing with each other on the road endseq',
'startseq black dog and white dog with brown spots are staring at each other in the street endseq',
'startseq two dogs of different breeds looking at each other on the road endseq',
'startseq two dogs on pavement moving toward each other endseq',
'startseq little girl covered in paint sits in front of painted rainbow with her hands in bowl endseq']

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
8426
# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
max_length
35
#Train Test and split
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size, skip_images=None):
# loop over images
X1, X2, y = list(), list(), list()
n = 0
while 1:
for key in data_keys:
# Check if key should be skipped
if skip_images and key in skip_images:
continue # Skip this image
n += 1
captions = mapping[key]
# process each caption
for caption in captions:
# encode the sequence
seq = tokenizer.texts_to_sequences([caption])[0]
# split the sequence into X, y pairs
for i in range(1, len(seq)):
# split into input and output pairs
in_seq, out_seq = seq[:i], seq[i]
# pad input sequence
in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
# encode output sequence
out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
# store the sequences
X1.append(features[key][0])
X2.append(in_seq)
y.append(out_seq)
if n == batch_size:
X1, X2, y = np.array(X1), np.array(X2), np.array(y)
yield [X1, X2], y
X1, X2, y = list(), list(), list()
n = 0

# encoder model
# image feature layers
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
#Using relu activation function to extract features from fixed length input
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)
# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
#Using softmax activation function helps in predicting next word in sequence
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# plot the model
plot_model(model, show_shapes=True)

from tensorflow.keras.callbacks import ModelCheckpoint
epochs = 10
batch_size = 64
steps = len(train) // batch_size #floor division operator in Python
'''
# Define the filepath to save the model checkpoints
checkpoint_path = "/content/drive/MyDrive/Img_caption_models/model_checkpoint.h5"
# Define a callback to save the model after each epoch

# Define a callback to save the model after each epoch
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
save_weights_only=True,
save_best_only=True)

'''
for i in range(epochs):
# create data generator
images_to_skip = ['2258277193_586949ec62'] # Add image IDs you want to skip
generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size, skip_images=images_t
# fit for one epoch
model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
model.save("/content/drive/MyDrive/Img_caption_models/trained_model1.h5")
from keras.models import load_model
model = load_model("/content/drive/MyDrive/Img_caption_models/trained_model1.h5")
def idx_to_word(integer, tokenizer):
for word, index in tokenizer.word_index.items():
if index == integer:
return word
return None
def predict_caption(model, image, tokenizer, max_length):
# add start tag for generation process
in_text = 'startseq'
# iterate over the max length of sequence
for i in range(max_length):
# encode input sequence
sequence = tokenizer.texts_to_sequences([in_text])[0]
# pad the sequence
sequence = pad_sequences([sequence], max_length)
# predict next word
yhat = model.predict([image, sequence], verbose=0)
# get index with high probability
yhat = np.argmax(yhat)
# convert index to word
word = idx_to_word(yhat, tokenizer)
# stop if word not found
if word is None:
break
# append word as input for generating next word
in_text += " " + word
# stop if we reach end tag
if word == 'endseq':
break
return in_text

from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()
for key in tqdm(test):
# get actual caption
captions = mapping[key]
# predict the caption for image
y_pred = predict_caption(model, features[key], tokenizer, max_length)
print(y_pred)
# split into words
actual_captions = [caption.split() for caption in captions]
y_pred = y_pred.split()
# append to the list
actual.append(actual_captions)
predicted.append(y_pred)
# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

startseq man in red shirt is standing on the air endseq
startseq dog is digging in the air endseq
startseq man in black jacket and black jacket is standing in front of building endseq
startseq man in red jacket and black jacket is sitting in front of window endseq
startseq man in green shirt is sitting on the street endseq
startseq man in red coat and red jacket and red jacket and red jacket and red jacket
startseq man in blue shirt is sitting on the ground endseq
startseq two boys are playing soccer endseq
startseq person is walking on the edge of the edge of the water endseq
startseq man in white shorts is jumping on the ocean endseq
startseq two dogs are playing with ball in the grass endseq
startseq two little girls are standing on the bed endseq
startseq two dogs are playing with dog in the grass endseq
startseq man in red and white jacket is walking down the street endseq
startseq people are gathered in front of fruit endseq
startseq man in black shirt is walking on the beach endseq
startseq two dogs are playing in the grass endseq
startseq man is sitting on the water endseq
startseq man in red and white shirt is standing on the street endseq
startseq man wearing blue jacket and blue jacket is wearing blue jacket endseq
startseq man wearing black jacket is walking down the street endseq
startseq dog is playing in the snow endseq
startseq two people are walking on beach endseq
startseq white dog is walking in the woods endseq
startseq dog is walking in the water endseq
startseq dog running through snow endseq
startseq hiker is walking through the mountains endseq
startseq two children are playing on the playground endseq
startseq man in black pants and black pants and black pants and black pants and black
startseq man in black jacket and white jacket is walking on the sidewalk endseq
startseq man in blue shirt is walking down the edge of the snow endseq
startseq man and woman are walking on the street endseq
startseq man and woman are walking in the woods endseq
startseq person is standing on rocky hill endseq
startseq dog running in the snow endseq
startseq man wearing black jacket walks down street endseq
startseq man in black and white shirt is jumping on the beach endseq
startseq man in black and white jacket is walking on the street endseq
startseq man in black and white pants is walking down the street endseq
startseq two dogs running on dirt track endseq
startseq dog is running on the grass endseq
startseq dog is running in the grass endseq
startseq two girls in traditional dresses are walking on the street endseq
startseq black dog is running through snow endseq
startseq woman in black shirt and white shirt is walking on the road endseq
startseq dog is wearing blue collar and white dog in its mouth endseq
startseq dog is playing in the grass endseq
startseq dog running through the grass endseq
startseq man in black shirt is walking on the street endseq
startseq black dog is running through the snow endseq
startseq little boy is swinging on playground endseq
startseq black dog is wearing collar is wearing collar endseq
startseq two girls are walking on the street endseq
startseq man in blue shirt and white shirt is standing on the sand endseq
startseq dog is walking through the woods endseq
startseq dog is jumping over the grass endseq
startseq dog is digging on the ground endseq
startseq dog with yellow dog in the grass endseq
startseq man in blue shirt is jumping on the sidewalk endseq
startseq man in black shirt is sitting on the sidewalk endseq
startseq dog is walking through the water endseq
startseq two dogs are playing on the edge of the edge of the snow endseq
startseq man in black shirt is standing on the street endseq
startseq two dogs are running through the grass endseq
startseq man in black and white pants is walking on the edge of the edge of the water
startseq man is riding up wave endseq
startseq skier skiing down snowy mountain endseq
startseq man in black shirt is jumping on the wall endseq
startseq brown dog is running through the grass endseq
startseq man is standing on rock endseq
startseq dog runs through the snow endseq
startseq man in black and white shorts is jumping into the ocean endseq
startseq two people are sitting on the bench endseq
startseq dog is playing on the grass endseq
startseq black and white dog is jumping over the air endseq
startseq dog is playing with toy endseq
startseq woman in red jacket and black jacket and black jacket and black jacket and b
startseq two girls are playing volleyball endseq
startseq dog is running on the grass endseq
startseq two dogs are playing with ball in the grass endseq
100% 810/810 [15:47<00:00, 1.10it/s]

q g p y g g q
startseq black dog is running through the grass endseq
startseq dog and white dog are walking in the water endseq
startseq black and white dog is running on the air endseq
startseq black and white dog is playing on the edge of the potted potted white dog en
startseq man and woman are sitting on table with their arm in the room endseq
startseq man is sitting on dock endseq
startseq man scales rock rock endseq
startseq man in black jacket and black jacket is walking on the street endseq
startseq dog is walking through water endseq
startseq man with blue collar is lying on his blanket endseq
startseq man is jumping through the air endseq
startseq man is jumping off high high in the air endseq
startseq man in black and white shirt is walking down the street endseq
startseq dog is running through the woods endseq
startseq man in red and white sail is walking on the water endseq
startseq dog is running in the grass endseq
startseq dog with yellow collar is walking in the grass endseq
startseq man in black jacket is standing in front of an outdoor library endseq
startseq dog is walking through the woods endseq
startseq two dogs are running in the grass endseq
startseq man is walking on the edge of cliff endseq
startseq man is climbing on the snow endseq
startseq two dogs are playing in the woods endseq
startseq black dog is walking on the ocean endseq
startseq two women in black and black and white and white and white and white and whi
startseq man in red shirt is walking down the street endseq
startseq two dogs are walking along the grass endseq
startseq dog is running in the snow endseq
startseq dog is playing in the snow endseq
startseq dog running through the grass endseq
startseq two children are sitting on bench endseq
startseq dog swimming in the water endseq
startseq man wearing helmet is jumping through the woods endseq
startseq woman in black and white shirt is riding on the road endseq
startseq man in black and white jacket is sitting on his arm in the air endseq
startseq man is sitting on the ground with his head on the ground endseq
startseq man is standing on the water endseq
startseq two people are sitting on the street endseq
startseq man in red jacket and white jacket is standing in front of an outdoor street
startseq man in black jacket and black jacket is standing in front of the camera ends
startseq man wearing black jacket and sunglasses is standing in front of the camera e
startseq man in blue shirt and blue shirt is sitting on the street endseq
startseq man is standing on the hill with his dog in the background endseq
startseq man wearing glasses is sitting on the camera endseq
startseq two dogs are playing in the snow endseq
startseq two girls are playing on the street endseq
startseq group of people in black clothes are standing in the street endseq
startseq black dog is running through the grass endseq
startseq man is walking on the water endseq
startseq black dog is running through the water endseq
startseq dog is running on the grass endseq
startseq man in black and white pants is jumping on the street endseq
startseq man in black shirt and white pants is jumping on the street endseq
startseq man in black shirt is walking on the grass endseq
startseq two dogs are walking through the snow endseq
startseq white dog is running through the snow endseq
startseq woman in black shirt and white shirt is walking on the beach endseq
startseq two people are walking down street endseq
startseq man in red shirt and white shirt is walking on the street endseq
startseq black dog is running in the grass endseq
startseq brown dog is running through the grass endseq
startseq two children are sitting on pool endseq
startseq white dog is walking through the snow endseq
startseq man is walking on the sand endseq
startseq two hikers are hiking up mountain endseq
startseq woman in red and black and black boots and silver handbag walks past traffic
startseq man in black and white pants is jumping over his bicycle endseq
startseq two men are playing soccer ball in the field endseq
startseq two dogs are playing on the grass endseq
startseq man riding his bike on dirt track endseq
startseq man rides dirt bike on dirt track endseq
startseq man rides dirt bike on dirt dirt endseq
startseq man wearing red helmet is riding his bike on the dirt endseq
startseq man in red and white coat is standing on the street endseq
startseq man in black and white jacket is standing in front of water endseq
startseq man in black jacket and black jacket is walking down the street endseq
startseq two people are walking on street endseq
startseq two dogs are playing in the grass endseq
startseq man in black jacket is sitting on bench endseq
startseq dog is playing in the water endseq
startseq two dogs are walking on the snow endseq
startseq man in black and black hair and black hair and black hair and black hair and

startseq man in black wetsuit is riding up wave endseq
startseq man in yellow canoe and yellow collar is in the water endseq
startseq man in red jacket is standing in front of an outdoor table endseq
startseq man in black and white jacket is standing on the street endseq
startseq man is riding boat on the water endseq
startseq man is kayaking through the water endseq
startseq man is jumping over the wall endseq
startseq man in black shirt and red shirt and red shirt and red shirt and red shirt a
startseq dog runs through the snow endseq
startseq man in blue shirt is jumping on the trampoline endseq
startseq the little boy is playing soccer ball endseq
startseq white dog is running on the snow endseq
startseq two people are walking down snowy hill endseq
startseq man is jumping over the water endseq
startseq man in black jacket and black jacket and black jacket and black jacket and b
startseq brown dog is running through the grass endseq
startseq boy in blue and blue swimsuit is being in pool endseq
startseq dog is digging in the snow endseq
startseq man is riding tricks on his bike on his bike endseq
startseq dog is running through the water endseq
startseq dog with red collar and white dog on the snow endseq
startseq two dogs are running through the field endseq
startseq man wearing blue jacket is walking down snowy wall endseq
startseq man in black shirt is jumping on the street endseq
startseq man in black shirt is sitting on the wall endseq
startseq two dogs are playing in the grass endseq
startseq dog is walking on the sidewalk endseq
startseq man in blue shirt and blue shirt is walking down the street endseq
startseq dog is walking along the water endseq
startseq brown dog is walking in the snow endseq
startseq man in black cap is sitting on the street endseq
startseq two boys are sitting on the sidewalk endseq
startseq brown dog is running on the beach endseq
startseq man in black and white shirt is walking on the street endseq
startseq woman wearing black jacket and white jacket is standing on the beach endseq
startseq man is swinging into lake endseq
startseq man and woman are walking on the street endseq
startseq dog running in the snow endseq
startseq two children are playing on the playground endseq
startseq two dogs are playing on the snow with black dog in the snow endseq
startseq man in black jacket is walking on the sidewalk endseq
startseq two dogs are running through the grass endseq
startseq the man is holding his head on the street endseq
startseq little boy in red jacket is sitting on the camera endseq
startseq little boy is sitting on playground endseq
startseq black dog is running through the grass endseq
startseq man is walking into lake endseq
startseq man in red and white and white and white and white and white and white and w
startseq two girls are playing on beach endseq
startseq dog is walking on the ground endseq
startseq two children are sitting on playground endseq
startseq man in blue shorts is jumping into the water endseq
startseq dog is running through the water endseq
startseq little girl is sitting on playground endseq
startseq man in white and white shirt is sitting on the bench endseq
startseq dog running through the grass endseq
startseq two children are walking on the street endseq
startseq two hockey players are playing in the snow endseq
startseq man in red and red jacket walks down traffic endseq
startseq two boys are playing soccer ball with the ball endseq
startseq man in black and white shirt is running on the ground endseq
startseq man in black and black jacket walks down the street endseq
startseq two dogs are playing on the beach endseq
startseq two men are playing soccer ball in the grass endseq
startseq brown dog running through the grass endseq
startseq man in blue and black and black and black and black and black and black and
startseq man in black and white shirt is playing on the air endseq
startseq two dogs are playing on the ground endseq
startseq dog is running on the grass endseq
startseq two dogs are playing on the beach endseq
startseq man in black and black jacket and black jacket and black jacket and black ja
startseq dog with red collar is running in the grass endseq
startseq man in blue wetsuit is swimming in the water endseq
startseq dog with something in its mouth is running in the grass endseq
startseq two children are playing on the playground endseq
startseq dog in the snow endseq
startseq two women in black pants and black pants and black pants and black pants and
startseq man in black jacket and black jacket is walking down the street endseq
startseq little girl in pink dress is sitting on the bed endseq
startseq little boy is eating confetti on the grass endseq
startseq man in blue shirt and white shirt is running on the ground endseq
startseq two dogs are playing in the grass endseq

startseq man and white dog are sitting on the tree endseq
startseq two dogs are running on the beach endseq
startseq man in blue shirt is sitting on the rocks endseq
startseq man in black shirt is sitting on the wall endseq
startseq man in black and white jacket is standing on the street endseq
startseq two dogs are running in the grass endseq
startseq woman and woman are sitting on the camera endseq
startseq two people are playing in pool endseq
startseq two children are sitting on bench endseq
startseq little girl in green and blue jacket is sitting in the bathtub endseq
startseq dog running through the woods endseq
startseq man wearing black jacket and white scarf is walking down the street endseq
startseq man and woman are standing in the distance endseq
startseq dog is jumping in the woods endseq
startseq man is jumping on the woods endseq
startseq dog is walking through the woods endseq
startseq man with glasses is wearing black shirt endseq
startseq boy in blue shirt and blue shirt is walking on the sidewalk endseq
startseq black dog is running in the water endseq
startseq black dog is running through the water endseq
startseq two dogs are playing in the snow endseq
startseq two girls are standing on the road endseq
startseq dog runs along the beach endseq
startseq man in black and white shirt is standing in front of an outdoor restaurant e
startseq man in white jacket is walking on the ocean endseq
startseq two women in black clothes are standing on the street endseq
startseq man in black jacket and white jacket is standing in front of the street ends
startseq man wearing glasses is sitting at the camera endseq
startseq man in black and white pants is walking on the water endseq
startseq little boy with pink dress is running on the grass endseq
startseq two children are sitting on bench endseq
startseq man is driving through the woods endseq
startseq two children are playing on the grass endseq
startseq man is standing on rock endseq
startseq little girl in pink dress is sitting on the beach endseq
startseq black dog is running in the grass endseq
startseq two people are sitting on the street endseq
startseq dog is running on the grass with yellow ball endseq
startseq little girl with pink hair is walking on the grass endseq
startseq two children are sitting on tree endseq
startseq man in red shirt is jumping down the wall endseq
startseq man is jumping into the water endseq
startseq man in red and red vest is sitting in the air endseq
startseq man in blue shirt is jumping on ramp endseq
startseq man in red jacket and white jacket is skiing down mountain endseq
startseq dog swimming in the water endseq
startseq black and black dog is running in the snow endseq
startseq black and white dog is jumping over the fence endseq
startseq white dog is running through the grass endseq
startseq man in black and white jacket is walking on the street endseq
startseq child with red hat and red hat is eating his head endseq
startseq man in black shirt is walking on the edge of rocks endseq
startseq brown dog is running through the grass endseq
startseq man is sitting on dock with lake endseq
startseq person is skiing down snowy mountain endseq
startseq black dog is walking through the water endseq
startseq woman in red and black boots and black boots and black boots and black boots
startseq two people are walking on the grass endseq
startseq two people in red suits are shoveling snowy hill endseq
startseq two children are playing on the grass endseq
startseq man in red shirt and red shirt walks down the street endseq
startseq man in black shirt is jumping on the street endseq
startseq two older dogs are sitting on the bench endseq
startseq man in life life life life life life life life life life life life life life
startseq man in blue wetsuit is riding wave endseq
startseq man in red and white and white and white and white and white and white and w
startseq man in red shirt is walking down the street endseq
startseq dog with superman collar is playing in the snow endseq
startseq man in red jacket is standing in front of fruit endseq
startseq man wearing glasses and glasses is eating his head endseq
startseq woman is walking on the ocean endseq
startseq two children are playing on the grass endseq
startseq black dog is running on the beach endseq
startseq man in black and white shirt is riding unicycle on the street endseq
startseq black dog running through the snow endseq
startseq man in black and white jacket is playing guitar endseq
startseq man in black and black pants is standing on his back in front of the street
startseq man in white shorts and white shorts is walking on the beach endseq
startseq man in black jacket and black jacket is standing on the street endseq
startseq man in blue swim trunks is jumping into pool endseq
startseq boy in blue and blue jacket is running in the water endseq
startseq two girls are standing on the camera endseq

startseq boy is wearing blue shirt and blue shirt is sitting on the water endseq
startseq man and woman are walking on city street endseq
startseq man in black shirt is walking on the street endseq
startseq man in red and black and black and black and black and black and black and b
startseq dog is walking on the grass endseq
startseq two children are sitting on tree endseq
startseq two dogs are playing in the snow endseq
startseq two boys are playing in the air in the air endseq
startseq man is sitting on the blanket with his blanket endseq
startseq man in black and white pants is walking on the edge of water endseq
startseq man in black and white jacket and white pants and white jacket is walking do
startseq man in black and black shirt is jumping on the air endseq
startseq man and woman are posing for picture endseq
startseq little girl in red shirt and white shirt is walking on the street endseq
startseq boy in blue trunks is swimming into pool endseq
startseq man in blue trunks is jumping into the ocean endseq
startseq man and woman are walking down the street endseq
startseq dog running along the beach endseq
startseq two people are sitting on the road endseq
startseq man in black and white vest is walking down the street endseq
startseq dog is sitting on the street endseq
startseq boy in blue shirt and blue shirt is walking on the beach endseq
startseq dog is walking through the water endseq
startseq man in blue jacket and black jacket is wearing blue jacket and black jacket
startseq dog is standing on the snow endseq
startseq dog runs through the air endseq
startseq man in blue shorts is jumping into the water endseq
startseq man in life life life jacket is swimming in the water endseq
startseq two people are sitting on the wall endseq
startseq black dog is jumping into the water endseq
startseq dog is walking through the water endseq
startseq little boy in pink dress is walking on the sand endseq
startseq man wearing black jacket and black jacket is walking on the street endseq
startseq man in black shorts is jumping into the water endseq
startseq two people are sitting on raft endseq
startseq man is jumping into the water endseq
startseq man in red jacket and red jacket is wearing red jacket and red jacket endseq
startseq man and woman are walking on the water endseq
startseq man standing on rocky cliff endseq
startseq man in black and white shirt is riding bicycle on the street endseq
startseq man in black and white jacket and white jacket and white jacket and black pa
startseq dog is running in the snow endseq
startseq man in black jacket is walking down the street endseq
startseq man in black and black shirt and black shirt and black shirt and black shirt
startseq man in black shirt is sitting on the street endseq
startseq dog is running on the grass endseq
startseq dog is walking through the grass endseq
startseq man in black shirt is sitting on bench endseq
startseq man in blue shirt is standing on the air endseq
startseq brown dog is walking on the beach endseq
startseq dog running along beach endseq
startseq man in blue shirt is sitting on the street endseq
startseq man in green shirt is sitting on table in front of an outdoor chairs endseq
startseq man and woman are sitting at table with food in the background endseq
startseq man in black and white shorts is walking on the beach endseq
startseq man wearing black and white jacket is walking on the street endseq
startseq dog in white and white dog and white dog and white dog and white dog and whi
startseq two dogs are running in the snow endseq
startseq man in blue shirt is sitting on bench endseq
startseq man wearing blue jacket and black hat is wearing blue jacket endseq
startseq man in red jacket and red jacket walks down street endseq
startseq woman in black and white jacket walks down the street endseq
startseq dog is running through the grass endseq
startseq two boys are playing on the goal endseq
startseq man and woman are walking on the road endseq
startseq two children ride on playground endseq
startseq dog with yellow collar is standing on the grass endseq
startseq man in black and black jacket and black jacket and black jacket and black ja
startseq man wearing black jacket and black scarf and scarf and woman walking down th
startseq man in red jacket and red jacket and red jacket and red jacket and red jacke
startseq black dog is walking through the grass endseq
startseq woman with pink hair and white hair is standing in front of the camera endse
startseq woman in red dress and white boots walks down street endseq
startseq two people are sitting on the street endseq
startseq two boys are playing with ball in the grass endseq
startseq woman in black and white skirt and white skirt and white skirt and white boo
startseq dog is sitting on the bed endseq
startseq black dog is jumping over the obstacle endseq
startseq two children are playing on playground endseq
startseq dog jumps over an obstacle course endseq
startseq little boy in yellow and blue jacket is running on the grass endseq
startseq man in black and white dog and black dog and black dog and black dog and bla

startseq two children are sitting on playground endseq
startseq boy is wearing blue shirt and hat is sitting on the camera endseq
startseq man in black and white jacket is standing on the grass endseq
startseq man in black and white jacket is walking on the grass endseq
startseq two children are playing on the grass endseq
startseq man is swimming in the water endseq
startseq two boys are playing soccer ball on soccer ball endseq
startseq boy in red shirt is walking on the beach endseq
startseq dog is running through the grass endseq
startseq boy in blue and blue life life life life life life life green green trunks i
startseq two dogs are playing with its hind legs endseq
startseq boy in blue shorts is running on the beach endseq
startseq person is walking on snowy hill endseq
startseq man in blue shirt is jumping on the street endseq
startseq man in pink shirt and white shirt is wearing pink shirt and pink shirt endse
startseq two children are sitting on the wall endseq
startseq two children are sitting in an orange colored yellow and yellow shirt endseq
startseq man is standing on rocky mountain endseq
startseq man is standing on rocky rocks endseq
startseq group of people are walking down street endseq
startseq man in black shirt and white shirt is walking on the sidewalk endseq
startseq boy is sitting on the sand endseq
startseq man in blue jacket is sitting on the floor endseq
startseq black and white dog is running on the grass endseq
startseq two children in red shirts are walking on the street endseq
startseq boy in red shirt and blue shirt is sitting on the street endseq
startseq baby is sitting on the floor endseq
startseq two girls are sitting on the street endseq
startseq group of people are sitting on the street endseq
startseq man wearing black and white jacket is riding his arm in the street endseq
startseq man in black and white jacket is walking down the street endseq
startseq man in black shirt is jumping down the street endseq
startseq boy is swinging on swing endseq
startseq man wearing black jacket and sunglasses and sunglasses and sunglasses endseq
startseq man in black jacket is walking on the street endseq
startseq man in blue shirt is climbing rock wall endseq
startseq man in white jacket and white jacket is standing on snowy mountain endseq
startseq two people are standing on snowy hillside endseq
startseq little boy in blue shirt and jeans is walking on the sand endseq
startseq boy in blue shirt is sitting on the street endseq
startseq woman in black and black and black and black and black and black and black a
startseq man is riding trick on his bicycle endseq
startseq man climbing rock climbing endseq
startseq woman wearing black and white jacket walks down the street endseq
startseq two people are walking on the beach endseq
startseq two people are riding sled on snowy hill endseq
startseq skier skiing down snowy mountain endseq
startseq boy in blue and blue blue blue blue blue blue blue blue blue blue blue blue
startseq little boy swings on swing endseq
startseq man in black and white and white and white and white and white and white and
startseq man is walking down the sidewalk endseq
startseq little boy in pink and white dress is walking down the street endseq
startseq two boys are playing soccer ball endseq
startseq two men are playing soccer ball on the ground endseq
startseq man in black shirt is climbing rock wall endseq
startseq man and woman are walking on the edge of the camera endseq
startseq man is standing on the street endseq
startseq man in black and white jacket is standing on the street endseq
startseq little boy is sitting on playground endseq
startseq two dogs are playing in the grass endseq
startseq two dogs are playing in the grass endseq
startseq person is walking down snowy mountain endseq
startseq little boy is hanging on playground endseq
startseq man wearing red jacket and white jacket is standing on the street endseq
startseq little boy is running through the grass endseq
startseq little boy is swinging on playground endseq
startseq man in blue shorts is playing on the beach endseq
startseq boy in white shirt is playing on the ball endseq
startseq dog is running on the beach endseq
startseq group of people in dresses and dresses are standing in the air endseq
startseq two children are playing in the water endseq
startseq boy in blue shirt is standing in the water endseq
startseq mountain climbers is hiking through snowy mountain endseq
startseq person climbing mountain endseq
startseq man is jumping on the air on the air endseq
startseq two people are walking on the edge of the edge of the water endseq
startseq black dog is running through snow endseq
startseq man in black shirt is sitting on the street endseq
startseq group of people are standing on the street endseq
startseq boy wearing red jacket is wearing red shirt and blue jacket endseq
startseq man in pink shorts and white shorts is jumping into the air endseq
startseq man is standing on the edge of the edge of the empty wall endseq
t t littl i l i i k t t d i k t t i lki th d d

startseq little girl in pink tutu and pink tutu is walking on the sand endseq
startseq man in white and white and white dog and white dog on the bench endseq
startseq man wearing red shirt and white shirt is walking on the street endseq
startseq woman in red and white shirt and white shirt and white shirt and white and w
startseq person is climbing up mountain endseq
startseq man is walking down rocky hill endseq
startseq hiker is skiing down mountain endseq
startseq man is standing on the mountains endseq
startseq brown dog is running in the grass endseq
startseq dog is standing on the street endseq
startseq woman in red shirt and white shirt is standing on the street endseq
startseq little boy swings on swing endseq
startseq dog is running on the grass endseq
startseq man in black shirt is jumping into the water endseq
startseq man wearing blue shirt and blue helmet is riding on dirt bike endseq
startseq two children are sitting on the wall endseq
startseq dog is walking through river endseq
startseq man is jumping over the woods endseq
startseq group of people are walking down the street endseq
startseq man in black and white pants is wearing black pants endseq
startseq man in black and white shirt and white shirt is walking on the street endseq
startseq woman and woman are walking on the street endseq
startseq man in black and white shirt and white shirt and white shirt and white shirt
startseq man wearing black jacket and black jacket and black jacket and black jacket
startseq man and woman are sitting in the middle of the camera endseq
startseq man and black dog are running through the grass endseq
startseq the man is walking down the street endseq
startseq man and woman are standing on rocky path endseq
startseq man wearing red hat is holding his head in the air endseq
startseq man is walking on the street endseq
startseq man in blue shirt is sitting on the street endseq
startseq man in black and black and white jacket is walking through the ocean endseq
startseq man is jumping into the water endseq
startseq group of people are sitting on the street endseq
startseq man wearing blue shirt and white shirt is standing on the street endseq
startseq two dogs are playing in the sand endseq
startseq man in red dress walks down the street endseq
startseq man stands on the edge of the ocean endseq
startseq two dogs are playing in the snow endseq
startseq man in red jacket and white jacket is standing in front of store endseq
startseq man in black and white jacket is walking on the grass endseq
startseq man is riding his bike on the air endseq
startseq group of people are walking down street endseq
startseq little girl in pink dress and pink dress is wearing pink dress and pink dres
startseq dog is running on the grass endseq
startseq two dogs are playing on the snow endseq
startseq man in blue shirt is standing on the rocks endseq
startseq black and white dog is running on the grass endseq
startseq man in blue shirt and blue shirt is running on the beach endseq
startseq woman with pink hair and pink dress is wearing pink shirt and pink dress end
startseq man wearing blue jacket and blue jacket is wearing blue jacket and blue jack
startseq little boy is sitting on the tub of the tub of the orange balls endseq
startseq dog is running in the grass endseq
startseq dog is running in the grass endseq
startseq dog running in the grass endseq
startseq dog with ball in its mouth endseq
startseq man in white shirt is jumping on bed endseq
startseq little boy is sitting on playground endseq
startseq man is walking up in the snow endseq
startseq black dog is jumping on the grass endseq
startseq two people are playing in the air endseq
startseq boy in blue shorts is jumping into the water endseq
startseq dog is digging in the snow endseq
startseq dog is digging in the snow endseq
startseq man in red shirt and white shirt is standing on the street endseq
startseq little boy wearing red and white hat and red hat and red hat and red hat and
startseq man in black and white jacket is walking down the street endseq
startseq black dog is running through the snow endseq
startseq dog jumping over the air endseq
startseq dog running through the snow endseq
startseq man in red shirt and white shirt is walking on the street endseq
startseq dog jumps over the air endseq
startseq man in yellow shirt is playing on the playground endseq
startseq man in black dress walks down the street endseq
startseq two puppies are sitting on bed endseq
startseq woman and woman are sitting on the air endseq
startseq man is standing on the water endseq
startseq man in black and white jacket is jumping over the ocean endseq
startseq dog running through the grass endseq
startseq man is climbing rock endseq
startseq little girl in red shirt and red shirt is standing on playground endseq
startseq man with blue hat is sitting on the bed endseq
t t i b thi h t d hit h t i j i i t th l d

startseq man in green bathing shorts and white shorts is jumping into the pool endseq
startseq black and white dog is running through the snow endseq
startseq man in black shirt and white shirt is walking on the beach endseq
startseq dog is running in the snow endseq
startseq man and man are sitting on the rocks endseq
startseq boy is swinging on swing endseq
startseq person is riding the water endseq
startseq man and woman are playing volleyball on beach endseq
startseq two people are walking on the beach endseq
startseq two dogs are playing in the grass endseq
startseq dog running in the grass endseq
startseq group of children are sitting in water fountain endseq
startseq two people are walking down hill endseq
startseq dog is running on the snow endseq
startseq woman in black and white jacket and white jacket and white jacket and white
startseq boy is eating baby in the tub of the water endseq
startseq man in black and white jacket is walking down the street endseq
startseq dog is running through the grass endseq
startseq man sitting on bench endseq
startseq man is running on the grass endseq
startseq two girls are walking on the street endseq
startseq little boy in red and white shirt and white shirt walks on the street endseq
startseq little girl with pink hair and white hair is standing on the camera endseq
startseq man in black shirt is sitting on piano endseq
startseq dog is walking through the air with yellow dog endseq
startseq man wearing black and black and white shirt is riding unicycle on the street
startseq man in blue shirt is walking on the street endseq
startseq two dogs are running in the snow endseq
startseq man wearing white and white hair is playing guitar endseq
startseq man in black shirt is sitting on the bed endseq
startseq man in blue shirt and white jacket is walking on the street endseq
startseq little boy is playing on the grass endseq
startseq man in blue shirt and white jacket is sitting on the beach endseq
startseq two boys are walking on the beach endseq
startseq two children are playing on the grass endseq
startseq man and woman are walking on the grass endseq
startseq dog is running in the grass endseq
startseq man wearing sunglasses and sunglasses and sunglasses and sunglasses endseq
startseq woman in green goggles is wearing goggles and goggles endseq
startseq man and woman are standing in front of the camera endseq
startseq dog is running on the grass endseq
startseq two dogs are playing in the water endseq
startseq man in white and white vest is playing on the arts endseq
startseq man in red shirt is standing on the beach endseq
startseq man with white dog is sitting on the bed endseq
startseq two children are sitting on the beach endseq
startseq boy wearing blue shirt and blue shirt is riding on the street endseq
startseq two boys are sitting on the water endseq
startseq dog is sitting on the street endseq
startseq two boys are playing on the grass endseq
startseq the dog is walking on the grass endseq
startseq two girls are walking along the beach endseq
startseq man is riding red red and white and white and white and white and white and
startseq man is climbing rock endseq
startseq two children are standing in the water endseq
startseq man in red and white and white and white and white and white and white and w
startseq man is sitting on the wall endseq
startseq dog treads in the water endseq
startseq man is looking up in the snow endseq
startseq hiker is standing on mountain mountain endseq
startseq two children are playing in the snow endseq
startseq man in black and white wetsuit is riding his surfboard in the ocean endseq
startseq white dog is standing in the snow endseq
startseq two people are walking through field endseq
startseq brown dog is running through the grass endseq
startseq two little girls are sitting on bed endseq
startseq man in red and white vest is standing on carnival endseq
startseq two people are sitting on the street endseq
startseq man in red shirt and white shirt is standing on the street endseq
startseq woman in red shirt and white jacket is walking down the street endseq
startseq man in red shirt is standing in the air endseq
startseq toddler in red and white jacket is sitting in the tub endseq
startseq dog runs through the snow endseq
startseq two dogs are running through the grass endseq
startseq two children are standing in the middle of the side of the side of the side
startseq man is standing on the water endseq
startseq child in blue jacket is eating his face endseq
startseq boy in blue life jacket is wearing blue life life life life life life life l
startseq two children are playing in the sand endseq
startseq man in black jacket is standing on the beach endseq
startseq man is standing on rock edge of cliff endseq
startseq man wearing black jacket and black jacket is standing in the air endseq
startseq man in red shirt is walking down the street endseq

startseq man in red shirt is walking down the street endseq
startseq black and white dog is walking through the water endseq
startseq man wearing sunglasses and sunglasses is standing in front of an outdoor str
startseq boy in blue shirt is walking on the grass endseq
startseq little girl in pink shirt is jumping on the grass endseq
startseq man and woman are walking on beach endseq
startseq dog walks on the water endseq
startseq dog running through the grass endseq
startseq baby in blue jacket is sitting in the tub endseq
startseq man in black and black coat is riding up in the snow endseq
startseq two girls are playing on beach endseq
startseq man in black and white jacket is standing in front of the camera endseq
startseq two people are walking on the street endseq
startseq little girl wearing pink dress and white dress is walking on the camera ends
startseq two dogs are playing in field endseq
startseq two dogs are playing with ball in the grass endseq
startseq dog is jumping into the ocean endseq
startseq two girls are playing on playground endseq
startseq boy in blue shirt is holding his arm on the beach endseq
startseq dog with blue collar is holding horse endseq
startseq boy is sitting in the sand endseq
startseq two boys are playing soccer ball on the grass endseq
startseq two people are walking on the beach endseq
startseq boy in red shorts is jumping on the beach endseq
startseq man in black and white jacket walks down the street endseq
startseq man in black and white pants is walking on the street endseq
startseq man in black and white pants is walking on the street endseq
startseq two dogs are playing on the grass endseq
startseq two children are walking down snowy hill endseq
startseq two people are sitting on the water endseq
startseq two boys are sitting on the bed endseq
startseq man in yellow shorts is jumping into the water endseq
startseq man in pink shirt is playing on the beach endseq
startseq dog swimming in the water endseq
startseq black dog with orange collar is running in the grass endseq
startseq man is jumping into the air endseq
startseq two baby are sitting on bed endseq
startseq two dogs are running through the grass endseq
startseq man in blue shirt is sitting on the ground endseq
startseq black and white dog is running on the grass endseq
startseq dog is walking on the street endseq
startseq man wearing black shirt is riding his bike on the street endseq
startseq man and woman are walking through the sand endseq
startseq man in black jacket is standing in front of an outdoor window endseq
startseq man in red and white jacket is standing in front of an outdoor parade endseq
startseq man in black and white shirt is standing on the street endseq
startseq man in red and white and white and white and white and white and white and w
startseq two people are sitting in the street endseq
startseq two people are playing on the snow endseq
startseq man in black and black life life life life life life life life life life lif
startseq man in red jacket and white jacket is walking on the street endseq
startseq man wearing red jacket and red jacket is wearing red jacket endseq
startseq man wearing blue jacket and blue jacket is sitting on bench endseq
startseq black dog runs through the grass endseq
startseq group of people are standing on the street endseq
startseq man is standing on cliff endseq
startseq man in black shirt and white shirt is jumping on the air endseq
startseq brown dog is running through the grass endseq
startseq man in costume and white vest is standing in parade endseq
startseq white dog running along rocky beach endseq
startseq man and woman are walking on street endseq
startseq man in black and white jacket is standing on the street endseq
startseq man is climbing cliff endseq
startseq man in black and white pants and white pants is walking on the street endseq
startseq man in black and white shirt is jumping on the air endseq
startseq dog is running in the snow endseq
startseq two boys are playing in the water endseq
startseq man in red shirt is climbing rock wall endseq
startseq two people are sitting on the street endseq
startseq black and white dog is running through the grass endseq
startseq two children are playing on the street endseq
startseq man in blue swimsuit and black and white jacket is swimming in the ocean end
startseq person is climbing up rock endseq
startseq two dogs are in the water endseq
startseq two people walk down sidewalk endseq
startseq man in white shirt and white shirt is playing on the ground endseq
startseq man in blue shirt is sitting on the sidewalk with his dog endseq
startseq black dog is walking on the grass endseq
startseq man climbs rock wall endseq
startseq two girls in black dress are standing on the camera endseq
startseq little boy in blue shirt and blue shirt is sitting on playground endseq
startseq dog is walking in the snow endseq
startseq dog walks on the water endseq

startseq dog walks on the water endseq
startseq dog runs through the water endseq
startseq two children are playing on the grass endseq
startseq dog with tennis collar is running in the snow endseq
startseq man in pink dress and white jacket is walking on the street endseq
startseq man with yellow dog is holding his dog endseq
startseq woman in black and white shirt is standing on the beach endseq
startseq toddler in blue shirt is sitting on the camera endseq
startseq man in black and white shirt is jumping into the air endseq
startseq dog is running through the snow endseq
startseq man in red shirt is climbing on cliff endseq
startseq man in black and white pants and white pants is walking on the street endseq
startseq man in black and white jacket is riding on the ocean endseq
startseq man in blue hat is wearing black hat endseq
startseq two people are walking on the hillside endseq
startseq man is walking through the water endseq
startseq two girls are walking on the beach endseq
startseq two people are playing on the beach endseq
startseq two children are playing in the water endseq
startseq black dog swims in the water endseq
startseq man wearing black jacket and black pants is sitting on the street endseq
startseq man in blue shirt is climbing up rock wall endseq
startseq boy in black and white shorts is jumping into pool endseq
startseq man is standing on snowy hill endseq
startseq man in red shirt and red shirt and red shirt and red shirt and red shirt and
startseq dog running on the beach endseq
startseq woman in black and white shirt is standing on the beach endseq
startseq man in red shirt and white shirt is riding on the beach endseq
startseq man in blue and white and white and white and white and white and white and
startseq hiker is hiking on the hill endseq
startseq man in black and white jacket is walking down the edge of the side of the oc
startseq two people are riding on the snow endseq
startseq hiker is hiking down snowy mountain endseq
startseq man and woman are standing on rocky rocks endseq
startseq black dog is running on the grass endseq
startseq man wearing black shirt and white shirt is walking on the snow endseq
startseq man is walking down the snow endseq
startseq boy is standing on the rocks endseq
startseq two girls are playing on the beach endseq
startseq man in red shirt and red shirt is sitting on the street endseq
startseq two boys are playing on the air endseq
startseq little boy wearing red shirt is riding on playground endseq
startseq man in red shorts is jumping on the beach endseq
startseq man in blue shirt is sitting on the camera endseq
startseq two dogs are standing on the water endseq
startseq two people are sitting on the water endseq
startseq dog running through snow endseq
startseq man is climbing on rock endseq
startseq person climbing up cliff endseq
startseq man in black shirt is walking on the street endseq
startseq woman in red jacket and white hat is walking down the street endseq
startseq man in red and white boots and white boots walks down street endseq
startseq two children are sitting on the water endseq
startseq black dog is jumping over the snow endseq
startseq two children are playing on the grass endseq
startseq man wearing blue jacket and blue jacket is walking down the street endseq
startseq man is skiing down snowy hill endseq
startseq man scales rock endseq
startseq two dogs are running on the street endseq
startseq two dogs running through field endseq
startseq man in black and white jacket is walking in the water endseq
startseq man in red and white jacket walks down the street endseq
startseq two children sitting in the middle of wall endseq
startseq brown dog is walking in the snow endseq
startseq little boy is sitting on the playground endseq
startseq dog and dog are running on the grass endseq
startseq boy in pink swimsuit and black black and black black and black black and bla
startseq man with pink orange and white dog and black dog with red toy endseq
startseq man in black and white shirt is walking on the street endseq
startseq two people are walking down snowy hill endseq
startseq dog is jumping into the water endseq
startseq two people are sitting on the window of the window endseq
startseq man is climbing up cliff endseq
BLEU-1: 0.523232
BLEU-2: 0.308826

from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(BASE_DIR,image_name):
# load the image
# image_name = "1001773457_577c3a7d70.jpg"
image_id = image_name.split('.')[0]
img_path = os.path.join(BASE_DIR, image_name)
image = Image.open(img_path)
captions = mapping[image_id]
print('---------------------Actual---------------------')
for caption in captions:
print(caption)
# predict the caption
y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
print('--------------------Predicted--------------------')
print(y_pred)
plt.imshow(image)
generate_caption('Flicker8k_Dataset',"1001773457_577c3a7d70.jpg")

---------------------Actual---------------------
startseq little girl covered in paint sits in front of painted rainbow with her hands
startseq little girl is sitting in front of large painted rainbow endseq
startseq small girl in the grass plays with fingerpaints in front of white canvas wit
startseq there is girl with pigtails sitting in front of rainbow painting endseq
startseq young girl with pigtails painting outside in the grass endseq
--------------------Predicted--------------------
startseq two people are standing on the street endseq
generate_caption('Flicker8k_Dataset',"1002674143_1b742ab4b8.jpg")

generate_caption('Flicker8k_Dataset',"1009434119_febe49276a.jpg")