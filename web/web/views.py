from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageForm

from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from PIL import Image

tokenizer = load(open('C:\\Users\\ACER\\Desktop\\College files\\AI\\project\\Django vala\\Image-Caption-Generator\\tokenizer.pkl', 'rb'))
model = load_model('C:\\Users\\ACER\\Desktop\\College files\\AI\project\\Django vala\\Image-Caption-Generator\\model_19.h5')

def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            file_name = img_obj.image
            max_length = 33
            photo = extract_features('C:/Users/ACER/Desktop/College files/AI/project/Django vala/Image-Caption-Generator/web/media/'+str(file_name))
            description = generate_desc(model, tokenizer, photo, max_length)
            description = clean_description(description)
            return render(request, 'index.html', {'form': form, 'img_obj': img_obj,
        							'description':description
            	})
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})

# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def clean_description(desc):
	words = ['startseq', 'endseq']
	desc = [word for word in desc.split() if word not in words]
	desc = ' '.join(desc)
	return desc

