import string
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from keras.models import load_model
from keras.applications.xception import Xception
from keras.preprocessing.sequence import pad_sequences

def load_doc(filename):
    """
    Load the content of a text file into a string.
    
    Args:
        filename (str): Path to the file.
    
    Returns:
        str: Content of the file as a string.
    """
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def all_img_captions(filename):
    """
    Extracts all image IDs and their corresponding captions from the given file.

    Args:
        filename (str): Path to the file containing image captions.
    
    Returns:
        dict: A dictionary where keys are image IDs and values are lists of captions.
    """
    file = load_doc(filename) 
    captions = file.split('\n') 
    descriptions = {}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions

def cleaning_text(captions):
    """
    Cleans image captions by converting text to lowercase, removing punctuation, short words, and numeric tokens.

    Args:
        captions (dict): Dictionary containing image IDs and their associated captions.
    
    Returns:
        dict: Cleaned captions with unnecessary characters removed.
    """
    table = str.maketrans('', '', string.punctuation)  # Translation table to remove punctuation
    for img, caps in captions.items(): 
        for i, img_caption in enumerate(caps):
            img_caption = img_caption.replace("-", " ")  # Replace dashes with spaces
            desc = img_caption.lower().split()  # Convert to lowercase and split into words
            desc = [word.translate(table) for word in desc]  # Remove punctuation
            desc = [word for word in desc if len(word) > 1]  # Remove short words
            desc = [word for word in desc if word.isalpha()]  # Remove tokens with numbers
            captions[img][i] = ' '.join(desc)  # Join the tokens back into a string
    return captions

def text_vocabulary(descriptions):
    """
    Builds a vocabulary set containing all unique words from the captions.

    Args:
        descriptions (dict): Dictionary containing image descriptions.
    
    Returns:
        set: Set of unique words (vocabulary).
    """
    vocab = set()  # Initialize an empty set for unique words
    for key in descriptions.keys():
        for d in descriptions[key]:
            vocab.update(d.split())  # Add words to vocabulary
    return vocab

def save_descriptions(descriptions, filename):
    """
    Saves image descriptions to a file in the format 'image_id\tcaption'.

    Args:
        descriptions (dict): Dictionary containing image descriptions.
        filename (str): Path to the file where descriptions will be saved.
    """
    lines = []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(f"{key}\t{desc}")
    data = "\n".join(lines)
    with open(filename, "w") as file:
        file.write(data)

def extract_features(directory):
    """
    Extracts deep learning features for all images in a directory using the Xception model.

    Args:
        directory (str): Path to the directory containing images.
    
    Returns:
        dict: A dictionary where keys are image filenames and values are the extracted features (2048-dimensional vectors).
    """
    model = Xception(include_top=False, pooling='avg')
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = os.path.join(directory, img)
        image = Image.open(filename).resize((299, 299))
        image = np.expand_dims(np.array(image) / 127.5 - 1.0, axis=0)
        feature = model.predict(image)
        features[img] = feature
    return features

def extract_features(filename, model):
    """
    Extracts deep learning features for a single image using a pre-trained model.

    Args:
        filename (str): Path to the image file.
        model: Pre-trained model for feature extraction.
    
    Returns:
        np.array: Extracted feature vector for the image, or None if an error occurs.
    """
    try:
        image = Image.open(filename).resize((299, 299))
    except:
        print("ERROR: Couldn't open image! Ensure the image path and extension are correct.")
        return None
    
    image = np.array(image)
    if image.shape[2] == 4:  # Convert 4-channel images to 3-channel
        image = image[..., :3]
    
    image = np.expand_dims(image / 127.5 - 1.0, axis=0)
    return model.predict(image)

def word_for_id(integer, tokenizer):
    """
    Retrieves the word corresponding to a given integer index from the tokenizer.

    Args:
        integer (int): The integer index of the word.
        tokenizer: Tokenizer object containing the word index.
    
    Returns:
        str: The word corresponding to the integer index, or None if the index is not found.
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    """
    Generates a caption for a given photo using a trained model and tokenizer.

    Args:
        model: The trained caption generation model.
        tokenizer: Tokenizer used to convert words to sequences.
        photo (np.array): Extracted feature vector for the image.
        max_length (int): Maximum length of the generated caption.
    
    Returns:
        str: The generated caption for the image.
    """
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = np.argmax(model.predict([photo, sequence], verbose=0))
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text
