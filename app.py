from helpers import *

# Define dataset paths for text and image data
dataset_text = "D:/dataflair_projects/Project_Image_Caption_Generator/Flickr_8k_text"
dataset_images = "D:/dataflair_projects/Project_Image_Caption_Generator/Flicker8k_Dataset"

# Load and process text data for image captions
filename = os.path.join(dataset_text, "Flickr8k.token.txt")
descriptions = all_img_captions(filename)  # Extract image captions
print("Length of descriptions =", len(descriptions))

# Clean descriptions and build vocabulary from the cleaned captions
vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary =", len(vocabulary))

# Save the cleaned descriptions to a text file for later use
save_descriptions(clean_descriptions, "descriptions.txt")

# Extract deep learning features from all images in the dataset using the Xception model
features = extract_features(dataset_images)

# Set up command line argument parsing for the image path input
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")  # Argument for image path
args = vars(ap.parse_args())
img_path = args['image']  # Retrieve the image path from the command line argument

# Load the pre-trained model and tokenizer for generating captions
max_length = 32  # Maximum length of the generated captions
tokenizer = load(open("tokenizer.p", "rb"))  # Load the saved tokenizer
model = load_model('models/model_9.h5')  # Load the pre-trained image captioning model
xception_model = Xception(include_top=False, pooling="avg")  # Pre-trained Xception model for feature extraction

# Extract features for the provided input image using the Xception model
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)  # Open the input image using PIL

# Generate a description (caption) for the input image using the model and tokenizer
description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)  # Print the generated description

# Display the input image using matplotlib
plt.imshow(img)
plt.show()
