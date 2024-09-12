import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.colors as mcolors

# Function to load and preprocess images from a folder
def load_images_from_folder(folder, label, img_size=(224, 224)):
    images = []
    labels = []

    # Only process files with image extensions
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]  # Add more extensions if needed

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)

        # Check if the file is an image based on its extension
        if os.path.isfile(img_path) and os.path.splitext(filename)[1].lower() in valid_image_extensions:
            # Load the image, resize, and preprocess
            img = load_img(img_path, target_size=img_size)  # Load image with the required size
            img = img_to_array(img)  # Convert to array
            img = preprocess_input(img)  # Preprocess for VGG16

            images.append(img)
            labels.append(label)

    images_array = np.array(images)
    labels_array = np.array(labels)

    # Print to debug
    print(f"Loaded {len(images_array)} images from {folder}")

    return images_array, labels_array

# Set the paths to the cats and dogs folders
cats_folder = "C:/Users/mukka/OneDrive/Desktop/task 3/cats"  # Replace with your actual cats folder path
dogs_folder = "C:/Users/mukka/OneDrive/Desktop/task 3/dogs"  # Replace with your actual dogs folder path

# Load the cat images (label 0) and dog images (label 1)
X_cats, y_cats = load_images_from_folder(cats_folder, label=0)
X_dogs, y_dogs = load_images_from_folder(dogs_folder, label=1)

# Check if images were loaded correctly
if X_cats.size == 0 or X_dogs.size == 0:
    raise ValueError("No images were loaded. Check the folder paths and ensure they contain images.")

# Concatenate the cats and dogs images and labels into a single dataset
X = np.concatenate((X_cats, X_dogs), axis=0)
y = np.concatenate((y_cats, y_dogs), axis=0)

# Verify the shape of the dataset
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Load VGG16 model with pre-trained ImageNet weights, excluding the top classification layers
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features from the images using VGG16
try:
    features = vgg_model.predict(X)
except Exception as e:
    print(f"Error during feature extraction: {e}")
    raise

# Reshape the features to be compatible with SVM (flatten the feature maps)
features = features.reshape(features.shape[0], -1)  # Flatten the feature maps into 2D array

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Create the SVM model with a linear kernel
svm = SVC(kernel='linear', random_state=42)

# Train the SVM on the training data
svm.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm.predict(X_test)

# Evaluate the model's performance using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print the classification report to see precision, recall, and F1-score for each class
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Define a custom colormap with distinct colors
colors = [(0.678, 0.847, 0.902), (0.678, 0.902, 0.678), (0.5, 0.5, 0.5)]  # Light blue, light green, grey
n_bins = 3  # Number of distinct colors
cmap_name = 'custom_cmap'
cmaps = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps, xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"], annot_kws={"size": 16})
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.show()
