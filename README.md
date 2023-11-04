# Assignment_3_ADM

## Image Search By an Artistic Style

The project you provided is titled "Neural Style Transfer Image Search." It is a combination of several machine learning and image processing techniques to achieve two main objectives: neural style transfer and image retrieval based on artistic style. Below is an explanation of the entire project's functionality:

1. Neural Style Transfer (NST):

The project begins with the concept of Neural Style Transfer. NST is a deep learning technique that applies the artistic style of one image to the content of another image.
Users can upload their own images and apply the artistic styles of famous artworks to their photos.
The project uses a pre-trained VGG19 model, which is a popular convolutional neural network, to extract style and content features from images.

2. Code Explaination

Importing Libraries: 
The code begins by importing various Python libraries, including TensorFlow for deep learning, Streamlit for the user interface, NumPy for numerical operations, Matplotlib for plotting, OpenCV for image processing, and Scikit-learn for some data analysis and distance calculation.

Loading Image Data: 
The code loads a set of images from a specified directory. These images will be used to compare styles with a reference image.

Defining Content and Style Layers: 
The code defines lists of content and style layers from a VGG19 neural network. These layers are used to extract the content and style information from images.

Loading a VGG Model: 
A pre-trained VGG19 model is loaded from TensorFlow's application module. This model will be used for style extraction.

Defining Functions: 
Several functions are defined in the code:

load_image(image): 
Loads and preprocesses an image for style transfer.
selected_layers_model(layer_names, baseline_model): 
Creates a new model with selected layers for feature extraction.
gram_matrix(input_tensor): 
Calculates the Gram matrix for a given input tensor.
StyleModel: 
Defines a custom model for extracting style and content representations from images.
Extracting Style: 
The code extracts style information from the loaded images and creates style embeddings for each image.

Visualizing Style Embeddings: 
It uses a dimensionality reduction technique (t-SNE) to visualize the style embeddings of the images in a 2D space. This allows for easier comparison and visualization of image styles.

Searching for Similar Images: 
The code provides a function, search_by_style, to find images with similar artistic styles to a given reference image. It computes the cosine similarity between style embeddings to rank the images by similarity.

Streamlit User Interface:
The code sets up a Streamlit user interface where users can upload a reference image. When the "Search for Similar Images" button is clicked, it displays images with similar styles to the reference image.

Running the Application: The code defines the main routine for running the Streamlit application. Users can interact with the app to perform style transfer and image searches.

3. Style Embeddings:

Style features are extracted from a set of reference images. These features describe the unique artistic style of each image.
The style features are represented as "style embeddings."

Image Retrieval by Style:

Users can choose a reference image. The project then searches for other images in a dataset that have similar artistic styles to the reference image.
This process is achieved through the calculation of cosine similarity between style embeddings. Images with lower cosine similarity are considered more similar in artistic style.
4. Streamlit User Interface:

The project provides a user-friendly interface using Streamlit, a Python library for creating web applications.
Users can upload their reference image through the interface.
Upon clicking a button, the project returns a set of images that match the artistic style of the reference image.
The interface allows users to explore and compare images with similar styles to their reference.
5. Image Visualization:

Images are visualized using Matplotlib and presented to the user for comparison.
The project also uses t-SNE, a dimensionality reduction technique, to display images in a 2D space, making it easier for users to see how images are distributed based on style.
6. Project Architecture:

The code structure is organized into different modules, including style_transfer.py, image_search.py, and utils.py, for better code organization and modularity.
7. Data Directory:
The project assumes the presence of a dataset of images in a directory named images-by-style.
In summary, this project offers a practical and interactive tool for users who are interested in exploring the artistic styles of images. It combines neural style transfer, style embeddings, image retrieval, and a user-friendly interface to make it easy for users to apply artistic styles to their images and discover other images with similar styles. It's a creative application of machine learning and image processing techniques that can be both fun and informative for users.



## Visual Search Similarity

1.  Project Description:
This project is an "Image Classification Demo" that utilizes a pre-trained deep learning model for image classification. The primary goal of this project is to provide a user-friendly interface for users to upload their own images and receive predictions about the category or class to which those images belong. Here's what this project is about and the output it provides:



2. Image Classification:

The core functionality of this project is image classification. It leverages a pre-trained deep learning model, which has been trained on a large dataset to recognize patterns in images and make accurate predictions about their content.

3. Code Explaination:






Library Imports:

The code starts by importing various Python libraries and frameworks, including TensorFlow, OpenCV (cv2), NumPy, Pandas, and Matplotlib. These libraries are essential for working with deep learning models, data processing, and visualization.


Data Loading and Preprocessing:

The code loads images and their corresponding category labels from a specified directory using Pandas. It filters the data to keep only specific categories, such as 'Shoes,' 'Dress,' 'Longsleeve,' 'T-Shirt,' and 'Hat.'
Data Preprocessing:

It pre-processes the image data by converting it to NumPy arrays and ensuring that the pixel values are in the float32 data type.
MultiLabelBinarizer:

The MultiLabelBinarizer is used to convert the category labels into binary form, making it suitable for deep learning classification. The y variable stores these binary labels.
Data Split:

The code splits the data into training and testing sets. The x_train and x_test represent the image data, and y_train and y_test contain the corresponding labels.
Visualizing Data:

The code visualizes a sample of the data by displaying a grid of images with their respective labels. It uses Matplotlib for this visualization.
Fine-Tuning a Pretrained Model:

The code loads a pre-trained model, EfficientNetB0, from TensorFlow's model library. EfficientNet is a type of deep learning architecture. The model's layers are then set to non-trainable to prevent changes to the pre-trained weights.
Creating a Custom Model:

A custom classification model is created by adding new layers on top of the pre-trained EfficientNetB0. This model is responsible for predicting the category labels. It consists of a Flatten layer followed by two Dense layers. The final layer uses softmax activation to provide class probabilities.
Compiling the Model:

The model is compiled with specific settings, including the loss function (categorical cross-entropy) and the optimizer (Adam with a learning rate of 0.001).
Training the Model:

The model is trained using the training data (x_train and y_train) for four epochs with a batch size of 4. The validation data (x_test and y_test) are also used to monitor the model's performance during training.
Visualizing Class Probabilities:

The code visualizes the probabilities of different classes for a set of test images. It displays the images and their corresponding class probabilities in a grid.
Embedding Space Visualization:

The code projects image embeddings into a 2D space using t-Distributed Stochastic Neighbor Embedding (t-SNE). This technique reduces the dimensionality of the image embeddings, making it easier to visualize them in a 2D scatter plot.
Building a Streamlit App:

Although not included in this code snippet, it's common to build a Streamlit web application for users to interact with the model. Users can upload images, and the model will provide predictions and visualizations through the app.







4. How to Use the Streamlit App:

Setup:
Make sure you have the required Python libraries installed, which can be done using pip if necessary.

Running the App: 
Execute the Python script containing the Streamlit app by running the following command in your terminal:

streamlit run your_script.py

Accessing the Streamlit App: 
After running the script, the Streamlit app will open in your web browser for you to use.

Image Upload: 
Click on the "Choose an image" button in the app to upload your image. Select an image file in JPG, JPEG, or PNG format.

Viewing Results: 
Once you've uploaded an image, the app will display the image and provide details on the predicted class and class probabilities.

Switching Images: 
If you want to use the app for another image, you can simply return to your web browser and upload a new image.


## Visual Search Variational Autoencoder (VVAE): 
 
Visual Search Variational Autoencoder (VVAE) is a type of deep learning model that combines elements of Variational Autoencoders (VAEs) and visual search technologies to enable the efficient and effective retrieval of visually similar items from a database. VVAEs are primarily used to encode and decode visual information, making them well-suited for tasks like content-based image retrieval and recommendation systems. 
 
Definitions: 
Variational Autoencoder (VAE): A VAE is a generative model that can encode high-dimensional data into a lower-dimensional latent space and then decode it back to the original data space. It is used in applications like image generation, data compression, and feature learning. 
Visual Search: Visual search is a technology that allows users to search for information or products using images as queries instead of text. It can be applied in e-commerce, art recognition, and more. 
 
Use Case Example: 
 
Company: Wayfair 
Use Case: Wayfair, an online home goods retailer, utilizes Visual Search Variational Autoencoders to enhance the shopping experience for their customers. Customers often find it challenging to describe the exact piece of furniture they are looking for with text searches. 
With VVAEs, Wayfair allows users to search for furniture and home decor items by simply uploading images or taking pictures. How it works: Image Upload: Users can take a photo or upload an image of a piece of furniture or decor they like. 
Encoding and Search: The VVAE encodes the uploaded image into a lower-dimensional latent space, which represents the essential features of the item. The system then searches its extensive product catalog for items with similar visual features. Recommendations: Wayfair presents the user with a list of visually similar products, making it easier for them to find items that match their preferences and style. 
 
Benefits: 
Improved User Experience: Customers can find products that match their tastes more easily, even if they can't describe them in words. 
Increased Sales: By providing visually similar recommendations, Wayfair increases the chances of customers finding and purchasing products they love. 
Reduced Search Time: Visual search reduces the time and effort required to find specific items, enhancing customer satisfaction. 
 
Conclusion: 
Visual Search Variational Autoencoders are becoming increasingly vital in various industries, including e-commerce, where the ability to search for products using images rather than text significantly enhances user experiences. Wayfair's implementation of VVAEs demonstrates how this technology can improve product discovery, increase sales, and simplify the customer journey by providing intuitive and visually guided search capabilities. As AI and deep learning continue to advance, VVAEs offer promising solutions for businesses seeking to meet the growing demand for visual search and recommendation systems.

https://codelabs-preview.appspot.com/?file_id=1pah8VgyBOk88BJSHaVjesSvFEEzKHBgiTwdcJfRaV_s#0 
