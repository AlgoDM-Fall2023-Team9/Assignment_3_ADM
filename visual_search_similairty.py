# import streamlit as st
# import tensorflow as tf
# import cv2
# import numpy as np

# # Load the pre-trained model from the H5 file
# model = tf.keras.models.load_model('visual_search.h5')

# # Function to preprocess the uploaded image
# def preprocess_image(image):
#     image = cv2.resize(image, (224, 224))
#     image = image / 255.0  # Normalize the image
#     return image

# # Main Streamlit app
# def main():
#     st.title("Image Classification App")
#     uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

#     if uploaded_image is not None:
#         # Process the uploaded image
#         image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
#         image = preprocess_image(image)

#         # Make predictions using the loaded model
#         class_probs = model.predict(np.expand_dims(image, axis=0))[0]

#         # Display the image
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Display class probabilities
#         st.write("Class Probabilities:")
#         for i, prob in enumerate(class_probs):
#             st.write(f"Class {i}: {prob:.2%}")

# if __name__ == "__main__":
#     main()
# import streamlit as st
# import tensorflow as tf
# import cv2
# import numpy as np

# # Load the pre-trained model from the H5 file
# model = tf.keras.models.load_model('visual_search.h5')

# # Function to preprocess the uploaded image
# def preprocess_image(image):
#     image = cv2.resize(image, (224, 224))
#     image = image / 255.0  # Normalize the image
#     return image

# # Main Streamlit app
# def main():
#     st.title("Image Classification App")
#     uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

#     if uploaded_image is not None:
#         # Process the uploaded image
#         image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
#         image = preprocess_image(image)

#         # Make predictions using the loaded model
#         class_probs = model.predict(np.expand_dims(image, axis=0))[0]

#         # Display the image
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Display class probabilities
#         st.write("Class Probabilities:")
#         for i, prob in enumerate(class_probs):
#             st.write(f"Class {i}: {prob:.2%}")

#         # Debugging info
#         st.write("Debugging Info:")
#         st.write(f"Class Labels (in the order used for the model):")
#         st.write(model.classes_)  # Print the class labels used by the model

# if __name__ == "__main__":
#     main()



import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title('Image Classification Demo')

# Load the model
model = tf.keras.models.load_model('/Users/siddhesh/Downloads/ADM_clip/Assignment_1.2/transfer_model.h5')



# Allow image upload
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load image and convert to numpy array
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))

    image = np.array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Make prediction
    predictions = model.predict(image)


    # Get index of top prediction
    predicted_class = np.argmax(predictions[0])


    
    
    # Map index to class name Dress. Tshirt, Long Sleeve , Hat , Shoes 
   # class_names = ['class1', 'class2', 'Long Sleve','Shoes','class5'] 
    class_names= ['Dress' ,'Hat' ,'Longsleeve' ,'Shoes' ,'T-Shirt']

    predicted_class_name = class_names[predicted_class]
    
    # Show image and prediction
    st.image(image, use_column_width=True)
    st.write(f'Predicted class: {predicted_class_name}')

    # Make prediction 
    #predictions = model.predict(image)

    # Get prediction probabilities for each class
    prediction_probs = predictions[0]

    # Map indexes to class names
    #class_names = ['class1', 'class2', 'class3','class4','class5'] 

    # Display probabilities for each class  
    for i in range(len(class_names)):
      st.write(f'{class_names[i]}: {100 * prediction_probs[i]:.2f}%') 

    # Or to return as dictionary:
    pred_dict = dict(zip(class_names, prediction_probs))