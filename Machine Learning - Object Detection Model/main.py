import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

#############Logic behind the code############################
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT #this will use the default COCO dataset.
categories = weights.meta["categories"] #this will involve all the background descriptions of the object like cars,motor, etc.
img_preprocess = weights.transforms() #prepare the image for the model by transforming it.

@st.cache_resource # it is for streamlit to cache the results based on input parameters, it basically tells streamlit to store the results of the functions

def load_model(): #create function load_model()
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8) #create instance of fasterRCNN together with the pre-trained weights, treshold sets the limit for considering bounding boxes predicts.
    model.eval() #tells the model to not change its training or the learned parameters or update the weights
    return model

model = load_model() 

#preprocess img and ready for predictions
def prediction(img):#create prediction function with a parameter 'img'
    img_processed = img_preprocess(img) #preprocess the image for the model (preparing)
    predict = model(img_processed.unsqueeze(0))#makes an extra batch of the preprocessed img as object detection models want images in batches, this basically prepares the preprocessed img into the model for predicts
    predict = predict[0]#selects the first element from the list
    predict["labels"] = [categories[label] for label in predict["labels"]] #updates the predicted labels by mapping the numerical labels obtained from the model using 'categories' mapping.
    return predict

#display bounding boxes after predict
def create_image_with_boxes(img, predict): #create function create_image_with_boxes with parameter img, predict
    img_tensor = torch.tensor(img)#converts the img to PyTorch tensor.
    #the actual box
    img_with_boxes = draw_bounding_boxes(img_tensor, boxes=predict["boxes"], labels=predict["labels"],
                                         colors=["red" if label == "person" else "green" for label in predict["labels"]],
                                         width=4)  # Increase the width of the bounding boxes for better visibility
    create_image_with_boxes_np = img_with_boxes.detach().numpy().transpose(1, 2, 0)#extracts the img with bounding boxes from the converted PyTorch img and converts it back to Numpy array. the numbers represent the dimensions and rearrange it for img visualization purposes i.e PyTorch to standard img
    return create_image_with_boxes_np



#loop through the detected bounding box and labels then convert the box coords to integers and draw bounding box
def display_predict(img, predict):#create display_predict with parameters img, predict
    img_with_boxes = np.array(img) #to input image as Numpy array (used for visualizing the bounding boxes and labels)
    for box, label in zip(predict["boxes"], predict["labels"]): #loop through predicts
        box = [int(coord) for coord in box] #converts the 'box' to an integer
        img_with_boxes = cv2.rectangle(img_with_boxes, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2) #uses OpenCV to draw bounding box on the image array "img_with_boxes". the rectangle box is rgb 255,0,0 and the pixel is 2
        img_with_boxes = cv2.putText(img_with_boxes, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)#displays text labels into the img array "img_with_boxes"text size is set to 0.9, in blue color (255,0,0) with a size of 2 pixel.
    return img_with_boxes
#############Logic behind the code############################



####################Main Content of the Webb App####################################
# Title and File Upload Section
st.title("Object Detection Model")
# Upload Image Section
st.header("Upload Image Here:")
upload = st.file_uploader(label="", type=["png", "jpg", "jpeg"])
if upload:
    img = Image.open(upload)
    predict = prediction(img) #calls the prediction function with the parameter img
    img_with_box = create_image_with_boxes(np.array(img).transpose(2, 0, 1), predict)#calls create_image_with_boxes with various parameters in it
    st.image(img, caption='Uploaded Image', use_column_width=True)

    st.write("") #empty space for layout adjustment

    # Object Detection Results 
    st.header("Object Detection Results")
    col1, col2 = st.columns([1, 2]) #create 2 columns to divide them.

    with col2:
        fig = plt.figure(figsize=(12, 12))#figure size of plt
        ax = fig.add_subplot(111)#add sublot for the figure
        plt.imshow(img_with_box)#show original img with bounding boxes around detected objects.
        plt.xticks([], [])#to make the plot clean
        plt.yticks([], [])
        ax.spines[["top", "bottom", "right", "left"]].set_visible(False)#hide the frames around the plot
        st.pyplot(fig, use_container_width=True)#tells streamlit to render plt figure, and adjust the plot to fit within streamlit

    with col1:
        del predict["boxes"]#to remove redundancy
        st.header("Predicted Probabilities")#display header 
         # Displaying Scores/Probabilities
        st.write("Labels and Probabilities:")
        for label, score in zip(predict["labels"], predict["scores"].tolist()):#iterate through the labels and scores, displaying them in a more organized and readable format.
            st.write(f"- {label}: {score:.4f}")




