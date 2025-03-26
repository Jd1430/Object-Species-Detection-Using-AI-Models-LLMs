import torch
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
from torchvision import models, transforms
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import time
import os

# Load YOLOv8 for Object Detection
yolo_model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model (lightweight & fast)

# Load iNaturalist Model for Animal Species Classification
inat_model = models.inception_v3(pretrained=True)
inat_model.eval()

# Define image transformations for iNaturalist model
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit UI
st.title("Animal Species & Object Detection App")
st.sidebar.header("Settings")
option = st.sidebar.radio("Choose input source:", ("Upload Image", "Live Camera", "Live Video Recording"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert image to OpenCV format
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Object Detection using YOLOv8
        results = yolo_model(img_cv)
        
        # Convert image back to PIL for annotation
        image_annotated = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_annotated)
        font = ImageFont.load_default()
        
        st.write("### Object Detection Results:")
        for result in results:
            for box in result.boxes:
                class_name = yolo_model.names[int(box.cls)]
                confidence = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 10), f"{class_name} ({confidence:.2f})", fill="black", font=font)
                st.write(f"Detected: {class_name} with confidence {confidence:.2f}")
        
        # Animal Species Classification using iNaturalist
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = inat_model(img_tensor)
        _, predicted = outputs.max(1)
        species_name = f"Species ID: {predicted.item()}"
        draw.text((10, 10), species_name, fill="blue", font=font)
        st.write("### Animal Species Prediction:")
        st.write(species_name)
        
        # Show annotated image
        st.image(image_annotated, caption="Detected Objects & Species", use_column_width=True)

elif option in ["Live Camera", "Live Video Recording"]:
    st.write("### Live Camera Feed")
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])
    save_frame = st.sidebar.button("Save Frame")
    record_video = st.sidebar.checkbox("Record Video")
    
    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()  # Track frame processing time
        
        # Object Detection
        results = yolo_model(frame)
        
        # Annotate the frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = yolo_model.names[int(box.cls)]
                confidence = box.conf.item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)
        
        # Save the detected frame
        if save_frame:
            cv2.imwrite("detected_frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            st.sidebar.success("Frame saved as detected_frame.jpg")
        
        # Save video recording
        if record_video:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Frame rate optimization
        time.sleep(max(0, 0.1 - (time.time() - start_time)))  # Limit FPS
    
    cap.release()
    if record_video:
        out.release()
        st.sidebar.success("Video saved as output.avi")
