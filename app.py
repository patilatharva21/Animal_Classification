import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

st.set_page_config(page_title="🐾 Animal Classification", layout="centered")


st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🐾 Animal Classifier"])


CLASSES = ["Buffalo", "Elephant", "Rhino", "Zebra"]
MODEL_PATH = "model/best_model.pth"

mean = [0.5200, 0.4986, 0.4165]
std = [0.2550, 0.2467, 0.2490]
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
])

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device


if page == "🏠 Home":
    st.title("🐾 Animal Classification Project")
    st.image("input_images/home.jpeg", use_container_width=True)

    st.markdown("""
    ## 🧠 Project Overview
    This is an **Animal Classification System** built using **Convolutional Neural Networks (CNNs)** in **PyTorch** and deployed with **Streamlit**.  
    The model classifies animal images into **four categories**:
    - 🐃 Buffalo  
    - 🐘 Elephant  
    - 🦏 Rhino  
    - 🦓 Zebra  

    ---

    ## ⚙️ Model Information
    - **Base Architecture:** ResNet-18  
    - **Optimizer:** SGD (lr=0.01, momentum=0.9, weight_decay=0.003)  
    - **Loss Function:** CrossEntropyLoss  
    - **Epochs:** 40  
    - **Normalization:** mean = [0.5200, 0.4986, 0.4165], std = [0.2550, 0.2467, 0.2490]  
    - **Trained on:** Custom animal dataset  

    ---

    ## 🧩 Technologies Used
    - Python 🐍  
    - PyTorch 🔥  
    - Torchvision 🖼️  
    - Streamlit 🌐  
    - PIL & NumPy  

    ---

    👉 Use the sidebar to navigate to the **Animal Classifier** page and try out the model yourself!
    """)

    # st.write("---")
    # st.caption("Developed by Atharva Patil • Powered by PyTorch & Streamlit")

# CLASSIFIER PAGE 
elif page == "🐾 Animal Classifier":
    st.title("🐾 Animal Classifier")

    model, device = load_model()

    uploaded_file = st.file_uploader("📁 Upload an animal image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Classify Animal"):
            with st.spinner("Classifying..."):
                input_image = image_transforms(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_image)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                    confidence = probs[0][predicted.item()].item() * 100

                result_class = CLASSES[predicted.item()] 

            if confidence < 75 or confidence > 97:
                st.warning(f"Cannot classify this image.")         
            else: 
                st.success(f"**Prediction:** {result_class} ({confidence:.2f}% confidence) 🐾")
    else:
        st.info("👆 Upload an image to start classification.")

    # st.write("---")
    # st.caption("Developed by Atharva Patil • Powered by PyTorch & Streamlit")
