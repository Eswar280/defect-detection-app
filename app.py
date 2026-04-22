import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO("best.pt")

st.title("STEEL SURFACE DEFECT DETECTION")

# -------------------------------
# CAUSES & REMEDIES
# -------------------------------
data = {
    "crazing": {
        "cause": """- Thermal stresses due to rapid or non-uniform cooling  
- Residual stresses from improper heat treatment or rolling""",
        "remedy": """- Implement controlled cooling  
- Optimize heat treatment cycles  
- Maintain uniform temperature during processing"""
    },

    "inclusion": {
        "cause": """- Non-metallic impurities from poor raw materials  
- Inadequate refining and filtration""",
        "remedy": """- Use high-purity raw materials  
- Ensure proper slag removal and clean handling"""
    },

    "patches": {
        "cause": """- Uneven surface due to improper rolling/coating  
- Oxide layer formation due to poor cleaning""",
        "remedy": """- Improve surface preparation  
- Ensure uniform rolling pressure  
- Remove oxide scale effectively"""
    },

    "pitted_surface": {
        "cause": """- Localized corrosion due to moisture/chemicals  
- Poor storage conditions""",
        "remedy": """- Apply protective coatings  
- Store in low humidity environments  
- Use proper surface treatment"""
    },

    "rolled-in_scale": {
        "cause": """- Oxide scale not removed before rolling  
- Ineffective descaling process""",
        "remedy": """- Remove oxide layers before rolling  
- Maintain proper process control"""
    },

    "scratches": {
        "cause": """- Mechanical damage from worn tools  
- Improper handling or contact with hard surfaces""",
        "remedy": """- Regular tool maintenance  
- Improve handling systems"""
    }
}

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# -------------------------------
# AUTO DETECTION
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # RUN MODEL
    results = model(image, conf=0.5)

    annotated_frame = results[0].plot()
    st.image(annotated_frame, caption="Detected Image", use_container_width=True)

    st.subheader("Detection Results")

    if len(results[0].boxes) == 0:
        st.success("No defects detected")
    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id].lower()

            cause = data.get(label, {}).get("cause", "Not available")
            remedy = data.get(label, {}).get("remedy", "Not available")

            # ---------------- OUTPUT ----------------
            st.markdown(f"### {label.upper()}")
            st.write(f"Confidence: {conf:.2f}")

            st.write("Cause:")
            st.markdown(cause)

            st.write("Remedy:")
            st.markdown(remedy)

            st.markdown("---")