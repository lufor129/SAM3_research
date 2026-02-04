import streamlit as st
import os
import sys
import numpy as np
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Add parent directory to path to import sam3 modules if not installed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import sam3_utils

# --- Configuration ---
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'sam3.pt')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="SAM3 Cross-Image Detection",
    layout="wide",
)

st.title("SAM3 Cross-Image Few-Shot Detection")
st.markdown("""
Ref: `examples/cross_image_few_shot_detection.ipynb`
This app allows you to:
1. Upload a **Support Image**.
2. Draw a **Bounding Box** around a target object.
3. Upload **Query Images**.
4. Detect similar objects in the query images.
""")

# --- Model Loading ---
@st.cache_resource
def get_model():
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        st.error(f"Checkpoint not found at {CHECKPOINT_PATH}. Please ensure sam3.pt is in the parent directory.")
        return None
        
    with st.spinner(f"Loading SAM3 Model from {CHECKPOINT_PATH}..."):
        try:
            processor = sam3_utils.load_sam3_model(CHECKPOINT_PATH, device=DEVICE)
            return processor
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

processor = get_model()

if not processor:
    st.stop()

# --- Session State ---
if 'visual_prompt_embed' not in st.session_state:
    st.session_state.visual_prompt_embed = None
if 'support_box' not in st.session_state:
    st.session_state.support_box = None

# --- Main Layout ---
col_sidebar = st.sidebar
col_main = st

# --- Sidebar: Controls ---
col_sidebar.header("Controls")
threshold = col_sidebar.slider("Score Threshold", 0.0, 1.0, 0.45, 0.05)

if col_sidebar.button("Reset / Retry", type="primary"):
    st.session_state.visual_prompt_embed = None
    st.session_state.support_box = None
    st.rerun()

# --- Step 1: Upload Images ---
st.header("Step 1: Upload Images")
col1, col2 = st.columns(2)

with col1:
    uploaded_support = st.file_uploader("1. Upload Support Image", type=["jpg", "jpeg", "png"])

with col2:
    uploaded_queries = st.file_uploader("2. Upload Query Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_queries:
        with st.expander(f"Uploaded Query Images ({len(uploaded_queries)})", expanded=True):
            cols_q = st.columns(4)
            for i, q_file in enumerate(uploaded_queries):
                cols_q[i % 4].image(q_file, caption=q_file.name, use_container_width=True)

# --- Step 2: Draw Box & Process ---
if uploaded_support:
    st.divider()
    st.header("Step 2: Draw Box on Support Image")
    
    # Open image
    try:
        support_img = Image.open(uploaded_support).convert("RGB")
        w, h = support_img.size
        
        col_draw, col_info = st.columns([2, 1])
        
        with col_draw:
            st.write("Draw a box around the Object of Interest:")
            
            # If visual prompt is already extracted, show static image with box
            if st.session_state.visual_prompt_embed is not None and st.session_state.support_box:
                 # draw_box is [x, y, w, h]
                draw_box = st.session_state.support_box
                draw_box_xyxy = [draw_box[0], draw_box[1], draw_box[0] + draw_box[2], draw_box[1] + draw_box[3]]
                preview_img = sam3_utils.draw_boxes_pil(support_img, [draw_box_xyxy], color="blue")
                st.image(preview_img, caption="Support Image (Locked)")
                canvas_result = None
            else:
                # Canvas for drawing
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=3,
                    stroke_color="#FF0000",
                    background_image=support_img,
                    update_streamlit=True,
                    height=h,
                    width=w,
                    drawing_mode="rect",
                    # stroke_width=2,
                    key="canvas",
                )

        with col_info:
            if st.session_state.visual_prompt_embed is None:
                if canvas_result and canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if len(objects) > 0:
                        obj = objects[-1]
                        box = [obj["left"], obj["top"], obj["width"], obj["height"]]
                        st.success(f"Box Selected: {box} (x, y, w, h)")
                        
                        if st.button("Confirm & Predict"):
                            if not uploaded_queries:
                                st.warning("Please upload Query Images first (Step 1).")
                            else:
                                with st.spinner("Extracting features & Predicting..."):
                                    try:
                                        # 1. Extract Support Feature
                                        embed = sam3_utils.extract_visual_prompt_embedding(
                                            processor, support_img, box
                                        )
                                        st.session_state.visual_prompt_embed = embed
                                        st.session_state.support_box = box
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Extraction failed: {e}")
                    else:
                        st.info("Draw a box to select the object.")
            else:
                st.success("Target Object Defined!")
                st.write(f"Box: {st.session_state.support_box}")
                
    except Exception as e:
        st.error(f"Error processing support image: {e}")

# --- Step 3: Results ---
if st.session_state.visual_prompt_embed is not None and uploaded_queries:
    st.divider()
    st.header("Step 3: Detection Results")
    
    # Grid display
    cols = st.columns(3)
    
    for i, q_file in enumerate(uploaded_queries):
        try:
            query_img = Image.open(q_file).convert("RGB")
            
            with st.spinner(f"Processing {q_file.name}..."):
                results = sam3_utils.grounding_with_visual_prompt(
                    processor, query_img, st.session_state.visual_prompt_embed
                )
                
            # Filter and Draw
            boxes, scores = sam3_utils.filter_results(results, threshold=threshold)
            
            result_img = sam3_utils.draw_boxes_pil(query_img, boxes, scores)
            
            # Display in grid
            with cols[i % 3]:
                st.image(result_img, caption=f"{q_file.name} ({len(boxes)} detected)")
        except Exception as e:
            st.error(f"Error processing {q_file.name}: {e}")
