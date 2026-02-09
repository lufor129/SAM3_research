import streamlit as st
import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw

# Add parent directory to path to import sam3 modules if not installed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from streamlit_image_coordinates import streamlit_image_coordinates
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
2. **Click twice** to draw a Bounding Box (top-left → bottom-right).
3. Upload **Query Images**.
4. Detect similar objects with **Attention Visualization**.
""")

# --- Model Loading ---
@st.cache_resource
def get_model():
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
if 'click_1' not in st.session_state:
    st.session_state.click_1 = None
if 'click_2' not in st.session_state:
    st.session_state.click_2 = None
if 'show_attention' not in st.session_state:
    st.session_state.show_attention = True
if 'click_count' not in st.session_state:
    st.session_state.click_count = 0
if 'last_coords' not in st.session_state:
    st.session_state.last_coords = None
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- Sidebar: Controls ---
col_sidebar = st.sidebar
col_sidebar.header("Controls")
threshold = col_sidebar.slider("Score Threshold", 0.0, 1.0, 0.45, 0.05)
st.session_state.show_attention = col_sidebar.checkbox("Show Attention Maps", value=True)

if col_sidebar.button("Reset / Retry", type="primary"):
    # Clear all states
    st.session_state.visual_prompt_embed = None
    st.session_state.support_box = None
    st.session_state.click_1 = None
    st.session_state.click_2 = None
    st.session_state.click_count = 0
    st.session_state.last_coords = None
    # Increment uploader key to force file uploader reset
    st.session_state.uploader_key += 1
    # Clear cached data
    st.cache_data.clear()
    st.rerun()

# --- Helper: Draw box on image ---
def draw_box_on_image(image, click_1=None, click_2=None, box_xywh=None, color="red"):
    """Draw box on image based on clicks or box coordinates."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    if box_xywh is not None:
        x, y, w, h = box_xywh
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
    elif click_1 and click_2:
        x1, y1 = click_1
        x2, y2 = click_2
        # Ensure correct order
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
    elif click_1:
        # Draw first point as a marker
        x, y = click_1
        draw.ellipse([x-8, y-8, x+8, y+8], fill="red", outline="white", width=2)
        # Draw crosshair
        draw.line([x-15, y, x+15, y], fill="white", width=2)
        draw.line([x, y-15, x, y+15], fill="white", width=2)
        
    return img_copy

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
    st.header("Step 2: Click to Draw Box on Support Image")
    
    # Maximum display size to prevent overflow
    MAX_DISPLAY_WIDTH = 800
    
    try:
        support_img = Image.open(uploaded_support).convert("RGB")
        orig_w, orig_h = support_img.size
        
        # Calculate scale factor for display
        if orig_w > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / orig_w
            display_w = MAX_DISPLAY_WIDTH
            display_h = int(orig_h * scale)
        else:
            scale = 1.0
            display_w, display_h = orig_w, orig_h
        
        col_draw, col_info = st.columns([2, 1])
        
        with col_draw:
            # If visual prompt is already extracted, show static image with box
            if st.session_state.visual_prompt_embed is not None and st.session_state.support_box:
                draw_box = st.session_state.support_box
                preview_img = draw_box_on_image(support_img, box_xywh=draw_box, color="blue")
                # Resize for display
                if scale < 1.0:
                    preview_img = preview_img.resize((display_w, display_h), Image.LANCZOS)
                st.image(preview_img, caption="Support Image (Locked)")
            else:
                st.write("**Click twice**: First click = top-left corner, Second click = bottom-right corner")
                
                # Draw current state on image (use original size for accurate drawing)
                display_img = draw_box_on_image(
                    support_img, 
                    click_1=st.session_state.click_1, 
                    click_2=st.session_state.click_2
                )
                
                # Resize for display if needed
                if scale < 1.0:
                    display_img_resized = display_img.resize((display_w, display_h), Image.LANCZOS)
                else:
                    display_img_resized = display_img
                
                # Use unique key based on click count to force new coordinates
                key = f"support_coords_{st.session_state.click_count}"
                
                # Use streamlit_image_coordinates for click detection
                coords = streamlit_image_coordinates(
                    display_img_resized,
                    key=key
                )
                
                # Handle clicks - only process if coords changed
                if coords is not None:
                    # Scale coordinates back to original image size
                    display_x, display_y = coords["x"], coords["y"]
                    original_x = int(display_x / scale)
                    original_y = int(display_y / scale)
                    current_coords = (original_x, original_y)
                    
                    # Check if this is a new click (different from last processed)
                    if current_coords != st.session_state.last_coords:
                        st.session_state.last_coords = current_coords
                        st.session_state.click_count += 1
                        
                        if st.session_state.click_1 is None:
                            st.session_state.click_1 = current_coords
                        elif st.session_state.click_2 is None:
                            st.session_state.click_2 = current_coords
                        
                        st.rerun()

        with col_info:
            if st.session_state.visual_prompt_embed is None:
                # Show click status
                if st.session_state.click_1 is None:
                    st.info("👆 Click on the **top-left** corner of the target object")
                elif st.session_state.click_2 is None:
                    st.info(f"✓ First corner: {st.session_state.click_1}\n\n👆 Now click on the **bottom-right** corner")
                else:
                    # Both clicks done - calculate box
                    x1, y1 = st.session_state.click_1
                    x2, y2 = st.session_state.click_2
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    box = [x_min, y_min, x_max - x_min, y_max - y_min]  # [x, y, w, h]
                    
                    st.success(f"✓ Box: {box} (x, y, w, h)")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        if st.button("🔄 Redraw"):
                            st.session_state.click_1 = None
                            st.session_state.click_2 = None
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("✅ Confirm & Predict", type="primary"):
                            if not uploaded_queries:
                                st.warning("Please upload Query Images first (Step 1).")
                            else:
                                with st.spinner("Extracting features..."):
                                    try:
                                        embed = sam3_utils.extract_visual_prompt_embedding(
                                            processor, support_img, box
                                        )
                                        st.session_state.visual_prompt_embed = embed
                                        st.session_state.support_box = box
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Extraction failed: {e}")
            else:
                st.success("✓ Target Object Defined!")
                st.write(f"Box: {st.session_state.support_box}")
                
    except Exception as e:
        st.error(f"Error processing support image: {e}")

# --- Step 3: Results with Attention ---
if st.session_state.visual_prompt_embed is not None and uploaded_queries:
    st.divider()
    st.header("Step 3: Detection Results")
    
    if st.session_state.show_attention:
        st.info("🔥 Attention maps show which regions the model focuses on based on the support object")
    
    for i, q_file in enumerate(uploaded_queries):
        try:
            query_img = Image.open(q_file).convert("RGB")
            
            st.subheader(f"📷 {q_file.name}")
            
            with st.spinner(f"Processing {q_file.name}..."):
                # Use attention-enabled grounding
                results, attention_info = sam3_utils.grounding_with_visual_prompt_and_attention(
                    processor, query_img, st.session_state.visual_prompt_embed
                )
                
            # Filter and Draw
            boxes, scores = sam3_utils.filter_results(results, threshold=threshold)
            result_img = sam3_utils.draw_boxes_pil(query_img, boxes, scores)
            
            # Display results
            if st.session_state.show_attention:
                # Create attention overlay
                attention_overlay, _ = sam3_utils.create_attention_overlay(attention_info, query_img)
                
                cols = st.columns(2)
                with cols[0]:
                    st.image(result_img, caption=f"Detections ({len(boxes)} found)", use_container_width=True)
                with cols[1]:
                    st.image(attention_overlay, caption="Attention Heatmap", use_container_width=True)
            else:
                st.image(result_img, caption=f"{q_file.name} ({len(boxes)} detected)", use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing {q_file.name}: {e}")
