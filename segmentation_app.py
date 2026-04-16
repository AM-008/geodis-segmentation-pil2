import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
from segmentation import segmentation_from_ui, visualize_segmentation
from io import BytesIO


def extract_seeds_from_canvas(canvas_result, original_image_size):
    """
    Extract blue and red seed points from canvas drawings.
    
    Args:
        canvas_result: Result object from st_canvas
        original_image_size: Tuple (width, height) of the original image
    
    Returns:
        tuple: (blue_seeds, red_seeds) - lists of (x, y) coordinates
    """
    blue_seeds = []
    red_seeds = []
    
    if canvas_result.json_data is None:
        return blue_seeds, red_seeds
    
    objects = canvas_result.json_data.get("objects", [])
    
    for obj in objects:
        # Extract color to determine if blue or red
        stroke_color = obj.get("stroke", "black")
        
        # Simple color detection
        is_blue = "0000ff" in stroke_color.lower() or stroke_color.lower() == "blue"
        is_red = "ff0000" in stroke_color.lower() or stroke_color.lower() == "red"
        
        # Handle point objects
        if obj.get("type") == "point":
            x = int(obj.get("left", 0) + obj.get("radius", 0))
            y = int(obj.get("top", 0) + obj.get("radius", 0))
            
            if is_blue:
                blue_seeds.append((x, y))
            elif is_red:
                red_seeds.append((x, y))
        
        # Handle circle objects
        elif obj.get("type") == "circle":
            x = int(obj.get("left", 0) + obj.get("radius", 0))
            y = int(obj.get("top", 0) + obj.get("radius", 0))
            
            if is_blue:
                blue_seeds.append((x, y))
            elif is_red:
                red_seeds.append((x, y))
        
        # Handle rectangle objects - use center
        elif obj.get("type") == "rect":
            x = int(obj.get("left", 0) + obj.get("width", 0) / 2)
            y = int(obj.get("top", 0) + obj.get("height", 0) / 2)
            
            if is_blue:
                blue_seeds.append((x, y))
            elif is_red:
                red_seeds.append((x, y))
    
    return blue_seeds, red_seeds


def segmentation_app():
    st.title("🎨 Geodesic Image Segmentation Tool")
    st.markdown("""
    Mark regions of interest on your image and let the segmentation algorithm identify them using geodesic distances.
    - **Blue strokes/dots**: Mark the regions you want to segment (foreground)
    - **Red strokes/dots**: Mark the background regions
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        drawing_mode = st.selectbox(
            "Drawing tool:",
            ("point", "circle", "rect", "line", "freedraw"),
            help="Choose how to mark seed regions"
        )
        
        stroke_width = st.slider("Stroke width:", 1, 25, 3)
        
        if drawing_mode in ["point", "circle"]:
            point_display_radius = st.slider("Point/circle radius:", 1, 25, 5)
        else:
            point_display_radius = 0
        
        stroke_color = st.color_picker("Stroke color hex: ", "#0000ff")
        
        gradient_sigma = st.slider(
            "Gradient smoothing sigma:",
            0.5, 5.0, 1.0,
            help="Higher values smooth the image more, affecting edge detection"
        )
        
        realtime_update = st.checkbox("Update canvas in realtime", True)
        display_toolbar = st.checkbox("Display drawing toolbar", True)
        
        st.markdown("---")
        st.markdown("**Drawing Instructions:**")
        st.markdown("1. Upload an image")
        st.markdown("2. Draw **blue** marks on regions to segment")
        st.markdown("3. Draw **red** marks on background regions")
        st.markdown("4. Click 'Run Segmentation'")
        st.markdown("5. View results in the visualization")
    
    # Image upload
    st.subheader("1. Load Image")
    uploaded_file = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg", "bmp", "gif"])
    
    if uploaded_file is not None:
        # Load image as PIL Image (same way as app.py)
        image = Image.open(uploaded_file).convert("RGB")
        image_width = image.width
        image_height = image.height
        
        st.write(f"Image size: {image_width}×{image_height} pixels")
        
        # Canvas for drawing
        st.subheader("2. Draw Seed Points")
        
        # Display the image first
        st.image(image, caption="Original image", use_column_width=False, width=min(image_width, 800))
        
        # Create canvas without background image (workaround for deprecated image_to_url)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="white",
            update_streamlit=realtime_update,
            height=min(image_height, 600),
            width=min(image_width, 800),
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius,
            display_toolbar=display_toolbar,
            key="segmentation_canvas",
        )
        
        st.info("💡 Tip: Use the color picker in the toolbar to switch between blue and red strokes")
        
        # Segmentation control
        st.subheader("3. Run Segmentation")
        
        col_run, col_clear = st.columns(2)
        with col_run:
            run_segmentation = st.button("🚀 Run Segmentation", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.rerun()
        
        # Perform segmentation
        if run_segmentation:
            try:
                # Extract seed points from canvas
                blue_seeds, red_seeds = extract_seeds_from_canvas(
                    canvas_result,
                    (image_width, image_height)
                )
                
                if not blue_seeds and not red_seeds:
                    st.warning("⚠️ Please draw at least one seed point (blue or red) on the image")
                else:
                    with st.spinner("Processing segmentation..."):
                        # Run segmentation using same interface as ui.py
                        seg_results = segmentation_from_ui(
                            image=image,
                            blue_dots=blue_seeds,
                            red_dots=red_seeds,
                            gradient_sigma=gradient_sigma
                        )
                        
                        st.session_state.seg_results = seg_results
                        st.session_state.blue_seeds = blue_seeds
                        st.session_state.red_seeds = red_seeds
                        st.session_state.image = image
                        
                        st.success("✅ Segmentation completed!")
            
            except Exception as e:
                st.error(f"❌ Error during segmentation: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        # Display results
        if "seg_results" in st.session_state:
            st.subheader("4. Segmentation Results")
            
            # Display seed points info
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Blue seed points", len(st.session_state.blue_seeds))
            with col_info2:
                st.metric("Red seed points", len(st.session_state.red_seeds))
            
            # Generate visualization
            with st.spinner("Generating visualization..."):
                vis_image = visualize_segmentation(
                    original_image=st.session_state.image,
                    segmentation_results=st.session_state.seg_results
                )
            
            st.image(vis_image, use_column_width=True, caption="Segmentation Visualization (6 panels)")
            
            st.markdown("""
            **Visualization panels:**
            1. **Original image** (top-left)
            2. **Segmentation result** - Blue=foreground, Red=background (top-center)
            3. **Overlay on original** (top-right)
            4. **Blue distance map** - Distance from foreground seeds (bottom-left)
            5. **Red distance map** - Distance from background seeds (bottom-center)
            6. **Confidence map** - Segmentation confidence (bottom-right)
            """)
            
            # Export options
            st.subheader("Export Results")
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                # Export visualization
                vis_buffer = BytesIO()
                vis_image.save(vis_buffer, format="PNG")
                vis_buffer.seek(0)
                st.download_button(
                    label="📥 Download Visualization",
                    data=vis_buffer,
                    file_name="segmentation_visualization.png",
                    mime="image/png"
                )
            
            with col_exp2:
                # Export label map
                label_map = st.session_state.seg_results['label_map']
                label_img = Image.fromarray((label_map * 255).astype(np.uint8))
                label_buffer = BytesIO()
                label_img.save(label_buffer, format="PNG")
                label_buffer.seek(0)
                st.download_button(
                    label="📥 Download Label Map",
                    data=label_buffer,
                    file_name="segmentation_labels.png",
                    mime="image/png"
                )
    
    else:
        st.info("👆 Upload an image to get started!")


def main():
    st.set_page_config(page_title="Geodesic Image Segmentation", layout="wide")
    
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    
    segmentation_app()


if __name__ == "__main__":
    main()
