import streamlit as st
import ollama
import io
import base64
import pandas as pd
from PIL import Image
from datetime import datetime
import csv
import json
import os

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="Curiosity AI Scans",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import PyMuPDF for PDF support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("PDF support requires PyMuPDF. Install it with: pip install pymupdf")

# --- HELPER FUNCTIONS ---

def resize_image(image, max_size=1920):
    """Resize an image while maintaining aspect ratio"""
    width, height = image.size
    
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    return image

def image_to_base64(image):
    """Convert PIL Image to base64 encoded string"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

def query_ollama(prompt, image_base64, model):
    """Query Ollama with an image and prompt"""
    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_base64]
        }]
    )
    return response['message']['content']

def extract_structured_data(content, fields):
    """Extract structured data from text content"""
    structured_data = {}
    
    try:
        # Try to find JSON content between ```json and ``` markers
        if "```json" in content and "```" in content.split("```json")[1]:
            json_str = content.split("```json")[1].split("```")[0].strip()
            structured_data.update(json.loads(json_str))
        else:
            # Try to find any JSON object in the text
            json_str = content
            for field in fields:
                field_clean = field.strip()
                if f'"{field_clean}"' in json_str or f"'{field_clean}'" in json_str:
                    try:
                        structured_data.update(json.loads(json_str))
                        break
                    except:
                        pass
    except:
        pass
            
    return structured_data

def process_image(image, filename, fields=None, model=None):
    """Process an image with optional field extraction"""
    # Resize image and convert to base64
    img_base64 = image_to_base64(resize_image(image))
    
    if fields is None:
        # General description mode
        prompt = 'Describe what you see in this image in detail.'
        content = query_ollama(prompt, img_base64, model)
        return {'filename': filename, 'description': content}, content, None
    else:
        # Custom field extraction mode
        fields_str = ", ".join(fields)
        prompt = f"Extract the following information from this image: {fields_str}. Return the results in JSON format with these exact field names."
        content = query_ollama(prompt, img_base64, model)
        
        # Extract structured data
        structured_data = {'filename': filename}
        structured_data.update(extract_structured_data(content, fields))
            
        return {'filename': filename, 'extraction': content}, content, structured_data

def process_pdf(file_bytes, filename, fields=None, process_pages_separately=True, model=None):
    """Process a PDF file using PyMuPDF"""
    try:
        # Open PDF document from memory
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(pdf_document)
        
        if process_pages_separately:
            # Process each page as a separate image
            for page_num in range(page_count):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_filename = f"{filename} (Page {page_num+1})"
                
                result, content, structured_data = process_image(img, page_filename, fields, model)
                yield page_num, page_count, img, page_filename, content, structured_data
        else:
            # Process only the first page
            page = pdf_document[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            result, content, structured_data = process_image(img, filename, fields, model)
            yield 0, page_count, img, filename, content, structured_data
            
    except Exception as e:
        yield None, None, None, filename, f"Error processing PDF: {str(e)}", None

def create_download_buttons(results, structured_results, extraction_mode):
    """Create and display download buttons for results"""
    st.header("Download Results")
    
    # Create general CSV data
    csv_data = io.StringIO()
    csv_writer = csv.writer(csv_data)
    
    if extraction_mode == "General description" or not structured_results:
        # Original CSV format with descriptions
        csv_writer.writerow(['Filename', 'Description'])
        for result in results:
            csv_writer.writerow([result['filename'], result.get('description', result.get('extraction', ''))])
        
        # Generate a filename with current date/time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"image_analysis_{timestamp}.csv"
        
        st.success("All files have been processed successfully!")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data.getvalue(),
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True
        )
    
    # Create structured CSV if available
    if extraction_mode == "Custom field extraction" and structured_results:
        # Get all possible fields from the results
        all_fields = set(['filename'])
        for result in structured_results:
            all_fields.update(result.keys())
        
        # Create structured CSV with sorted fields
        field_list = sorted(list(all_fields))
        structured_csv = io.StringIO()
        structured_writer = csv.writer(structured_csv)
        structured_writer.writerow(field_list)
        
        for result in structured_results:
            row = [result.get(field, '') for field in field_list]
            structured_writer.writerow(row)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        structured_filename = f"structured_data_{timestamp}.csv"
        
        st.success("Structured data extracted successfully!")
        st.download_button(
            label="📥 Download Structured Data as CSV",
            data=structured_csv.getvalue(),
            file_name=structured_filename,
            mime="text/csv",
            use_container_width=True
        )

# --- MAIN APP UI ---

# Display the app title
st.title("Curiosity AI Scans")

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = []
if 'structured_results' not in st.session_state:
    st.session_state.structured_results = []

# Create a sidebar for the file upload functionality
with st.sidebar:
    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "Choose images or PDFs", 
        accept_multiple_files=True, 
        type=['png', 'jpg', 'jpeg', 'pdf']
    )
    
    # Model selection
    st.header("Model Settings")
    selected_model = st.selectbox(
        "Choose vision model:",
        ["gemma3:12b", "llama3.2-vision", "granite3.2-vision", "MHKetbi/Unsloth_gemma3-12b-it:latest"],
        help="Select which AI model to use for image analysis"
    )
    
    extraction_mode = "General description"
    pdf_process_mode = "Process each page separately"
    fields = None
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        # Add option for structured data extraction
        st.header("Data Extraction Options")
        extraction_mode = st.radio(
            "Choose extraction mode:",
            ["General description", "Custom field extraction"]
        )
        
        # If custom extraction is selected, show field input
        if extraction_mode == "Custom field extraction":
            custom_fields = st.text_area(
                "Enter fields to extract (comma separated):", 
                value="Invoice number, Date, Company name, Total amount"
            )
            fields = [field.strip() for field in custom_fields.split(",")]
            
            # Option to process PDF pages separately or as a whole
            if any(file.name.lower().endswith('.pdf') for file in uploaded_files):
                pdf_process_mode = st.radio(
                    "How to process PDF files:",
                    ["Process each page separately", "Process entire PDF as one document"]
                )
        
        # Process button in sidebar
        process_button = st.button("Process Files")
    else:
        st.info("Please upload images or PDF files to analyze")
        process_button = False

# Main app logic
if uploaded_files and process_button:
    st.header("Processing Results")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear previous results when starting a new batch
    st.session_state.results = []
    st.session_state.structured_results = []
    
    # Count total items to process (including PDF pages)
    total_items = 0
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer after reading
        
        if uploaded_file.name.lower().endswith('.pdf') and PDF_SUPPORT:
            if pdf_process_mode == "Process each page separately":
                try:
                    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
                    total_items += len(pdf_document)
                except Exception as e:
                    st.error(f"Error checking PDF {uploaded_file.name}: {e}")
                    total_items += 1
            else:
                total_items += 1
        else:
            total_items += 1
    
    processed_count = 0
    
    # Process each file
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        
        # Handle PDF files
        if uploaded_file.name.lower().endswith('.pdf'):
            if not PDF_SUPPORT:
                st.error(f"Cannot process PDF file {uploaded_file.name}. Please install PyMuPDF library.")
                processed_count += 1
                progress_bar.progress(processed_count / total_items)
                continue
                
            try:
                process_separately = pdf_process_mode == "Process each page separately"
                
                for page_info in process_pdf(file_bytes, uploaded_file.name, fields, process_separately, selected_model):
                    page_num, page_count, image, page_filename, content, structured_data = page_info
                    
                    if page_num is None:  # Error case
                        st.error(content)
                        continue
                    
                    status_text.text(f"Processing {page_filename} ({page_num+1}/{page_count})")
                    
                    # Add to session state
                    result = {'filename': page_filename, 'description': content}
                    st.session_state.results.append(result)
                    
                    if structured_data and len(structured_data) > 1:
                        st.session_state.structured_results.append(structured_data)
                    
                    # Display the processed image and its results
                    st.subheader(page_filename)
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image, width=250)
                        if page_count > 1 and not process_separately:
                            st.info(f"PDF has {page_count} pages. Showing first page only.")
                    with col2:
                        st.write(content)
                        if structured_data and len(structured_data) > 1:
                            st.success("Successfully extracted structured data")
                            st.json(structured_data)
                    
                    st.divider()
                    
                    processed_count += 1
                    progress_bar.progress(min(processed_count / total_items, 1.0))
                    
            except Exception as e:
                st.error(f"Error processing PDF {uploaded_file.name}: {e}")
                processed_count += 1
                progress_bar.progress(processed_count / total_items)
        
        else:
            # Process regular image file
            status_text.text(f"Processing image {uploaded_file.name}")
            
            try:
                image = Image.open(uploaded_file)
                
                result, content, structured_data = process_image(image, uploaded_file.name, fields, selected_model)
                st.session_state.results.append(result)
                
                if structured_data and len(structured_data) > 1:
                    st.session_state.structured_results.append(structured_data)
                
                # Display the processed image and its results
                st.subheader(f"Image: {uploaded_file.name}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, width=250)
                with col2:
                    st.write(content)
                    if structured_data and len(structured_data) > 1:
                        st.success("Successfully extracted structured data")
                        st.json(structured_data)
                
                st.divider()
                
            except Exception as e:
                st.error(f"Error processing image {uploaded_file.name}: {e}")
            
            processed_count += 1
            progress_bar.progress(processed_count / total_items)
    
    status_text.text("Processing complete!")
    
    # Create download buttons
    if st.session_state.results:
        create_download_buttons(
            st.session_state.results, 
            st.session_state.structured_results, 
            extraction_mode
        )

# Display instructions when no files are processed yet
if not uploaded_files:
    st.info("👈 Upload files using the sidebar to get started")
    st.write("""
    ## How to use this app:
    1. Upload one or more images or PDF files using the sidebar on the left
    2. Select which vision model to use for analysis
    3. Choose between general description or custom field extraction
    4. If using custom extraction, specify the fields you want to extract
    5. For PDFs, choose whether to process each page separately or the entire document
    6. Click the 'Process Files' button to analyze them
    7. View the results for each image or PDF page
    8. Download results as a CSV file
    
    This app uses either the Gemma 3 12B vision model or Llama 3.2 Vision model to analyze images and PDFs.
    """)

# Add a footer with attribution
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; opacity: 0.7;">
        Made with ❤️ by Adrian with Claude - <a href="https://ad1x.com" target="_blank">ad1x.com</a>
    </div>
    """, 
    unsafe_allow_html=True
)