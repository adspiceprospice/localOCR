## Thank you to everyone for starring my repo! I'll do my best to extend the functionality regularly and fix things if people find problems. 

# Curiosity AI Scans

A simple application that uses AI vision models to analyze and describe images and PDF documents. Upload multiple files, get detailed descriptions powered by Gemma 3 or Llama 3.2, and export the results to CSV.

## What this application does

This application allows you to:

- Upload multiple images (JPG, PNG) and PDF documents
- Process these files using either Gemma 3 12B or Llama 3.2 11B Vision models through Ollama
- Choose which vision model to use for different analysis needs
- Get detailed AI-generated descriptions of the content in each image or PDF page
- Extract specific fields from documents (e.g., invoice numbers, dates, amounts)
- View the analysis results in a clean, organized interface
- Export all results to a CSV file for further analysis or record-keeping

The app uses Streamlit for the interface, Ollama as the backend for running the vision models locally, PIL for image processing, and PyMuPDF for PDF support.

## Installation and setup

### Step 1: Install Ollama

#### Linux
```bash
# setup ollama on linux 
curl -fsSL https://ollama.com/install.sh | sh
```

#### macOS
```bash
# Install Ollama using Homebrew
brew install ollama
# Or download from https://ollama.com/download
```

#### Windows
1. Download the Ollama installer from [https://ollama.com/download](https://ollama.com/download)
2. Run the installer and follow the on-screen instructions

### Step 2: Pull the required vision models

After installing Ollama, pull the required models:

```bash
# Pull the Gemma 3 Vision model 
ollama pull gemma3:12b

# Pull the Llama 3.2 Vision model
ollama pull llama3.2-vision

# Pull the Granite 3.2 Vision model (smaller option for limited RAM)
ollama pull granite3.2-vision
```

You can pull just one or both models depending on your needs. The app will work with whichever model you have installed.

### Step 3: Set up Python environment

Ensure you have Python 3.9 or later installed (Python 3.9-3.12 recommended for best compatibility).

#### Create and activate a virtual environment

##### Linux/macOS
```bash
# Clone or download this repository
git clone <repository-url>
cd <repository-folder>

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

##### Windows
```powershell
# Clone or download this repository
git clone <repository-url>
cd <repository-folder>

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

### Step 4: Install dependencies

With your virtual environment activated, install the required packages:
```bash
# Install dependencies using requirements.txt
pip install -r requirements.txt

# If PyMuPDF is not included in requirements.txt, install it manually:
pip install pymupdf
```

## Running the Application

1. **Start Ollama** (if not already running):
   ```bash
   # On Linux/macOS
   ollama serve
   
   # On Windows
   # Ollama typically runs as a service after installation
   ```

2. **Launch the Streamlit app**:
   ```bash
   # Make sure your virtual environment is activated
   # The command prompt should show (venv) at the beginning
   
   # Start the app
   streamlit run app.py
   ```

3. The application will open in your default web browser, typically at `http://localhost:8501`

## Features

- Upload and analyze multiple images and PDF documents using vision models
- Choose between Gemma 3 12B and Llama 3.2 Vision models for analysis
- Select between general description and custom field extraction
- Process PDF files page by page or as complete documents
- Extract specific fields like invoice numbers, dates, and amounts
- View detailed descriptions of each uploaded file
- Download analysis results as a CSV file for further processing

## Model Comparison

- **Gemma 3 12B**: Google's vision-language model that provides detailed descriptions and good structure recognition
- **Llama 3.2 Vision**: Meta's vision model that excels at recognizing visual content and following detailed instructions
- **Granite 3.2 Vision**: A smaller 2B parameter model that works well on systems with limited RAM while still providing reasonable analysis
- **Unsloth_gemma3-12b-it**: A custom model that uses the Gemma 3 12B model optimized by Unsloth 

Choose the appropriate model based on your specific needs and hardware constraints.

## Technical Details

This application leverages several key technologies:

- **Streamlit**: Provides the web interface and interactive components
- **Ollama**: Runs the vision models locally 
- **Gemma 3 12B & Llama 3.2 Vision**: Vision-language models that power the image analysis
- **Pillow (PIL)**: Handles image processing and format conversion
- **PyMuPDF**: Converts PDF pages to images for processing (imported as `fitz`)
- **Pandas**: Used for data manipulation and CSV export

## Troubleshooting

- **Package installation issues**: If you encounter problems installing the dependencies, try updating pip first: `pip install --upgrade pip`
- **Compatibility issues**: This application works best with Python 3.9-3.12. Python 3.13 might have compatibility issues with some packages.
- **Ollama connection errors**: Ensure Ollama is running by checking `ollama serve` or restarting the Ollama service.
- **Model availability**: If you get an error about a model not being available, make sure you've pulled it with `ollama pull <model_name>`
- **PDF processing errors**: If PDF processing isn't working, ensure PyMuPDF is installed correctly with `pip install pymupdf`.
- **PDF support optional**: The application will run without PDF support, but will display a warning. Install PyMuPDF to enable PDF processing.

## Try to keep your prompts short with Gemma for better results, larger promps with detailed instructions don't seem to result in better outputs
---

Made with ❤️ by Adrian with Claude - [ad1x.com](https://ad1x.com)
