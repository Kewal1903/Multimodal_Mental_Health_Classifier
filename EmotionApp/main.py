import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware  # To allow our frontend to talk to the backend
import uvicorn
import numpy as np
from PIL import Image
import cv2
# (Keep all your existing imports)
from audio_model import CNN_GRU_Attention, preprocess_for_model, features_to_tensor
import soundfile as sf # We need this to read the audio file bytes
import io
from fastapi.templating import Jinja2Templates
from fastapi import Request
from starlette.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles # <-- Add this line too

# --- Import our custom model classes and functions ---
# We'll import these from the files we've created
from image_model import get_resnet50_1ch, get_image_transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Define Master Label Set ---
# This is crucial for synchronizing model outputs
# Based on your fine-tuned model and your teammate's notebook
MASTER_LABELS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# --- 2. Define Weights ---
# We'll start with these and can tune them later
WEIGHTS = {
    'text': 0.5,
    'image': 0.2,
    'audio': 0.3  # We'll add this when your teammate sends the file
}

# --- 3. Setup FastAPI App ---
app = FastAPI(title="Multimodal Emotion API", version="1.0")

# Mount the 'static' folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up the 'templates' folder
templates = Jinja2Templates(directory="templates")

# Allow requests from all origins (useful for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Load Models ---
# We'll create a "model cache" to hold our models in memory
model_cache = {}


@app.on_event("startup")
def load_models():
    """
    This function runs once when the API server starts.
    It loads all our heavy models into the model_cache.
    """
    print("Loading models into memory...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cache["device"] = device

    # --- Load Text Model (RoBERTa) ---
    text_model_path = "./models/text_model"  # Path to your saved model folder
    model_cache["text_tokenizer"] = AutoTokenizer.from_pretrained(text_model_path)
    model_cache["text_model"] = AutoModelForSequenceClassification.from_pretrained(text_model_path)
    model_cache["text_model"].to(device).eval()

    # --- Load Image Model (InceptionLikeCNN) ---
    image_model_path = "./models/image_model.pth"  # Path to your teammate's .pth file
    # We must first create the "blueprint" of the model
    image_model_arch = get_resnet50_1ch(num_classes=7)
    # Then, load the saved weights into that blueprint
    image_model_arch.load_state_dict(torch.load(image_model_path, map_location=device))
    image_model_arch.to(device).eval()
    model_cache["image_model"] = image_model_arch

    # Load the specific image transforms this model needs
    model_cache["image_transforms"] = get_image_transforms()

    # --- Load Audio Model (GRU) ---
    audio_model_path = "./models/audio_model.pt"
    audio_model_arch = CNN_GRU_Attention(num_classes=7)  # 7 classes
    audio_model_arch.load_state_dict(torch.load(audio_model_path, map_location=device))
    audio_model_arch.to(device).eval()
    model_cache["audio_model"] = audio_model_arch

    # We also need the label encoder from his notebook
    model_cache["audio_label_encoder"] = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

    print("Models loaded successfully.")


# --- 5. Helper Functions for Prediction ---

def predict_text(text: str):
    """
    Runs inference on the text model.
    Returns a probability dictionary synced to MASTER_LABELS.
    """
    device = model_cache["device"]
    tokenizer = model_cache["text_tokenizer"]
    model = model_cache["text_model"]

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    # Map probabilities to our MASTER_LABELS
    # model.config.id2label gives us the model's original label order
    synced_probs = {label: 0.0 for label in MASTER_LABELS}
    for i, prob in enumerate(probabilities):
        model_label = model.config.id2label[i]

        # We must map the model's labels to our master labels
        # (e.g., your model's "joy" becomes our "Happy")
        our_label = model_label  # This assumes your labels already match. Update if needed.
        if our_label in synced_probs:
            synced_probs[our_label] = float(prob)

    return synced_probs


def predict_image(image_bytes: bytes):
    """
    Preprocesses image bytes, runs inference on the image model,
    and returns a probability dictionary synced to MASTER_LABELS.
    """
    device = model_cache["device"]
    model = model_cache["image_model"]
    transforms = model_cache["image_transforms"]

    # Convert bytes to a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode image as grayscale
    img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Apply the Albumentations transforms
    augmented = transforms(image=img_np)
    image_tensor = augmented['image'].to(device).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)

    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]

    # Map probabilities to our MASTER_LABELS
    # IMPORTANT: We MUST know the order of your teammate's model output.
    # We are assuming it's the same as MASTER_LABELS. If not, we must change this.
    synced_probs = {label: 0.0 for label in MASTER_LABELS}
    for i, prob in enumerate(probabilities):
        label = MASTER_LABELS[i]  # Assumes model output order matches
        synced_probs[label] = float(prob)

    return synced_probs

def predict_audio(audio_bytes: bytes):
    """
    Preprocesses audio bytes, runs inference on the audio model,
    and returns a probability dictionary synced to MASTER_LABELS.
    """
    device = model_cache["device"]
    model = model_cache["audio_model"]
    le_classes = model_cache["audio_label_encoder"] # The model's original class order

    # Read audio bytes using soundfile
    # We use io.BytesIO to treat the bytes as a file
    sf_data, sr = sf.read(io.BytesIO(audio_bytes))

    # If stereo, convert to mono by averaging channels
    if sf_data.ndim > 1:
        sf_data = sf_data.mean(axis=1)

    # librosa expects a file path, so we save the bytes to a temporary file
    temp_file = "temp_audio.wav"
    sf.write(temp_file, sf_data, sr)

    # Use the preprocessing pipeline from audio_model.py
    feats = preprocess_for_model(temp_file) # (188, 94)
    tensor = features_to_tensor(feats).to(device) # (1, 1, 188, 94)

    # Predict
    with torch.no_grad():
        outputs = model(tensor)

    # Apply softmax
    probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]

    # Map probabilities to our MASTER_LABELS
    synced_probs = {label: 0.0 for label in MASTER_LABELS}
    for i, prob in enumerate(probabilities):
        model_label = le_classes[i] # e.g., 'angry'
        if model_label in synced_probs: # Check if it's in our master list
            synced_probs[model_label] = float(prob)

    return synced_probs

# --- 6. Main API Endpoint ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    text_input: str = Form(...),
    image_input: UploadFile = File(...),
    audio_input: UploadFile = File(...)
):
    """
    Main endpoint to predict emotion from text, image, and (later) audio.
    """

    # --- Run Predictions ---
    # We run them in parallel
    text_probs_dict = predict_text(text_input)

    # Read image bytes
    image_bytes = await image_input.read()
    image_probs_dict = predict_image(image_bytes)

    # Read audio bytes
    audio_bytes = await audio_input.read()
    audio_probs_dict = predict_audio(audio_bytes)

    # --- Ensemble (Combine) Results ---
    final_probs = np.zeros(len(MASTER_LABELS))

    # Convert dictionaries to arrays in the correct order
    text_probs_array = np.array([text_probs_dict[label] for label in MASTER_LABELS])
    image_probs_array = np.array([image_probs_dict[label] for label in MASTER_LABELS])
    audio_probs_array = np.array([audio_probs_dict[label] for label in MASTER_LABELS])

    # Apply weights
    final_probs += text_probs_array * WEIGHTS['text']
    final_probs += image_probs_array * WEIGHTS['image']
    final_probs += audio_probs_array * WEIGHTS['audio']


    # Normalize weights if they don't sum to 1
    total_weight = WEIGHTS['text'] + WEIGHTS['image'] + WEIGHTS['audio']
    if total_weight > 0:
        final_probs /= total_weight

    final_prediction_index = np.argmax(final_probs)
    final_emotion = MASTER_LABELS[final_prediction_index]

    # --- Format for Chart.js ---
    # Convert back to dictionaries for a clean JSON response
    final_probs_dict = {label: float(prob) for label, prob in zip(MASTER_LABELS, final_probs)}

    return {
        "final_prediction": final_emotion,
        "final_probabilities": final_probs_dict,
        "text_probabilities": text_probs_dict,
        "image_probabilities": image_probs_dict,
        "audio_probabilities": audio_probs_dict  # Placeholder for now
    }


# --- 7. Run the App (for local testing) ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)