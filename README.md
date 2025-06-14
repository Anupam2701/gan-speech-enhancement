# 🎧 GAN-Based Speech Enhancement

A deep learning-powered Streamlit app to enhance speech from noisy audio using a Generative Adversarial Network (GAN).

🌐 **Live App**: [https://gan-speech-enhancement-he9bfp8kagdkgbfznh6djy.streamlit.app](https://gan-speech-enhancement-he9bfp8kagdkgbfznh6djy.streamlit.app/)

---

## 📁 Project Structure

GAN-SPEECH-ENHANCEMENT/
├── app.py # Streamlit interface

├── model.py # Generator & Discriminator architecture

├── model_utils.py # Inference + audio enhancement logic

├── generator_model.pth # Trained generator weights

├── requirements.txt # Python dependencies

├── README.md # Project overview and usage

└── .gitignore # Git exclusions (e.g. .pyc, .wav, venv)


---

## 📌 Project Workflow

### ✅ Step 1: Define the Problem

- **Objective**: Enhance noisy audio to improve speech clarity while preserving the original content.
- **Evaluation**:  
  - PESQ (Perceptual Evaluation of Speech Quality)  
  - STOI (Short-Time Objective Intelligibility)  
  - SNR (Signal-to-Noise Ratio)

---

### 📂 Step 2: Collect and Prepare Data

- **Dataset**: [VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/1942)
- **Preprocessing**:
  - Convert audio to 16 kHz sample rate
  - Pad/truncate to 4 seconds
  - STFT to get complex spectrogram
  - Split into training/validation/test

---

### 🧠 Step 3: Design the Model Architecture

#### 🎛 Generator (TRCAE)
- Residual Complex Autoencoder
- Temporal Convolution Module
- Produces complex-valued mask for speech

#### 🧪 Discriminator
- CNN-based binary classifier
- Distinguishes clean vs enhanced spectrograms

---

### 🎯 Step 4: Define Loss Functions

- **Complex Loss** – L1 loss on real/imag spectrogram
- **Perceptual Loss** – CNN feature loss
- **Adversarial Loss** – From discriminator
- **AB-SNR Loss** – Audio-based signal-to-noise

---

### 🏋️ Step 5: Train the Model

- Train Discriminator to classify clean vs enhanced
- Train Generator with total loss:
  total_loss = complex + ab_snr + 0.1 * perceptual + 0.1 * adversarial

- Save best model based on validation loss
- Track metrics across epochs

---

### 📈 Step 6: Evaluate the Model

- Use test set from VoiceBank
- Compute:
- Average PESQ
- STOI
- SNR
- Optionally visualize spectrograms

---

### 💾 Step 7: Save the Model

- Generator saved as: `generator_model.pth`
- Inference-ready for Streamlit app

---

### 🚀 Step 8: Deploy the Model

- **Inference Pipeline**:
- Upload `.wav`
- Compute STFT → Model → ISTFT
- Save & download enhanced audio

- **Streamlit Web App**:  
Hosted at [https://gan-speech-enhancement-he9bfp8kagdkgbfznh6djy.streamlit.app](https://gan-speech-enhancement-he9bfp8kagdkgbfznh6djy.streamlit.app/)

---

## 🖥️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Anupam2701/gan-speech-enhancement.git
cd gan-speech-enhancement

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # on Windows

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py



