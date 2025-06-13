import streamlit as st
import librosa
import soundfile as sf
import numpy as np
from model_utils import load_model, enhance_audio
from pystoi import stoi
#from pesq import pesq

st.title("🎧 GAN-based Speech Enhancement")

# Upload audio
uploaded_file = st.file_uploader("Upload a Noisy Audio File (.wav)", type=["wav"])
sample_rate = 16000
n_fft = 512
hop_length = 256

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav', start_time=0)
    
    # Load audio
    noisy_audio, sr = librosa.load(uploaded_file, sr=sample_rate)
    st.write(f"Sample Rate: {sr}, Duration: {len(noisy_audio) / sr:.2f} sec")

    # Load model
    generator = load_model("generator_model.pth")

    # Enhance audio
    enhanced_audio = enhance_audio(generator, noisy_audio, sample_rate, n_fft, hop_length)

    # Save enhanced audio
    sf.write("enhanced_output.wav", enhanced_audio, sample_rate)

    st.success("✅ Enhancement Done!")
    st.audio("enhanced_output.wav", format="audio/wav")

    # Metrics (Optional clean upload)
    st.subheader("Optional: Upload Ground Truth Clean Audio for Evaluation")
    clean_file = st.file_uploader("Upload Clean Audio File (.wav)", type=["wav"], key="clean_audio")

    if clean_file:
        clean_audio, _ = librosa.load(clean_file, sr=sample_rate)

        try:
            #pesq_score = pesq(sample_rate, clean_audio, enhanced_audio, 'wb')
            stoi_score = stoi(clean_audio, enhanced_audio, sample_rate, extended=False)
            noise = clean_audio - enhanced_audio
            snr_score = 10 * np.log10(np.sum(clean_audio**2) / (np.sum(noise**2) + 1e-8))

            #st.metric("PESQ", f"{pesq_score:.4f}")
            st.metric("STOI", f"{stoi_score:.4f}")
            st.metric("SNR (dB)", f"{snr_score:.2f}")

        except Exception as e:
            st.error(f"Could not compute metrics: {e}")
