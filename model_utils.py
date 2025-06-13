from model import ResidualComplexAutoencoderTemporal
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = ResidualComplexAutoencoderTemporal().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
def enhance_audio(model, audio_np, sample_rate=16000, n_fft=512, hop_length=256):
    audio_tensor = torch.tensor(audio_np, dtype=torch.float32).to(device)
    window = torch.hann_window(n_fft).to(device)

    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      window=window, return_complex=True)
    
    mag = torch.log1p(torch.abs(spec)).unsqueeze(0).unsqueeze(0)
    phase = torch.angle(spec).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        enhanced_complex = model(mag, phase)

    enhanced_audio = torch.istft(enhanced_complex.squeeze(0), n_fft=n_fft, hop_length=hop_length,
                                 window=window, length=audio_tensor.shape[0])
    
    return enhanced_audio.cpu().numpy()