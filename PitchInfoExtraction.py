import torch
import numpy as np
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from praatio import pitch_and_intensity as PI
import matplotlib.pylab as plt
from STFT import STFT

filter_length = 1024
hop_length = 256
win_length = 1024
spec_frame_bins = 1024 - hop_length*2
n_mel_channels = 80
sampling_rate = 22050
mel_fmin = 0.0
mel_fmax = 8000.0
max_wav_value = 32768.0
clip_val = 1e-5
C = 1

minPitch_female = 65
maxPitch_female = 540

sampling_step = 0.01161
spec_total_band = sampling_rate / 2

output_path_pitch = "/PitchContour.PitchTier"
output_path_int = "/Intensity.IntensityTier"
praatEXE = "Praat.exe"  # your Praat .exe file location
# wav paths location:
training_files = "/filelists/ljs_audio_text_train_filelist.txt"
validation_files = "/filelists/ljs_audio_text_val_filelist.txt"
# Torch tensors will be stored in:
train_prosody_features = "/training_prosody_features/"
val_prosody_features = "/validation_prosody_features/"

with open(validation_files) as f:
    validation_audiopaths_and_text = [line.strip().split("|") for line in f]

mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
mel_basis = torch.from_numpy(mel_basis).float()
mel_basis_np = mel_basis.cpu().numpy()
mel_basis_np = np.flipud(mel_basis_np)

# --------------------------------- FIND HUMAN VOCAL PITCH RANGE MEL BINS ----------------------------------- #
spec_hertz_values = np.linspace(0, spec_total_band, spec_frame_bins)
fmin_margin = np.absolute(spec_hertz_values - minPitch_female)
fmax_margin = np.absolute(spec_hertz_values - maxPitch_female)
min_idx = np.argmin(fmin_margin)
max_idx = np.argmin(fmax_margin)

mel_min_idx = np.absolute(n_mel_channels - (np.argmax(mel_basis_np[:,min_idx])))
mel_max_idx = np.absolute(n_mel_channels - (np.argmax(mel_basis_np[:,max_idx])))

# ---------------------------------------------------------------------------------------------------------- #

counter = 0

for paths in validation_audiopaths_and_text:

    audiopath, sentence = paths[0], paths[1]

    WavName = audiopath.split("/")
    WavName, extension = WavName[-1].split(".")

    PiList = PI.extractPitch(audiopath, output_path_pitch, praatEXE, minPitch_female, maxPitch_female,
                             sampleStep=sampling_step)
    # IntList = PI.extractIntensity(audiopath, output_path_int, praatEXE, minPitch_female, sampleStep=sampling_step)

    stft_fn = STFT(filter_length, hop_length, win_length)

    # load audio wav file from the given path
    sr, data = read(audiopath)
    assert sampling_rate == sr, "Sample rate does not match with the configuration"

    audio_torch = torch.FloatTensor(data.astype(np.float32))
    audio_torch_norm = audio_torch / max_wav_value
    audio_torch_norm = audio_torch_norm.unsqueeze(0)
    audio_torch_norm = torch.autograd.Variable(audio_torch_norm, requires_grad=False)

    assert (torch.min(audio_torch_norm.data) >= -1)
    assert (torch.max(audio_torch_norm.data) <= 1)

    magnitudes, phases = stft_fn.transform(audio_torch_norm)
    magnitudes = magnitudes.data

    mask_spec = torch.zeros((magnitudes.shape[1], magnitudes.shape[2]))

    vect = np.linspace(0, spec_total_band, int(magnitudes.shape[1]) + 1)

    for ptch in PiList:
        spec_time_frame = int(ptch[0] / sampling_step)
        spec_freq_value = ptch[1]
        dist_margins = vect - spec_freq_value
        dist_margins = np.absolute(dist_margins)
        indx = np.argmin(dist_margins)
        if indx == 0:
            adapted_spec_freq_value = 0
        elif indx == (len(vect) - 1):
            adapted_spec_freq_value = (len(vect) - 1)
        else:
            if spec_freq_value <= vect[indx]:
                adapted_spec_freq_value = indx - 1
            else:
                adapted_spec_freq_value = indx
        # mask_spec[adapted_spec_freq_value][spec_time_frame] = 1
        mask_spec[adapted_spec_freq_value, spec_time_frame] = int(1)

    mel_output = torch.matmul(mel_basis, magnitudes)
    mel_mask_output = torch.matmul(mel_basis, mask_spec)

    mel_mask_output = torch.clamp(mel_mask_output, min=clip_val * C)
    mel_output = torch.log(torch.clamp(mel_output, min=clip_val) * C)
    mel_spec = torch.squeeze(mel_output, 0)

    mel_mask_output_np = mel_mask_output.data.cpu().numpy()
    mel_mask_output_np = np.flipud(mel_mask_output_np)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(mel_mask_output_np)
    plt.colorbar()
    plt.show()

    mel_spec_np = mel_spec.data.cpu().numpy()

    dense_pc_features = np.zeros((4, mel_mask_output_np.shape[1]))
    for col in range(mel_mask_output_np.shape[1]):
        col_val = np.amax(mel_mask_output_np[:, col])
        ind = np.argmax(mel_mask_output_np[:, col])
        if col_val > clip_val:
            bin_values_db = np.array([mel_spec_np[ind - 1, col], mel_spec_np[ind, col], mel_spec_np[ind + 1, col]])
            dense_pc_features[0, col] = float(ind)
            dense_pc_features[1, col] = bin_values_db[0]
            dense_pc_features[2, col] = bin_values_db[1]
            dense_pc_features[3, col] = bin_values_db[2]
        else:
            dense_pc_features[0, col] = float(-1)
            dense_pc_features[1, col] = float(-1)
            dense_pc_features[2, col] = float(-1)
            dense_pc_features[3, col] = float(-1)

        print(dense_pc_features[:, col])

    # Count iterations:
    counter += 1
    print(counter)
    # Save pitch contour tensor
    prosody_features_path = val_prosody_features + WavName + "_dense_pitch_info.pt"
    torch.save(dense_pc_features, prosody_features_path)
