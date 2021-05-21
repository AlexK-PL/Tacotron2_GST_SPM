import torch
import numpy as np
from scipy.io.wavfile import write

import sys
sys.path.append('waveglow/')

from hyper_parameters import tacotron_params
from training import load_model
from text import text_to_sequence

predicted_melspec_folder = 'Predicted_melspec/'
# audio_path = '/homedtic/apeiro/GST_Tacotron2_prosody_dense_synthesis/Synth_wavs/synth_wav_
# 40500steps_second_02_fourth_05.wav'
audio_path = 'Synth_wavs/short_synth_10_62000steps_softmax_8tokens_1head_'

extension = '_014.wav'

hparams = tacotron_params
MAX_WAV_VALUE = 32768.0

# load trained tacotron 2 model:
checkpoint_path = "outputs/checkpoint_62000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.eval()

# load pre trained waveglow model for mel2audio:
waveglow_path = 'waveglow/waveglow_old.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda()

test_text = "There were others less successful." 

gst_head_scores = np.array([0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11])

for j in range(8):
    
    gst_head_scores[j] = 0.14
    gst_scores = torch.from_numpy(gst_head_scores)
    gst_scores = torch.autograd.Variable(gst_scores).cuda().float()
    
    sequence = np.array(text_to_sequence(test_text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    
    # text2mel:
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, gst_scores)
    
    # save the predicted outputs from tacotron2:
    mel_outputs_path = predicted_melspec_folder + "output.pt"
    mel_outputs_postnet_path = predicted_melspec_folder + "output_postnet.pt"
    alignments_path = predicted_melspec_folder + "alignment.pt"
    torch.save(mel_outputs, mel_outputs_path)
    torch.save(mel_outputs_postnet, mel_outputs_postnet_path)
    torch.save(alignments, alignments_path)
    
    print("text2mel prediction successfully performed...")
    
    save_path = audio_path + str(j + 1) + extension
    
    with torch.no_grad():
        audio = MAX_WAV_VALUE*waveglow.infer(mel_outputs_postnet, sigma=0.666)[0]
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    write(save_path, 22050, audio)
    
    print("mel2audio synthesis successfully performed.")

