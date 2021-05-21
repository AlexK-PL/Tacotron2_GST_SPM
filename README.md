# Tacotron2_GST_SparsePitchMatrix
This is our work on learning speaking style in speech synthesis but only using the pitch frequency sub-band as a speaker reference. We trained a modified version of the NVIDIA's Tacotron2 model but including Global Style Tokens (GST). In this work, instead of using the whole log-mel spectrogram representation to train the bank of embeddings, we extracted only the pitch sub-band of all samples and represented them as a binary matrix which we named Pitch Binary Matrix (PBM).  

## Speech pre-processin
You need to process your training samples before using PitchInfoExtraction.py, which saves pitch information as torch tensors to the path you select.

## Training
Executing MAIN.py training is executed. You just have to change the paths to yours. Note you will have to point to the place where you saved your pre-processed torch tensors.

## Inference
Executing INFERENCE_SYNTHESIS.py you will generate samples you define inside the script. Remember to also change paths where to save your WAV clips.
Note that WaveGlow model was used to synthesize our samples, you can clone it from the NVIDIA repository (https://github.com/NVIDIA/waveglow).

## Audio samples
In the folder /example_wavs you can find synthesized clips emphasizing each token over different samples for comparison. 
