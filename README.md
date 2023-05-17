# Piper Sample Generator

Generates samples using [Piper](https://github.com/rhasspy/piper/) for training a wake word system like [openWakeWord](https://github.com/dscripka/openWakeWord).


## Install

Create a virtual environment and install the requirements:

``` sh
git clone https://github.com/rhasspy/piper-sample-generator.git
cd piper-sample-generator/

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Download the LibriTTS generator:

``` sh
wget -O models/en-us-libritts-high.pt 'https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt'
```


## Run

Generate a small set of samples:

``` sh
python3 generate_samples.py 'okay, piper.' --max-samples 10 --output-dir okay_piper/
```

Check the `okay_piper/` directory for 10 WAV files (named `0.wav` to `9.wav`).

Generation can be much faster and more efficient if you have a GPU available and PyTorch is configured to use it. In this case, increase the batch size:

``` sh
python3 generate_samples.py 'okay, piper.' --max-samples 100 --batch-size 10 --output-dir okay_piper/
```

On an NVidia 2080 Ti with 11GB, a batch size of 100 was possible (generating approximately 100 samples per second).

See `--help` for more options, including adjust the `--length-scales` (speaking speeds) and `--slerp-weights` (speaker blending) which are cycled per batch.

### Augmentation

Once you have samples generating, you can augment them using [audiomentation](https://iver56.github.io/audiomentations/):

``` sh
python3 augment.py --sample-rate 16000 okay_piper/ okay_piper_augmented/
```

This will do several things to each sample:

1. Randomly decrease the volume
    * The original samples are normalized, so different volume levels are needed
2. Randomly [apply an impulse response](https://iver56.github.io/audiomentations/waveform_transforms/apply_impulse_response/) using the files in `impulses/`
    * Change the acoustics of the sample to sound like the speaker was in a room with echo or using a poor quality microphone
3. Resample to 16Khz for training (e.g., [openWakeWord](https://github.com/dscripka/openWakeWord))

