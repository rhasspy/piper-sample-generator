# Piper Sample Generator

Generate spoken audio samples using [Piper][piper] for training a wake word system like [openWakeWord][] or [microWakeWord][].

Supports normal [Piper voices][piper voices] or a special [generator][] that can mix speaker embeddings (English only).

## Install

Create a virtual environment and install the requirements:

``` sh
git clone https://github.com/rhasspy/piper-sample-generator.git
cd piper-sample-generator/

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

## Piper Voices

Download one or more [Piper voices][piper voices] (both the `.onnx` and `.onnx.json` files for each voice). [Audio samples][piper samples] are available.

As an example, we'll download the U.S. English "lessac" voice in medium quality:

``` sh
mkdir -p voices
wget -O voices/en_US-lessac-medium.onnx 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true'
wget -O voices/en_US-lessac-medium.onnx.json 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true'
```

Generate a small set of samples with the CLI:

``` sh
python3 generate_samples.py 'okay piper.' --model voices/en_US-lessac-medium.onnx --max-samples 10 --output-dir okay_piper/
```

Check the `okay_piper/` directory for 10 WAV files (named `0.wav` to `9.wav`).

You can add multiple `--model <voice>` arguments to cycle between different voices when generating samples.

See `--help` for more options, including `--length-scales` (speaking speeds).

## Generator

Download the LibriTTS-R generator (exported from [checkpoint][]):

``` sh
wget -O models/en-us-libritts-high.pt 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'
```

Generate a small set of samples with the CLI:

``` sh
python3 generate_samples.py 'okay piper.' --model models/en-us-libritts-high.pt --max-samples 10 --output-dir okay_piper/
```

Check the `okay_piper/` directory for 10 WAV files (named `0.wav` to `9.wav`).

Generation can be much faster and more efficient if you have a GPU available and PyTorch is configured to use it. In this case, increase the batch size:

``` sh
python3 generate_samples.py 'okay piper.' --model models/en-us-libritts-high.pt --max-samples 100 --batch-size 10 --output-dir okay_piper/
```

On an NVidia 2080 Ti with 11GB, a batch size of 100 was possible (generating approximately 100 samples per second).

Setting `--max-speakers` to a value less than 904 (the number of speakers LibriTTS) is recommended. Because very few samples of later speakers were in the original dataset, using them can cause audio artifacts.

See `--help` for more options, including the `--length-scales` (speaking speeds) and `--slerp-weights` (speaker blending) which are cycled per batch.

### Augmentation

Once you have samples generated, you can augment them using [audiomentation](https://iver56.github.io/audiomentations/):

``` sh
python3 augment.py --sample-rate 22050 okay_piper/ okay_piper_augmented/
```

This will do several things to each sample:

1. Randomly decrease the volume
    * The original samples are normalized, so different volume levels are needed
2. Randomly apply an [impulse response][] using the files in `impulses/`
    * Change the acoustics of the sample to sound like the speaker was in a room with echo or using a poor quality microphone
3. Resample to 16Khz for training (e.g., [openWakeWord][])


<!-- Links -->
[piper]: https://github.com/OHF-Voice/piper1-gpl/
[openWakeWord]: https://github.com/dscripka/openWakeWord
[microWakeWord]: https://github.com/kahrendt/microWakeWord/
[piper voices]: https://huggingface.co/rhasspy/piper-voices
[generator]: https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt
[piper samples]: https://rhasspy.github.io/piper-samples/
[checkpoint]: https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US/libritts_r/medium
[impulse response]: https://iver56.github.io/audiomentations/waveform_transforms/apply_impulse_response/
