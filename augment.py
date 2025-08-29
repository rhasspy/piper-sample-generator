#!/usr/bin/env python3
import argparse
import audioop
import sys
import wave
from pathlib import Path

import numpy as np
from audiomentations import Compose, ApplyImpulseResponse, Gain

_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--sample-rate", type=int)
    args = parser.parse_args()

    impulses = list((_DIR / "impulses").glob("*.wav"))

    augment = Compose(
        transforms=[
            Gain(min_gain_db=-12, max_gain_db=0),
            ApplyImpulseResponse(impulses),
        ]
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_wav in input_dir.glob("*.wav"):
        output_wav = output_dir / (input_wav.relative_to(input_dir))
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(input_wav), "rb") as input_wav_file, wave.open(
            str(output_wav), "wb"
        ) as output_wav_file:
            assert input_wav_file.getsampwidth() == 2
            assert input_wav_file.getnchannels() == 1

            input_audio = (
                np.frombuffer(
                    input_wav_file.readframes(input_wav_file.getnframes()),
                    dtype=np.int16,
                ).astype(np.float32)
                / 32767.0
            )

            output_audio = augment(
                input_audio, sample_rate=input_wav_file.getframerate()
            )
            output_wav_file.setframerate(
                args.sample_rate or input_wav_file.getframerate()
            )
            output_wav_file.setsampwidth(2)
            output_wav_file.setnchannels(1)

            output_audio_16 = audio_float_to_int16(output_audio)
            if args.sample_rate != input_wav_file.getframerate():
                output_audio_16, _state = audioop.ratecv(
                    output_audio_16,
                    2,
                    1,
                    input_wav_file.getframerate(),
                    args.sample_rate,
                    None,
                )

            output_wav_file.writeframes(output_audio_16)

        print(output_wav)


def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    # Don't normalize
    audio_norm = audio * max_wav_value
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm


if __name__ == "__main__":
    main()
