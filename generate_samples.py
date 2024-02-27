#!/usr/bin/env python3
import argparse
import gc
import itertools as it
import json
import logging
import os
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import webrtcvad
from piper_phonemize import phonemize_espeak

from piper_train.vits import commons

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# Main generation function
def generate_samples(
    text: Union[List[str], str],
    output_dir: Union[str, Path],
    max_samples: Optional[int] = None,
    file_names: Optional[List[str]] = None,
    model: Union[str, Path] = _DIR / "models" / "en_US-libritts_r-medium.pt",
    batch_size: int = 1,
    slerp_weights: Tuple[float, ...] = (0.5,),
    length_scales: Tuple[float, ...] = (0.75, 1, 1.25),
    noise_scales: Tuple[float, ...] = (0.667,),
    noise_scale_ws: Tuple[float, ...] = (0.8,),
    max_speakers: Optional[float] = None,
    verbose: bool = False,
    auto_reduce_batch_size: bool = False,
    min_phoneme_count: Optional[int] = None,
    **kwargs,
) -> None:
    """
    Generate synthetic speech clips, saving the clips to the specified output directory.

    Args:
        text (List[str]): The text to convert into speech. Can be either a
                          a list of strings, or a path to a file with text on each line.
        output_dir (str): The location to save the generated clips.
        max_samples (int): The maximum number of samples to generate.
        file_names (List[str]): The names to use when saving the files. Must be the same length
                                as the `text` argument, if a list.
        model (str): The path to the STT model to use for generation.
        batch_size (int): The batch size to use when generated the clips
        slerp_weights (List[float]): The weights to use when mixing speakers via SLERP.
        length_scales (List[float]): Controls the average duration/speed of the generated speech.
        noise_scales (List[float]): A parameter for overall variability of the generated speech.
        noise_scale_ws (List[float]): A parameter for the stochastic duration of words/phonemes.
        max_speakers (int): The maximum speaker number to use, if the model is multi-speaker.
        verbose (bool): Enable or disable more detailed logging messages (default: False).
        auto_reduce_batch_size (bool): Automatically and temporarily reduce the batch size
                                       if CUDA OOM errors are detected, and try to resume generation.
        min_phoneme_count (int): If set, ensure this number of phonemes is always sent to the model.
                                 Clip audio to extract original phrase.

    Returns:
        None
    """

    if max_samples is None:
        max_samples = len(text)

    _LOGGER.debug("Loading %s", model)
    model_path = Path(model)

    torch_model = torch.load(model_path)
    torch_model.eval()
    _LOGGER.info("Successfully loaded the model")

    if torch.cuda.is_available():
        torch_model.cuda()
        _LOGGER.debug("CUDA available, using GPU")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = f"{model_path}.json"
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    voice = config["espeak"]["voice"]
    sample_rate = config["audio"]["sample_rate"]
    num_speakers = config["num_speakers"]
    if max_speakers is not None:
        num_speakers = min(num_speakers, max_speakers)

    max_len = None

    sample_idx = 0
    is_done = False
    settings_iter = it.cycle(
        it.product(
            slerp_weights,
            length_scales,
            noise_scales,
            noise_scale_ws,
        )
    )

    # Define resampler to get to 16khz (https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best)
    sample_rate = 22050
    resample_rate = 16000
    resampler = torchaudio.transforms.Resample(
        sample_rate,
        resample_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )

    speakers_iter = it.cycle(it.product(range(num_speakers), range(num_speakers)))
    speakers_batch = list(it.islice(speakers_iter, 0, batch_size))
    if isinstance(text, str) and os.path.exists(text):
        texts = it.cycle(
            [
                i.strip()
                for i in open(text, "r", encoding="utf-8").readlines()
                if len(i.strip()) > 0
            ]
        )
    elif isinstance(text, list):
        texts = it.cycle(text)
    else:
        texts = it.cycle([text])

    if file_names:
        file_names = it.cycle(file_names)

    batch_idx = 0
    while speakers_batch:
        if is_done:
            break

        batch_size = len(speakers_batch)
        slerp_weight, length_scale, noise_scale, noise_scale_w = next(settings_iter)

        with torch.no_grad():
            speaker_1 = torch.LongTensor([s[0] for s in speakers_batch])
            speaker_2 = torch.LongTensor([s[1] for s in speakers_batch])

            phoneme_ids_by_batch = []
            clip_indexes_by_batch = []
            for i in range(batch_size):
                phoneme_ids, clip_phoneme_index = get_phonemes(
                    voice, config, next(texts), verbose, min_phoneme_count
                )
                phoneme_ids_by_batch.append(phoneme_ids)
                clip_indexes_by_batch.append(clip_phoneme_index)

            def right_pad_lists(lists):
                max_length = max(len(lst) for lst in lists)
                padded_lists = []
                for lst in lists:
                    padded_l = lst + [1] * (
                        max_length - len(lst)
                    )  # phoneme 1 (corresponding to '^' character seems to work best)
                    padded_lists.append(padded_l)
                return padded_lists

            phoneme_ids_by_batch = right_pad_lists(phoneme_ids_by_batch)

            if auto_reduce_batch_size:
                oom_error = True
                counter = 1
                while oom_error is True:
                    try:
                        audio, phoneme_samples = generate_audio(
                            torch_model,
                            speaker_1[0 : batch_size // counter],
                            speaker_2[0 : batch_size // counter],
                            phoneme_ids_by_batch[0 : batch_size // counter],
                            slerp_weight,
                            noise_scale,
                            noise_scale_w,
                            length_scale,
                            max_len,
                        )
                        oom_error = False
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        gc.collect()
                        counter += 1  # reduce batch size to avoid OOM errors
            else:
                audio, phoneme_samples = generate_audio(
                    torch_model,
                    speaker_1,
                    speaker_2,
                    phoneme_ids_by_batch,
                    slerp_weight,
                    noise_scale,
                    noise_scale_w,
                    length_scale,
                    max_len,
                )

            # Clip audio when using min_phoneme_count
            for i, clip_phoneme_index in enumerate(clip_indexes_by_batch):
                if clip_phoneme_index is not None:
                    first_sample_idx = int(
                        phoneme_samples[i].flatten()[:clip_phoneme_index-1].sum().item()
                    )
                    
                    # Fill start of audio with silence until actual sample.
                    # It will be removed in the next stage.
                    audio[i, 0, :first_sample_idx] = 0

                # Fill time after last speech with silence.
                # It will be removed in the next stage
                last_sample_idx = int(phoneme_samples[i].flatten().sum().item())
                audio[i, 0, last_sample_idx+1:] = 0

            # Resample audio
            audio = resampler(audio.cpu()).numpy()

            audio_int16 = audio_float_to_int16(audio)
            for audio_idx in range(audio_int16.shape[0]):
                # Trim any silenced audio
                audio_data = np.trim_zeros(audio_int16[audio_idx].flatten())
                
                # Use webrtcvad to trim any remaining silence from the clips
                audio_data = remove_silence(audio_int16[audio_idx].flatten())[
                    None,
                ]

                if isinstance(file_names, it.cycle):
                    wav_path = output_dir / next(file_names)
                else:
                    wav_path = output_dir / f"{sample_idx}.wav"

                wav_file: wave.Wave_write = wave.open(str(wav_path), "wb")
                with wav_file:
                    wav_file.setframerate(resample_rate)
                    wav_file.setsampwidth(2)
                    wav_file.setnchannels(1)
                    wav_file.writeframes(audio_data)

                sample_idx += 1
                if sample_idx >= max_samples:
                    is_done = True
                    break

            # print(f"Batch {batch_idx +1}/{max_samples//batch_size} complete", " "*200, end='\r')

        # Next batch
        _LOGGER.debug("Batch %s/%s complete", batch_idx + 1, max_samples // batch_size)
        speakers_batch = list(it.islice(speakers_iter, 0, batch_size))
        batch_idx += 1

    _LOGGER.info("Done")


def remove_silence(
    x: np.ndarray,
    frame_duration: float = 0.030,
    sample_rate: int = 16000,
    min_start: int = 2000,
) -> np.ndarray:
    """Uses webrtc voice activity detection to remove silence from the clips"""
    vad = webrtcvad.Vad(0)
    if x.dtype in (np.float32, np.float64):
        x = (x * 32767).astype(np.int16)
    x_new = x[0:min_start].tolist()
    step_size = int(sample_rate * frame_duration)
    for i in range(min_start, x.shape[0] - step_size, step_size):
        vad_res = vad.is_speech(x[i : i + step_size].tobytes(), sample_rate)
        if vad_res:
            x_new.extend(x[i : i + step_size].tolist())
    return np.array(x_new).astype(np.int16)


def generate_audio(
    model,
    speaker_1,
    speaker_2,
    phoneme_ids,
    slerp_weight,
    noise_scale,
    noise_scale_w,
    length_scale,
    max_len,
):
    x = torch.LongTensor(phoneme_ids)
    x_lengths = torch.LongTensor([len(i) for i in phoneme_ids])

    if torch.cuda.is_available():
        speaker_1 = speaker_1.cuda()
        speaker_2 = speaker_2.cuda()
        x = x.cuda()
        x_lengths = x_lengths.cuda()

    x, m_p_orig, logs_p_orig, x_mask = model.enc_p(x, x_lengths)
    emb0 = model.emb_g(speaker_1)
    emb1 = model.emb_g(speaker_2)
    g = slerp(emb0, emb1, slerp_weight).unsqueeze(-1)  # [b, h, 1]

    if model.use_sdp:
        logw = model.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    else:
        logw = model.dp(x, x_mask, g=g)
    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(
        commons.sequence_mask(y_lengths, y_lengths.max()), 1
    ).type_as(x_mask)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p_orig.transpose(1, 2)).transpose(
        1, 2
    )  # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p_orig.transpose(1, 2)).transpose(
        1, 2
    )  # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = model.flow(z_p, y_mask, g=g, reverse=True)
    o = model.dec((z * y_mask)[:, :, :max_len], g=g)

    audio = o
    phoneme_samples = w_ceil * 256  # hop length

    return audio, phoneme_samples


def get_phonemes(
    voice: str,
    config: Dict[str, Any],
    text: str,
    verbose: bool = False,
    min_phoneme_count: Optional[int] = None,
) -> Tuple[List[int], Optional[int]]:
    # Combine all sentences
    phonemes = [
        p
        for sentence_phonemes in phonemize_espeak(text, voice)
        for p in sentence_phonemes
    ]
    if verbose is True:
        _LOGGER.debug("Phonemes: %s", phonemes)

    id_map = config["phoneme_id_map"]

    # Beginning of utterance
    phoneme_ids = list(id_map["^"])
    phoneme_ids.extend(id_map["_"])

    # Phoneme ids for just the text
    text_phoneme_ids = []

    for phoneme in phonemes:
        p_ids = id_map.get(phoneme)
        if p_ids is not None:
            phoneme_ids.extend(p_ids)
            text_phoneme_ids.extend(p_ids)
            phoneme_ids.extend(id_map["_"])
            text_phoneme_ids.extend(id_map["_"])

    # Index where audio should be clipped at.
    # When None, all of the audio will be used.
    clip_phoneme_index: Optional[int] = None

    if min_phoneme_count is not None:
        # Repeat phrase until minimum phoneme count is met.
        # NOTE: It is critical that the ^ and $ phonemes are not repeated here.
        while (len(phoneme_ids) - 1) < min_phoneme_count:
            # We will clip audio at the beginning of the last phrase
            clip_phoneme_index = len(phoneme_ids) - 1

            phoneme_ids.extend(text_phoneme_ids)

    # End of utterance
    phoneme_ids.extend(id_map["$"])

    return phoneme_ids, clip_phoneme_index


def slerp(v1, v2, t: float, DOT_THR: float = 0.9995, zdim: int = -1):
    """SLERP for pytorch tensors interpolating `v1` to `v2` with scale of `t`.

    `DOT_THR` determines when the vectors are too close to parallel.
        If they are too close, then a regular linear interpolation is used.

    `zdim` is the feature dimension over which to compute norms and find angles.
        For example: if a sequence of 5 vectors is input with shape [5, 768]
        Then `zdim = 1` or `zdim = -1` computes SLERP along the feature dim of 768.

    Theory Reference:
    https://splines.readthedocs.io/en/latest/rotation/slerp.html
    PyTorch reference:
    https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
    Numpy reference:
    https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    """

    # take the dot product between normalized vectors
    v1_norm = v1 / torch.norm(v1, dim=zdim, keepdim=True)
    v2_norm = v2 / torch.norm(v2, dim=zdim, keepdim=True)
    dot = (v1_norm * v2_norm).sum(zdim)

    # if the vectors are too close, return a simple linear interpolation
    if (torch.abs(dot) > DOT_THR).any():
        res = (1 - t) * v1 + t * v2

    # else apply SLERP
    else:
        # compute the angle terms we need
        theta = torch.acos(dot)
        theta_t = theta * t
        sin_theta = torch.sin(theta)
        sin_theta_t = torch.sin(theta_t)

        # compute the sine scaling terms for the vectors
        s1 = torch.sin(theta - theta_t) / sin_theta
        s2 = sin_theta_t / sin_theta

        # interpolate the vectors
        res = (s1.unsqueeze(zdim) * v1) + (s2.unsqueeze(zdim) * v2)

    return res


def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """Normalize audio and convert to int16 range"""
    audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm


def main() -> None:
    """Main entry point."""

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--max-samples", required=True, type=int)
    parser.add_argument(
        "--model", default=_DIR / "models" / "en_US-libritts_r-medium.pt"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--slerp-weights", nargs="+", type=float, default=[0.5])
    parser.add_argument(
        "--length-scales", nargs="+", type=float, default=[1.0, 0.75, 1.25, 1.4]
    )
    parser.add_argument(
        "--noise-scales",
        nargs="+",
        type=float,
        default=[0.667, 0.75, 0.85, 0.9, 1.0, 1.4],
    )
    parser.add_argument("--noise-scale-ws", nargs="+", type=float, default=[0.8])
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers to use (default: all)",
    )
    parser.add_argument("--min-phoneme-count", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args().__dict__

    # Generate speech
    generate_samples(**args)


if __name__ == "__main__":
    main()
