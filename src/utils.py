"""Utility functions for the SDS API Gateway."""

import io
import logging

import av
import numpy as np

log = logging.getLogger("app.utils")


def np_wav_to_compressed_buffer(sample_rate: int, wav: np.ndarray):
    """Compress raw audio and store inside a file buffer."""
    # Some assumptions about audio passed from gradio:
    assert wav.dtype == np.int16
    # Mono without channel dim.
    if wav.ndim == 1:
        wav = np.tile(wav[:, None], (1, 2))
    # mono with channel dim.
    elif wav.ndim == 2 and wav.shape[1] == 1:
        wav = np.tile(wav, (1, 2))

    # Groq downsamples to 16kHz mono, so we compress to that to save bandwidth.
    # Balance between file size (upload speed) and decode latency.
    out_rate = 16000
    frame_size = 120  # 120ms max supported by Opus.
    bitrate = 16000  # 16kbps, good enough quality for speech.

    buf = io.BytesIO()
    frame_size = sample_rate // 1000 * frame_size
    container = av.open(buf, mode="w", format="ogg")
    resampler = av.AudioResampler(
        format="s16", layout="mono", rate=out_rate, frame_size=frame_size
    )

    stream = container.add_stream(
        "libopus", rate=out_rate, bit_rate=bitrate, layout="mono"
    )

    for i in range(0, len(wav), frame_size):
        chunk = np.ascontiguousarray(wav[i: i + frame_size].T)
        frame = av.AudioFrame.from_ndarray(
            chunk, format="s16p", layout="stereo")
        frame.rate = sample_rate
        frames = resampler.resample(frame)

        for frm in frames:
            container.mux(stream.encode(frm))

    # Flush all packets.
    container.mux(stream.encode())

    container.close()
    buf.seek(0)

    # with open("test.ogg", "wb") as f:
    #     f.write(buf.getbuffer())
    # buf.seek(0)  # Reset buffer position for reading.

    return buf


def setup_logging(log_path):
    """Setup logging."""
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s|%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s|%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(ch)
