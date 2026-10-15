"""Sample sources: IQ recordings and (optionally) live SDRs via SoapySDR."""
from __future__ import annotations

import numpy as np

_DTYPES = {"cf32": np.complex64, "ci16": np.int16, "cs16": np.int16, "ci8": np.int8}


def read_iq(path: str, fmt: str = "cf32", offset: int = 0, count: int | None = None) -> np.ndarray:
    """Read complex samples from a raw interleaved IQ file (memory-mapped).
    Formats: cf32 (GNU Radio / srsRAN / uhd_rx_cfile), ci16/cs16, ci8."""
    dt = _DTYPES[fmt]
    if fmt == "cf32":
        mm = np.memmap(path, dtype=np.complex64, mode="r")
        end = len(mm) if count is None else min(len(mm), offset + count)
        return np.array(mm[offset:end])
    mm = np.memmap(path, dtype=dt, mode="r")
    n_total = len(mm) // 2
    end = n_total if count is None else min(n_total, offset + count)
    raw = np.array(mm[2 * offset: 2 * end], dtype=np.float32)
    scale = 32768.0 if dt == np.int16 else 128.0
    return ((raw[0::2] + 1j * raw[1::2]) / scale).astype(np.complex64)


def n_samples(path: str, fmt: str = "cf32") -> int:
    import os
    size = os.path.getsize(path)
    return size // (8 if fmt == "cf32" else 2 * np.dtype(_DTYPES[fmt]).itemsize)


def record_soapy(args: str, freq_hz: float, rate: float, seconds: float, gain_db: float,
                 out_paths: list[str], channels=(0,), antenna: str | None = None) -> None:
    """Record ``seconds`` of IQ from one or more phase-coherent channels
    (e.g. USRP B210 ch0 = reference, ch1 = surveillance) to cf32 files.
    Requires SoapySDR + the device module. Use a GPSDO for multistatic work."""
    import SoapySDR  # type: ignore
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX  # type: ignore

    dev = SoapySDR.Device(args)
    for ch in channels:
        dev.setSampleRate(SOAPY_SDR_RX, ch, rate)
        dev.setFrequency(SOAPY_SDR_RX, ch, freq_hz)
        dev.setGain(SOAPY_SDR_RX, ch, gain_db)
        if antenna:
            dev.setAntenna(SOAPY_SDR_RX, ch, antenna)
    st = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, list(channels))
    dev.activateStream(st)
    total = int(seconds * rate)
    blk = 1 << 16
    bufs = [np.empty(blk, np.complex64) for _ in channels]
    files = [open(p, "wb") for p in out_paths]
    got = 0
    try:
        while got < total:
            sr = dev.readStream(st, bufs, blk, timeoutUs=1_000_000)
            if sr.ret <= 0:
                raise RuntimeError(f"readStream error {sr.ret}")
            for f, b in zip(files, bufs):
                b[: sr.ret].tofile(f)
            got += sr.ret
    finally:
        dev.deactivateStream(st)
        dev.closeStream(st)
        for f in files:
            f.close()
