import wfdb
import numpy as np
from i_signal_reader import ISignalReader


class SignalReader(ISignalReader):
    def __init__(self, record_path: str):

        self._record_path = record_path
        self._record = None
        self._fs = None
        self._code = None
        self._load_record()

    def _load_record(self):
        try:
            self._record = wfdb.rdrecord(self._record_path)
            self._fs = self._record.fs
            self._code = self._record_path.split('/')[-1]
        except Exception as e:
            raise ValueError(f"Nie można wczytać zapisu EKG z {self._record_path}: {e}")

    def read_signal(self) -> np.ndarray:
        if self._record is None:
            raise ValueError("Rekord EKG nie został poprawnie wczytany.")
        return self._record.p_signal

    def read_fs(self) -> float:
        if self._fs is None:
            raise ValueError("Częstotliwość próbkowania nie została poprawnie wczytana.")
        return self._fs

    def get_code(self) -> str:
        if self._code is None:
            raise ValueError("Kod rekordu nie został poprawnie ustalony.")
        return self._code
    