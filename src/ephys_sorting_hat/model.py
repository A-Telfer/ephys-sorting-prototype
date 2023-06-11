import numpy as np
import pyabf
import pandas as pd
import pickle

from pathlib import Path
from enum import Enum
from PyQt6 import QtCore
from PyQt6.QtCore import QObject

class SignalGroup(Enum):
    NOISE = 0
    ACTIVITY = 1

class Sweep(QObject):
    sweep_changed = QtCore.pyqtSignal()
    def __init__(self, number: int, data: np.ndarray, sample_rate: int, group:SignalGroup=SignalGroup.ACTIVITY):
        super().__init__()

        self.number: int = number
        self.data: np.ndarray = data
        self._group: SignalGroup = group
        self.sample_rate: int = sample_rate
        self.time: np.ndarray = np.arange(len(data)) / self.sample_rate
        self.was_moved_by_user: bool = False
        self.label = f"Sweep {self.number}"

    @property
    def group(self):
        return self._group
    
    @group.setter
    def group(self, value: SignalGroup):
        self._group = value
        self.sweep_changed.emit()
        self.was_moved_by_user = True

    def to_dict(self):
        return {
            'sweep_number': int(self.number),
            'data': self.data.tolist(),
            'group': int(self.group.value),
            'sample_rate': int(self.sample_rate),
            'time': self.time.tolist(), 
            'was_moved_by_user': bool(self.was_moved_by_user),
            'label': str(self.label)
        }

class Model(QObject):
    on_sweeps_changed = QtCore.pyqtSignal(list)
    on_signal_detect_complete = QtCore.pyqtSignal()
    on_save_complete = QtCore.pyqtSignal()
    on_load_complete = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._file_location = None
        self._save_location = None

        # Fourier filter parameters
        self.lowband = 5
        self.highband = 5
        # self.trigger_yraw = 30
        self.trigger_ysmoothed = 30 
        self.trigger_xmin = 0
        self.trigger_xmax = 1.0

        self.sample_frequency = 1000
        self.sample_rate = None

        # Load the sweeps into here
        self.sweeps = []
        self.active_sweep_index = None

    def reset_signals(self):
        """Reset signals to all be noise
        """
        self.active_sweep_index = -1
        for signal in self.signals:
            signal.group = SignalGroup.ACTIVITY
            signal.was_moved_by_user = False

        self.on_sweeps_changed.emit()

    @property
    def active_sweep(self) -> Sweep:
        if self.active_sweep_index == -1:
            return 0

        return self.sweeps[self.active_sweep_index]

    def get_signal_by_label(self, label):
        return next(filter(lambda x: x.label==label, self.sweeps))

    def next_sweep(self):
        self.active_sweep_index = (self.active_sweep_index + 1) % len(self.sweeps)
        self.on_sweeps_changed.emit(self.sweeps)

    def previous_sweep(self):
        self.active_sweep_index = (self.active_sweep_index - 1) % len(self.sweeps)
        self.on_sweeps_changed.emit(self.sweeps)

    def set_active_sweep_group(self, value: SignalGroup):
        self.active_sweep.group = value
        self.on_sweeps_changed.emit(self.sweeps)

    def load_abf_file(self, filepath):
        self._file_location = filepath
        self.sweeps = []
        abf = pyabf.ABF(self._file_location)
        self.sample_rate = abf.sampleRate

        for sweep_number in abf.sweepList:
            abf.setSweep(sweep_number)
            data = abf.sweepY
            sweep = Sweep(sweep_number, data, sample_rate=self.sample_rate)
            sweep.sweep_changed.connect(lambda: self.on_sweeps_changed.emit(self.sweeps))
            self.sweeps.append(sweep)

            # Emit event here to show that items are loading
            self.on_sweeps_changed.emit(self.sweeps)

        self.on_load_complete.emit()

    def low_pass_filter(self, data):
        bandlimit_index = int(self.bandlimit * data.size / self.sample_rate)
        fsig = np.fft.fft(data)
        for i in range(bandlimit_index + 1, len(fsig) - bandlimit_index):
            fsig[i] = 0
            
        data_filtered = np.fft.ifft(fsig)
        return np.real(data_filtered)
        
    def fourier_sort(self):
        self.low_pass_filter()
        for sweep in self.sweeps:
            s = sweep.data
            s = s[self.signal_start:self.signal_stop]
            s = self.low_pass_filter(s)
            s = pd.Series(s)
            sweep.group = SignalGroup.ACTIVITY if (s > self.trigger).sum() > 0 else SignalGroup.NOISE
        
        self.on_signal_detect_complete.emit()

    def save(self, save_location):
        self._save_location = save_location

        load_location = Path(self._file_location)
        save_location = Path(self._save_location)
        fname = load_location.parts[-1].split('.')[0]
        data = np.stack([sweep.data for sweep in self.sweeps if sweep.group == SignalGroup.ACTIVITY])
        outfile = save_location / f"{fname}_signals.abf"
        pyabf.abfWriter.writeABF1(data, outfile, self.sample_rate)
        
        data = {
            'meta': {
                'sample_rate': int(self.sample_rate),
                'out_file': str(outfile),
                'original_file': str(self._file_location)
            },
            'data': [s.to_dict() for s in self.sweeps]
        }
        with open(save_location / f"{fname}_signals.pkl", 'wb') as fp:
            pickle.dump(data, fp)