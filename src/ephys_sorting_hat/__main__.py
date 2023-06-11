import sys
import matplotlib
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd

from ephys_sorting_hat.model import Model, SignalGroup, Sweep
from matplotlib.figure import Figure
from PyQt6 import QtGui
from PyQt6 import QtCore
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator, QDoubleValidator
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
# matplotlib.use('Qt5Agg')

LOAD_LABEL_WIDTH = 120
SAVE_LABEL_WIDTH = 120

def pass_filter(data, lowband, upperband, sample_rate):
    upperband_index = int(upperband * data.size / sample_rate)
    fsig = np.fft.fft(data)
    for i in range(upperband_index + 1, len(fsig) - upperband_index):
        fsig[i] = 0

    lowband_index = int(lowband * data.shape[0] / sample_rate)
    for i in range(lowband_index):
        fsig[i] = 0
        fsig[len(fsig)-i-1] = 0

    data_filtered = np.fft.ifft(fsig)
    return np.real(data_filtered)

class LoadFileLayout(QtWidgets.QHBoxLayout):
    load_file_event = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        label = QtWidgets.QLabel("Open .abf File")
        # label.setFixedWidth(LOAD_LABEL_WIDTH)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.load_file_input = QtWidgets.QLineEdit()
        buttons = QtWidgets.QHBoxLayout()
        browse = QtWidgets.QPushButton("Browse")
        browse.setFixedWidth(100)
        load = QtWidgets.QPushButton("Load")
        load.setFixedWidth(100)
        
        browse.clicked.connect(self.on_browse)
        load.clicked.connect(self.on_load)

        buttons.setSpacing(0)
        label.setContentsMargins(0,0,10,0)
        load.setContentsMargins(0,0,0,0)
        browse.setContentsMargins(0,0,0,0)
        buttons.setContentsMargins(0,0,0,0)
        buttons.addWidget(browse,0)
        buttons.addWidget(load,0)

        self.setSpacing(0)
        self.setContentsMargins(0,0,0,0)
        self.addWidget(label)
        self.addWidget(self.load_file_input)
        self.addLayout(buttons)
        
    def on_browse(self, event):
        dialog = QtWidgets.QFileDialog()
        filepath, _ = dialog.getOpenFileName(None, "Load .abf file", filter="abf files (*.abf)")
        if filepath:
            self.load_file_input.setText(filepath)

    def on_load(self, event):
        self.load_file_event.emit()

    @property
    def value(self):
        return self.load_file_input.text()
    
class SaveFileLayout(QtWidgets.QHBoxLayout):
    save_file_event = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        label = QtWidgets.QLabel("Select Output Folder")
        # label.setFixedWidth(SAVE_LABEL_WIDTH)
        # label.setAlignment(Qt.AlignmentFlag.AlignRight, Qt.AlignVCenter)
        # label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.save_file_input = QtWidgets.QLineEdit()
        buttons = QtWidgets.QHBoxLayout()
        browse = QtWidgets.QPushButton("Browse")
        browse.setFixedWidth(100)
        save = QtWidgets.QPushButton("Save")
        save.setFixedWidth(100)
        
        browse.clicked.connect(self.on_browse)
        save.clicked.connect(self.on_save)

        buttons.setSpacing(0)
        label.setContentsMargins(0,0,10,0)
        save.setContentsMargins(0,0,0,0)
        browse.setContentsMargins(0,0,0,0)
        buttons.setContentsMargins(0,0,0,0)
        buttons.addWidget(browse,0)
        buttons.addWidget(save,0)

        
        self.setContentsMargins(0,0,0,0)
        self.setSpacing(0)
        self.addWidget(label)
        self.addWidget(self.save_file_input)
        self.addLayout(buttons)
        
    def on_browse(self, event):
        dialog = QtWidgets.QFileDialog()
        filepath = dialog.getExistingDirectory(None, "Save directory")
        if filepath:
            self.save_file_input.setText(filepath)

    def on_save(self, event):
        self.save_file_event.emit()

    @property
    def value(self):
        return self.save_file_input.text()


class GraphWidgetWrapper(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0,0,0,0)
        self.graph_widget = GraphWidget()

        # Override the default tool item
        NavigationToolbar2QT.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Back', 'Back to previous view', 'back', 'back'),
            ('Forward', 'Forward to next view', 'forward', 'forward'),
            ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'),
            ('Pan',
                'Left button pans, Right button zooms\n'
                'x/y fixes axis, CTRL fixes aspect',
                'move', 'pan'),
        )
        self.toolbar = NavigationToolbar2QT(self.graph_widget, self)
        self.toolbar.setFixedHeight(20)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.graph_widget)
        self.setLayout(layout)


class GraphWidget(FigureCanvasQTAgg):
    limits_updated = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None, width=6, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlabel("ms")
        self.axes.set_ylabel("pA")
        self.line,*_ = self.axes.plot([],[], linewidth=1.0, label="Raw Signal")
        self.smoothed_line,*_ = self.axes.plot([],[], linewidth=1.0, label="Bandpass Signal")
        self.trigger_line,*_ = self.axes.plot([],[], linewidth=1.5, color='red', linestyle='dashed', label="Bandpass Trigger")
        self.axes.legend(loc='upper right')
        self.axes.grid()
        super().__init__(self.fig)
        self.setContentsMargins(0,0,0,0)

        self.lowband = None
        self.highband = None
        self.sweep = None

        self.settings_widget: SettingsWidget = None

    def plot_sweep(self, sweep):
        self.sweep = sweep
        self.line.set_data(sweep.time,sweep.data)

        if self.settings_widget is not None:
            band_information = self.settings_widget.get_band_information()
            lowband = band_information['lowband']
            lowband = 0 if lowband is None else lowband

            highband = band_information['highband']
            highband = 0 if highband is None else highband
            y = pass_filter(self.sweep.data, lowband, highband, sample_rate=self.sweep.sample_rate)
            self.smoothed_line.set_data(self.sweep.time, y)

        self.fig.canvas.draw()

    def on_plot_limits_changed(self, plot_limits):
        xmin, xmax = self.axes.get_xlim()
        ymin, ymax = self.axes.get_ylim()

        xmin = xmin if plot_limits['xmin'] is None else plot_limits['xmin']
        xmax = xmax if plot_limits['xmax'] is None else plot_limits['xmax']
        ymin = ymin if plot_limits['ymin'] is None else plot_limits['ymin']
        ymax = ymax if plot_limits['ymax'] is None else plot_limits['ymax']

        self.axes.set_xlim([xmin,xmax])
        self.axes.set_ylim([ymin,ymax])
        self.fig.canvas.draw()
        self.limits_updated.emit(plot_limits)

    def update_trigger(self, trigger_limits):
        xmin = trigger_limits['xmin']
        xmax = trigger_limits['xmax']
        y = trigger_limits['y']
        self.trigger_line.set_data([xmin, xmax], [y,y])
        self.fig.canvas.draw()
        # print('updated')

    def update_bandwidth(self, bandlimits):
        try:
            lowband = bandlimits['lowband']
            highband = bandlimits['highband']
            # print(self.sweep.data, lowerband, highband, self.sweep.sample_rate)
            # print('update bandwidth')
        except:
            print('error bandwidth')
            # likely if sweep is not defined

        if self.sweep is not None:
            y = pass_filter(self.sweep.data, lowband, highband, sample_rate=self.sweep.sample_rate)
            self.smoothed_line.set_data(self.sweep.time, y)

        self.fig.canvas.draw()

class SettingsWidget(QtWidgets.QTabWidget):
    plot_limits_changed_event = QtCore.pyqtSignal(dict)
    show_smoothed_plot = QtCore.pyqtSignal(bool)
    show_triggers = QtCore.pyqtSignal(bool)
    apply = QtCore.pyqtSignal()
    trigger_changed = QtCore.pyqtSignal(dict)
    band_changed = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(180)
        self.plotting_tab = QtWidgets.QWidget()
        self.bandpass_tab = QtWidgets.QWidget()

        self.addTab(self.plotting_tab, "Plotting")
        # self.addTab(self.bandpass_tab, "Trigger")

        self.setup_plotting_tab()
        # self.setup_bandpass_tab()
        self.setStyleSheet('''
        QTabWidget::tab-bar {
            alignment: left;
        }''')

    def setup_plotting_tab(self):
        layout = QtWidgets.QHBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        column = QtWidgets.QFormLayout()
        column.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self.plot_xmin_input = QtWidgets.QLineEdit()
        self.plot_xmax_input = QtWidgets.QLineEdit()
        self.plot_ymin_input = QtWidgets.QLineEdit()
        self.plot_ymax_input = QtWidgets.QLineEdit()

        self.xmin_validator = QDoubleValidator()
        self.xmax_validator = QDoubleValidator()
        self.ymin_validator = QDoubleValidator()
        self.ymax_validator = QDoubleValidator()

        column.addRow("View Min. Time (ms)", self.plot_xmin_input)
        column.addRow("View Max. Time (ms)", self.plot_xmax_input)
        column.addRow("View Min. Amplitude (ms)", self.plot_ymin_input)
        column.addRow("View Max. Amplitude (ms)", self.plot_ymax_input)

        self.plot_xmin_input.setValidator(self.xmin_validator)
        self.plot_xmax_input.setValidator(self.xmax_validator)
        self.plot_ymin_input.setValidator(self.ymin_validator)
        self.plot_ymax_input.setValidator(self.ymax_validator)

        layout.addLayout(column)

        self.trigger_xmin = QtWidgets.QLineEdit()
        self.trigger_xmin.setValidator(QDoubleValidator())
        self.trigger_xmax = QtWidgets.QLineEdit()
        self.trigger_xmax.setValidator(QDoubleValidator())
        self.trigger_ysmoothed = QtWidgets.QLineEdit()
        self.trigger_ysmoothed.setValidator(QDoubleValidator())
        self.lowband_input = QtWidgets.QLineEdit()
        self.lowband_input.setValidator(QIntValidator())
        self.highband_input = QtWidgets.QLineEdit()
        self.highband_input.setValidator(QIntValidator())
        # self.show_smoothed_checkbox = QtWidgets.QCheckBox()
        # self.show_smoothed_checkbox.setChecked(True)
        # self.show_triggers_checkbox = QtWidgets.QCheckBox()
        # self.show_triggers_checkbox.setChecked(True)
        self.apply_button = QtWidgets.QPushButton("Apply")

        # layout = QtWidgets.QHBoxLayout()
        column = QtWidgets.QFormLayout()
        # column.setFieldGrowthPolicy(column.FieldsStayAtSizeHint)
        # column.addRow("Trigger Raw (pA)", self.trigger_yraw)
        column.addRow("Trigger Min. Time (ms)", self.trigger_xmin)
        column.addRow("Trigger Max. Time (ms)", self.trigger_xmax)
        layout.addLayout(column)
        
        column = QtWidgets.QFormLayout()
        # column.setFieldGrowthPolicy(column.FieldsStayAtSizeHint)
        column.addRow("Trigger Smoothed (pA)", self.trigger_ysmoothed)
        column.addRow("Low band (Hz)", self.lowband_input)
        column.addRow("High band (Hz)", self.highband_input)
        column.addRow("", self.apply_button)
        layout.addLayout(column)

        layout.addStretch()
        self.plotting_tab.setLayout(layout)

        self.plot_xmin_input.setText("0.0")
        self.plot_xmax_input.setText("1.0")
        self.plot_ymin_input.setText("-10")
        self.plot_ymax_input.setText("50")
        self.trigger_xmin.setText("0.0")
        self.trigger_xmax.setText("0.1")
        self.trigger_ysmoothed.setText("10")
        self.lowband_input.setText("2")
        self.highband_input.setText("100")

        self.plot_xmin_input.editingFinished.connect(self.on_plot_limits_changed)
        self.plot_xmax_input.editingFinished.connect(self.on_plot_limits_changed)
        self.plot_ymin_input.editingFinished.connect(self.on_plot_limits_changed)
        self.plot_ymax_input.editingFinished.connect(self.on_plot_limits_changed)
        self.trigger_ysmoothed.editingFinished.connect(self.on_trigger_changed)
        self.trigger_xmax.editingFinished.connect(self.on_trigger_changed)
        self.lowband_input.editingFinished.connect(self.on_band_changed)
        self.trigger_xmin.editingFinished.connect(self.on_trigger_changed)
        self.highband_input.editingFinished.connect(self.on_band_changed)
        self.apply_button.clicked.connect(lambda: self.apply.emit())
        
        self.on_plot_limits_changed()

    def get_band_information(self):
        lowband = self.lowband_input.text()
        highband = self.highband_input.text()
        # print('highband', highband)
        # print('lowband', lowband)
        float_or_none = lambda x: None if isinstance(x,str) and len(x)==0 else float(x)
        return {
            'lowband': float_or_none(lowband),
            'highband': float_or_none(highband)
        }

    def get_trigger_information(self):
        xmin = self.trigger_xmin.text()
        xmax = self.trigger_xmax.text()
        y = self.trigger_ysmoothed.text()
        float_or_none = lambda x: None if isinstance(x,str) and len(x)==0 else float(x)
        return {
            'xmin': float_or_none(xmin), 
            'xmax': float_or_none(xmax), 
            'y': float_or_none(y),
        }

    def on_band_changed(self, event=None):
        lowband = self.lowband_input.text()
        highband = self.highband_input.text()
        try:
            lowband = float(lowband)
            highband = float(highband)
            self.band_changed.emit({'lowband': lowband, 'highband': highband})
        except:
            pass

    def on_trigger_changed(self, event=None):
        self.trigger_changed.emit(self.get_trigger_information())

    def update_plot_limits(self, plot_limits):
        string_or_none = lambda x: str(round(x,5)) if x is not None else ""
        self.plot_xmin_input.setText(string_or_none(plot_limits['xmin']))
        self.plot_xmax_input.setText(string_or_none(plot_limits['xmax']))
        self.plot_ymin_input.setText(string_or_none(plot_limits['ymin']))
        self.plot_ymax_input.setText(string_or_none(plot_limits['ymax']))

    def on_plot_limits_changed(self, event=None):
        try:
            float_or_none = lambda x: None if isinstance(x,str) and len(x)==0 else float(x)
            self.plot_limits_changed_event.emit({
                'xmin': float_or_none(self.plot_xmin_input.text()),
                'xmax': float_or_none(self.plot_xmax_input.text()),
                'ymin': float_or_none(self.plot_ymin_input.text()),
                'ymax': float_or_none(self.plot_ymax_input.text()),
            })
            # print('emitted')
        except:
            print('plot limit error')
        # except ValueError as e:
        #     QtWidgets.QMessageBox.about(self,'Error',"Plot limits (xmin, xmax, ymin, ymax) must be valid numbers and not empty")

class SignalListWidgetItem(QtWidgets.QListWidgetItem):
    def __init__(self, sweep):
        self.sweep = sweep
        self.label = f"Sweep {self.sweep.number}"
        super().__init__(self.label)

    def __lt__(self, other):
        try:
            return float(self.sweep.number) < float(other.sweep.number)
        except Exception:
            return QListWidgetItem.__lt__(self, other)

class ListWidget(QtWidgets.QListWidget):
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
    focused = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.keyPressed.emit(event) 
        event.accept()

    def focusInEvent(self, e):
        super(QtWidgets.QListWidget, self).focusInEvent(e)
        self.focused.emit()

class SignalListView(QtWidgets.QHBoxLayout):
    sweep_changed_event = QtCore.pyqtSignal(Sweep)
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)

    def __init__(self):
        super().__init__()
        self.setContentsMargins(0,0,0,0)
        self.setSpacing(0)

        signal_vlayout = QtWidgets.QVBoxLayout()
        signal_label = QtWidgets.QLabel("Signal")
        signal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        signal_vlayout.addWidget(signal_label)
        self.signal_list = ListWidget()
        signal_vlayout.addWidget(self.signal_list)

        noise_vlayout = QtWidgets.QVBoxLayout()
        noise_vlayout = QtWidgets.QVBoxLayout()
        noise_label = QtWidgets.QLabel("Noise")
        noise_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        noise_vlayout.addWidget(noise_label)

        self.noise_list = ListWidget()
        noise_vlayout.addWidget(self.noise_list)

        self.signal_list.setFixedWidth(100)
        self.noise_list.setFixedWidth(100)

        self.addLayout(signal_vlayout)
        self.addLayout(noise_vlayout)

        self.noise_list.itemSelectionChanged.connect(self.on_noise_item_selection_changed)
        self.signal_list.itemSelectionChanged.connect(self.on_signal_item_selection_changed)
        self.noise_list.keyPressed.connect(self.on_key_pressed_from_noise_list)
        self.signal_list.keyPressed.connect(self.on_key_pressed_from_signal_list)
        self.signal_list.focused.connect(self.on_signal_item_selection_changed)
        self.noise_list.focused.connect(self.on_noise_item_selection_changed)

    def update_sweeps(self, sweeps):
        self.noise_list.clear()
        self.signal_list.clear()
        # print("update sweeps")
        for sweep in sweeps:
            sweep_item = SignalListWidgetItem(sweep)
            if sweep.group == SignalGroup.ACTIVITY:
                self.signal_list.addItem(sweep_item)
            elif sweep.group == SignalGroup.NOISE:
                self.noise_list.addItem(sweep_item)

    def on_noise_item_selection_changed(self):
        if not self.noise_list.hasFocus():
            return 

        item = self.noise_list.currentItem()
        if item is not None:
            sweep = item.sweep
            # print("Item changed", sweep.number)
            self.sweep_changed_event.emit(sweep)


    def on_signal_item_selection_changed(self):
        if not self.signal_list.hasFocus():
            return 

        item = self.signal_list.currentItem()
        if item is not None:
            sweep = item.sweep
            # print("Item changed", sweep.number)
            self.sweep_changed_event.emit(sweep)

    def on_key_pressed_from_signal_list(self, event):
        try:
            count = self.signal_list.count()
            row = self.signal_list.currentRow()
            sweep = self.signal_list.currentItem().sweep

            if event.key() == Qt.Key.Key_Right:
                # print("Move to noise")
                sweep.group = SignalGroup.NOISE

                # In the other list, set the row to active
                for i in range(self.noise_list.count()):
                    item = self.noise_list.item(i)
                    if item.sweep == sweep:
                        self.noise_list.scrollToItem(item)
                        self.noise_list.setCurrentItem(item)

                # Set the next row to active
                if row + 1 < count:
                    self.signal_list.setCurrentRow(row)
                elif count > 0:
                    self.signal_list.setCurrentRow(row-1)

        except AttributeError as e:
            pass

    def on_key_pressed_from_noise_list(self, event):
        try:
            count = self.noise_list.count()
            row = self.noise_list.currentRow()
            sweep = self.noise_list.currentItem().sweep

            if event.key() == Qt.Key.Key_Left:
                # print("move to signal")
                sweep.group = SignalGroup.ACTIVITY

                # In the other list, set the row to active
                for i in range(self.signal_list.count()):
                    item = self.signal_list.item(i)
                    if item.sweep == sweep:
                        self.signal_list.scrollToItem(item)
                        self.signal_list.setCurrentItem(item)

                # Set the next row to active
                if row + 1 < count:
                    self.noise_list.setCurrentRow(row)
                elif count > 0:
                    self.noise_list.setCurrentRow(row-1)

        except AttributeError as e:
            pass

class View(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.model = Model()

        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.load_file_layout = LoadFileLayout()
        layout.addLayout(self.load_file_layout)
        center_layout = QtWidgets.QHBoxLayout()
        center_left_layout = QtWidgets.QVBoxLayout()

        self.graph_widget_wrapper = GraphWidgetWrapper()
        self.graph_widget = self.graph_widget_wrapper.graph_widget
        self.settings_widget = SettingsWidget()

        center_left_layout.addWidget(self.graph_widget_wrapper)
        center_left_layout.addWidget(self.settings_widget)

        center_layout.addLayout(center_left_layout)

        self.signal_list_view = SignalListView()
        center_layout.addLayout(self.signal_list_view)
        layout.addLayout(center_layout)

        self.save_file_widget = SaveFileLayout()
        layout.addLayout(self.save_file_widget)
        self.setLayout(layout)

        self.load_file_layout.load_file_event.connect(self.load)
        self.save_file_widget.save_file_event.connect(self.save)
        self.model.on_sweeps_changed.connect(self.update_sweeps)
        self.signal_list_view.sweep_changed_event.connect(self.graph_widget.plot_sweep)
        self.settings_widget.plot_limits_changed_event.connect(self.graph_widget.on_plot_limits_changed)
        self.model.on_load_complete.connect(self.reset_plot_limits)
        self.graph_widget.limits_updated.connect(self.settings_widget.update_plot_limits)
        self.settings_widget.trigger_changed.connect(self.graph_widget.update_trigger)
        self.settings_widget.band_changed.connect(self.graph_widget.update_bandwidth)
        self.settings_widget.apply.connect(self.autosort_sweeps)

        # For communication
        self.graph_widget.settings_widget = self.settings_widget

        self.settings_widget.on_plot_limits_changed()
        self.settings_widget.on_trigger_changed(None)
        self.settings_widget.on_band_changed(None)
        self.settings_widget.on_trigger_changed(None)

    def setup_center_area(self):
        layout = QtWidgets.QHBoxLayout()
        left_vbox = QtWidgets.QVBoxLayout()
        right_vbox = QtWidgets.QVBoxLayout()
        layout.addLayout(left_vbox)
        layout.addLayout(right_vbox)
        
    def load(self):
        try:
            self.model.load_abf_file(self.load_file_layout.value)
        except:
            QtWidgets.QMessageBox.about(self,'Error',"Invalid .abf file path")

    def save(self):
        if self.save_file_widget.value is not None:
            self.model.save(self.save_file_widget.value)

    def update_sweeps(self):
        self.signal_list_view.update_sweeps(self.model.sweeps)

    def reset_plot_limits(self):
        data = []
        for sweep in self.model.sweeps:
            data.append({
                'xmin': min(sweep.time),
                'xmax': max(sweep.time),
                'ymin': min(sweep.data),
                'ymax': max(sweep.data),
            })

        df = pd.DataFrame(data)
        xmin, ymin = df[['xmin', 'ymin']].min()
        xmax, ymax = df[['xmax', 'ymax']].max()

        self.graph_widget.on_plot_limits_changed(dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax))

    def autosort_sweeps(self):
        trigger_information = self.settings_widget.get_trigger_information()
        band_information = self.settings_widget.get_band_information()

        empty_keys = []
        for key, value in {**trigger_information, **band_information}.items():
            if value is None:
                empty_keys.append(key)

        if len(empty_keys):
            QtWidgets.QMessageBox.about(self,'Error',"Fields cannot be empty: " + ', '.join(empty_keys))

        xmin = trigger_information['xmin']
        xmax = trigger_information['xmax']
        y = trigger_information['y']

        highband = band_information['highband']
        lowband = band_information['lowband']

        for sweep in self.model.sweeps:
            s = pass_filter(sweep.data, lowband, highband, sample_rate=sweep.sample_rate)
            s = s[(sweep.time>=xmin) & (sweep.time<=xmax)]

            if sweep.was_moved_by_user is False:
                if (s > y).sum():
                    sweep.group = SignalGroup.ACTIVITY
                    sweep.was_moved_by_user = False # Reset this manually
                else:
                    sweep.group = SignalGroup.NOISE
                    sweep.was_moved_by_user = False # Reset this manually


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Electrophysiology Sorting Hat")
        view = View()
        self.setCentralWidget(view)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    icon_path = str(Path(__file__).parent / 'hat-wizard-solid.png')
    icon = QtGui.QIcon(icon_path)
    app.setWindowIcon(icon)
    
    w = MainWindow()
    w.show()
    app.exec()