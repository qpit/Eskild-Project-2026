from __future__ import annotations
import pyvisa
import numpy as np
import h5py as h5
import time

from .digitizer_base import Digitizer
class Scope(Digitizer):
    def __init__(self, visa_address: str = 'TCPIP0::k-dx3024g-60147.local::inst0::INSTR'):
        self.rm = pyvisa.ResourceManager()
        self.DSO = None
        self._last_t = None
        self._last_y = None

        self.connect(visa_address)

    # lifecycle
    def connect(self, address: str | None = None) -> None:
        if address:
            self.VISAAddress = address
        self.DSO = self.rm.open_resource(self.VISAAddress, open_timeout=10000)
        self.DSO.timeout = 600000  # 900 seconds (15 minutes) for long acquisitions
        self.DSO.chunk_size = 2048 * 2048
        self.DSO.read_termination = '\n'
        self.DSO.write_termination = '\n'
        self.DSO.visalib.set_buffer(self.DSO.session, pyvisa.constants.VI_IO_OUT_BUF, 16000000)
        _ = self.DSO.query('*IDN?')
        self.DSO.write('*CLS')
        self.DSO.write(':WAVeform:POINts:MODE MAXimum')
        self.DSO.write(':WAVEFORM:FORMAT WORD')
        self.DSO.write(':WAVEFORM:BYTEORDER LSBFirst')
        # Use normal acquisition to avoid min/max interleaved samples.
        self.DSO.write(':ACQuire:TYPE NORM')

    def disconnect(self) -> None:
        try:
            if self.DSO is not None:
                self.DSO.close()
        except Exception:
            pass
        self.DSO = None

    # configuration
    def set_trigger(self, trigger_type: str = "software", **kwargs) -> None:
        # Placeholder: adapt SCPI here if needed
        return None

    # properties mirrored from original
    @property
    def y_offset(self):
        """Gets the channel offset."""
        return float(self.DSO.query(f':CHAN{self.channel}:OFFSET?'))
    @y_offset.setter
    def y_offset(self, value):
        """Sets the channel offset."""
        self.DSO.write(f':CHAN{self.channel}:OFFSET {value}')

    @property
    def y_range(self):
        """Gets the Y-axis range."""
        return float(self.DSO.query(f':CHAN{self.channel}:RANGE?'))
    
    @y_range.setter
    def y_range(self, value):
        """Sets the Y-axis range."""
        self.DSO.write(f':CHAN{self.channel}:RANGE {value}')

    @property
    def timespan(self):
        """Gets the timebase range."""
        return float(self.DSO.query(':TIMebase:RANGe?'))
    @timespan.setter
    def timespan(self, value):
        """Sets the timebase range."""
        self.DSO.write(f':TIMebase:RANGe {value}')

    @property
    def points(self):
        """Gets the number of points that will be recorded"""
        return int(self.DSO.query(':WAVeform:POINts?'))
    @points.setter
    def points(self, value):
        """Sets the number of points"""
        self.DSO.write(f':WAVeform:POINts {value//2}')

    # unified acquisition
    def acquire(self, channels=None, num_points=None, sample_rate=None, **kwargs):
        self.channel = channels
        self.DSO.write(f':WAVEFORM:SOURCE CHAN{channels}')
        self.DSO.write(':DIGITIZE')
        y_raw = np.array(self.DSO.query_binary_values(':WAV:DATA?', datatype='H', is_big_endian=False), dtype=np.uint16)
        yinc = float(self.DSO.query(':WAV:YINC?'))
        yorigin = float(self.DSO.query(':WAV:YOR?'))
        yref = float(self.DSO.query(':WAV:YREF?'))
        y = (y_raw.astype(float) - yref) * yinc + yorigin
        N = len(y)
        try:
            xinc = float(self.DSO.query(':WAV:XINC?'))
            xorigin = float(self.DSO.query(':WAV:XOR?'))
            t = np.arange(N) * xinc + xorigin
        except Exception:
            Ts = self.timespan
            t = np.linspace(0.0, Ts, N, endpoint=False)
        self._last_t = t
        self._last_y = y
        return t, y
    def acquire2(self, channels=None, **kwargs):
        if channels is None: channels = [1]
        if isinstance(channels, int): channels = [channels]

        # 1. Stop and Setup
        self.DSO.write(":STOP")
        time.sleep(0.1)
        self.DSO.write(":WAVeform:POINts:MODE NORMal") # Normal mode is safer for dongles
        self.DSO.write(":WAVeform:FORMat BYTE")       # Byte is the smallest data size

        y_data = []
        for ch in channels:
            # 2. Select Source
            self.DSO.write(f":WAVEFORM:SOURCE CHAN{ch}")
            time.sleep(0.1) # CRITICAL: Give the dongle time to process the command
            
            # 3. Query binary data with an explicit delay
            # We use a 0.2s delay to prevent "Query Unterminated"
            y_raw = self.DSO.query_binary_values(
                ":WAVeform:DATA?", 
                datatype='B', 
                is_big_endian=True, 
                delay=0.2 
            )
            y_raw = np.array(y_raw, dtype=np.uint8)

            # 4. Get Scaling (one by one)
            yinc = float(self.DSO.query(":WAVeform:YINC?"))
            yorigin = float(self.DSO.query(":WAVeform:YOR?"))
            yref = float(self.DSO.query(":WAVeform:YREF?"))
            
            y = (y_raw.astype(float) - yref) * yinc + yorigin
            y_data.append(y)

        # 5. Build Time Axis
        Ts = self.timespan
        N = len(y_data[0])
        t = np.linspace(0.0, Ts, N, endpoint=False)
        
        self.DSO.write(":RUN") # Put the scope back in Run mode
        return t, np.vstack(y_data)

    def acquire_segmented(self, channels, points=None, seg=1):
         """
         Acquire segmented data. If points is None, infer points per segment
         from the returned waveform length.
         """
         if channels is None:
             channels = [1]
         if isinstance(channels, int):
             channels = [channels]

         for channel in channels:
             self.DSO.write(f'CHAN{int(channel)}:DISP ON')
         self.DSO.write(":ACQuire:MODE SEGM")
         self.DSO.write(f":ACQuire:SEGMented:COUNt {int(seg)}")
         if points is not None:
             self.DSO.write(f"ACQUIRE:POINTS {int(points)}")
         self.DSO.write(":WAVeform:SEGMented:ALL 1")
         self.DSO.write(':DIG')  # acquire data and wait until finish.

         # Read first channel to determine points if needed.
         self.DSO.write(f":WAV:SOUR CHAN{int(channels[0])}")
         first_data = self.DSO.query_binary_values(":WAV:DATA?", datatype="H", is_big_endian=False)
         if points is None:
             total_points = len(first_data)
             if total_points % int(seg) != 0:
                 raise ValueError("Segmented data length is not divisible by seg; cannot infer points.")
             points = total_points // int(seg)
         first_data = np.reshape(first_data, (int(seg), int(points)))

         Y = np.zeros((len(channels), int(seg), int(points)))
         yinc = float(self.DSO.query(":WAV:YINC?"))
         yorigin = float(self.DSO.query(":WAV:YOR?"))
         yref = float(self.DSO.query(":WAV:YREF?"))
         Y[0, :, :] = (np.array(first_data) - yref) * yinc + yorigin

         for c, channel in enumerate(channels[1:], start=1):
             self.DSO.write(f":WAV:SOUR CHAN{int(channel)}")
             data = self.DSO.query_binary_values(":WAV:DATA?", datatype="H", is_big_endian=False)
             data = np.reshape(data, (int(seg), int(points)))
             yinc = float(self.DSO.query(":WAV:YINC?"))
             yorigin = float(self.DSO.query(":WAV:YOR?"))
             yref = float(self.DSO.query(":WAV:YREF?"))
             Y[c, :, :] = (np.array(data) - yref) * yinc + yorigin

         xinc = float(self.DSO.query(":WAV:XINC?"))
         xorigin = float(self.DSO.query(":WAV:XOR?"))
         x = np.arange(len(Y[0, 0, :])) * xinc + xorigin
         self.DSO.write(":ACQuire:MODE RTIM")
         self.DSO.write(':RUN')
         return x, Y

    def save_data(self, filename, folder = r"C:\Users\qpitlab\Videos\mb-gbs-experiment\Notebooks\data" , data_set_name = "data"):
        with h5.File(folder+"\\" + filename + ".hdf5",'a') as f:
            f.create_dataset(data_set_name,data = np.vstack([self._last_y,self._last_t]))