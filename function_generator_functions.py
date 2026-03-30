import pyvisa as visa
import numpy as np
import time

def find_funcgen(pattern = "SDG2XFBX801056"):
    rm=visa.ResourceManager()
    visa_address=rm.list_resources()
    for i, string in enumerate(visa_address):
        try: 
            inst = rm.open_resource(visa_address[i])
            string = inst.query('*IDN?')
            if pattern in string:
                return visa_address[i]
        except:
            pass
    print("Func Gen: cannot find the specified function generator.")

class RSDG:
    def __init__(self, visa_address, name = 'RSDG'):
        self.name = name
        self.visa_address = visa_address
        self.connect()

    def connect(self):
        try:
            self.rm = visa.ResourceManager()
            self.inst = self.rm.open_resource(self.visa_address)
            # Set timeout for commands (in milliseconds)
            self.inst.timeout = 2000
            print(f"Func Gen: connected to: {self.inst.query('*IDN?')}")
        except Exception as e:
            print(f"Func Gen: cannot connect to instrument. Error: {e}")
            raise

    def set_output(self, chn, enable = True, load = '50'):
        r'''
        config the output
        
        args:
        ----------
            chn : {1, 2}
                output channel
            enable : bool, optional
                set whether the channels are enabled or disabled
            load : {'50', 'HZ'}, optional
                set output impedance of the channel: 50\Ohm or high impedance
        '''
        if ('OUTP ON' in self.inst.query(f'C{chn:d}:OUTP?')) != enable:
            self.inst.write('C' + str(chn) + ':OUTP ' + ('ON' if enable else 'OFF') + ',LOAD,' + load + ',PLRT,NOR')
        self.inst.query('*OPC?')

    # for future: WVTP := {SINE，SQUARE，RAMP，PULSE，NOISE，ARB，DC，PRBS，IQ}。
    def set_square_wave(self, chn, phase = None, duty = None, freq = None, period = None, 
                        amp = None, offset = None, high = None, low = None):
        r'''
        config the waveform as a square wave, which is the only needed waveform for ee-dl trig

        the arguments {frequency, amp, offset} overshadows {period, high, low}
        {amp, offset} or {high, low} must be defined together
        high = amp / 2 + offset, low = - amp / 2 + offset
        
        args:
        ----------
            chn : {1, 2}
                output channel
            phase : optional, in the unit of degrees
                set the starting phase, defined as the location of the rising edge wrt the entire period
            duty : optional, in the unit of percent
                set the duty cycle, defined as the up time of the square wave
            freq, period, amp, offset, high, low: optional, self-evident
        '''
        try:
            ctrl_str1 = f'PHSE,{phase:.2f},' if -360 <= phase <= 360 else ''
        except:
            ctrl_str1 = ''
        try:
            ctrl_str2 = f'DUTY,{duty:.2f},' if 0 <= duty <= 100 else ''
        except:
            ctrl_str2 = ''
        try: 
            ctrl_str3 = f'FRQ,{freq:.3f}HZ,' if 1e-3 <= freq <= 1e6 else ''
        except: 
            try: 
                ctrl_str3 = f'PERI,{period:.6f}S,' if 1e-6 <= period < 1e3 else ''
            except: 
                ctrl_str3 = ''
        try: 
            ctrl_str4 = f'AMP,{amp:.2f}V,OFST,{offset:.2f}V,' if 0 <= amp <= 5 and np.abs(offset) + amp <= 5 else ''
        except:
            try:
                ctrl_str4 = f'HLEV,{high:.2f}V,LLEV,{low:.2f}V,' if -5 <= low <= high <= 5 else ''
            except:
                ctrl_str4 = ''
        self.inst.write('C' + str(chn) + ':BSWV WVTP,SQUARE,' + ctrl_str1 + ctrl_str2 + ctrl_str3 + ctrl_str4 + 'MAX_OUTPUT_AMP,20V')
        # print('C' + str(chn) + ':BSWV WVTP,SQUARE,' + ctrl_str1 + ctrl_str2 + ctrl_str3 + ctrl_str4 + 'MAX_OUTPUT_AMP,20V')
        self.inst.query('*OPC?')
        
    def set_DC_wave(self, chn, offset = None):
        r'''
        config the waveform as a DC output

        args:
        ----------
            chn : {1, 2}
                output channel
            offset: optional, self-evident
        '''
        try: 
            ctrl_str = f'OFST,{offset:.2f}V,'
        except:
            ctrl_str = ''
        self.inst.write('C' + str(chn) + ':BSWV WVTP,DC,' + ctrl_str + 'MAX_OUTPUT_AMP,20V')
        self.inst.query('*OPC?')
        
    def set_sine_wave(self, chn, phase=None, freq=None, period=None,
                  amp=None, offset=None):
        '''
        Configure a SINE wave on the specified channel.

        Args:
            chn : {1, 2}
                Output channel.
            phase : float, optional (degrees)
                Initial phase of sine wave.
            freq : float, optional (Hz)
                Frequency. Overrides period.
            period : float, optional (seconds)
                Period (used only if freq is not given).
            amp : float, optional (Vpp)
                Amplitude (peak-to-peak).
            offset : float, optional (V)
                DC offset.
        '''

        # ----- Build parameter strings -----
        try:
            ctrl_phase = f'PHSE,{phase:.2f},' if -360 <= phase <= 360 else ''
        except:
            ctrl_phase = ''

        try:
            ctrl_freq = f'FRQ,{freq:.6f}HZ,' if 1e-3 <= freq <= 1e8 else ''
        except:
            try:
                ctrl_freq = f'PERI,{period:.6f}S,' if 1e-9 <= period < 1e3 else ''
            except:
                ctrl_freq = ''

        try:
            ctrl_amp = f'AMP,{amp:.4f}V,' if 0 <= amp <= 5 else ''
        except:
            ctrl_amp = ''

        try:
            ctrl_offset = f'OFST,{offset:.4f}V,' if -2.5 <= offset <= 2.5 else ''
        except:
            ctrl_offset = ''

        # ----- Send SCPI -----
        scpi = (
            f'C{chn}:BSWV '
            f'WVTP,SINE,'    # <-- The important change
            f'{ctrl_phase}'
            f'{ctrl_freq}'
            f'{ctrl_amp}'
            f'{ctrl_offset}'
            'MAX_OUTPUT_AMP,20V'
        )

        self.inst.write(scpi)
        self.inst.query('*OPC?')  # wait until done

    def activate_burst_mode(self, ch):
        # --- Burst Mode Settings ---
        inst = self.inst
        inst.write(f"C{ch}:BTWV STATE,ON")         # Enable Burst Mode
        inst.write(f"C{ch}:BTWV BURS,TRIG")        # Use trigger burst mode
        inst.write(f"C{ch}:BTWV TRSR,EXT")         # External trigger (or INT)
        inst.write(f"C{ch}:BTWV GATE,NORM")        # Normal gate mode
        inst.write(f"C{ch}:BTWV Ncycle,1")         # Number of cycles per burst
        
    def turn_on(self, ch):
        self.inst.write(f"C{ch}:OUTP ON")
        
    def turn_off(self, ch):
        self.inst.write(f"C{ch}:OUTP OFF")

    def mbgbs_lock_meausure(self, status:bool):
        self.inst.write("C2:OUTP ON")
        if status:
            self.set_square_wave(
                chn = 2,
                freq = 125,  # 125 MHz
                high = 5,
                low = 0.0,
                duty = 62.5,
                phase = 0.0
            )
            self.activate_burst_mode(2)
            self.set_sine_wave(chn=1, freq=80_000_000, amp=1800, offset=0.0)
            self.activate_burst_mode(1)
        if not status:
            self.set_DC_wave(2,3.0)
            self.inst.write(f"C1:BTWV STATE,OFF")
            self.inst.write(f"C2:BTWV STATE,OFF")
            self.inst.write("C2:OUTP ON")

    def sidebands_off_on(self, status):
        if status:
            self.inst.write(f"C1:OUTP ON")
            self.inst.write(f"C2:OUTP ON")
            self.set_sine_wave(chn=2, freq=75_000_000, amp=2, offset=0)
        if not status:   
            self.inst.write(f"C1:OUTP OFF")
            self.inst.write(f"C2:OUTP OFF")
            
    def disp_cal(self, voltage):
        self.set_square_wave(
                chn = 2,
                freq = 125,  # 125 MHz
                high = 3,
                low = voltage,
                duty = 62.5,
                phase = 0.0
            )
        self.activate_burst_mode(2)

if __name__ == "__main__":
    fg = RSDG(visa_address = find_funcgen("SDG2XCBX6R1764"))
    fg.set_square_wave(chn = 2, period = 0.02, duty = 75, high = 3.5, low = 0)
    fg.set_output(chn = 2, enable = True)
    time.sleep(1)
    fg.inst.write('C2:BSWV HLEV,2.5')