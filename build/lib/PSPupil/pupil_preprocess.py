import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pickle
from glob import glob
from os.path import join, expanduser


class PupilFrame(object):

    def __init__(self, subject, group, session, run, directory):
        '''
        Initialize
        --------------------------------------------------------------
        ARGUMENTS:
        - subject: e.g. '001'
        - group: 'patient'
        - session: 'Baseline' / 'Followup'
        - run: '000', etc.
        - directory: e.g. '/Volumes/XKCD/PSP'
        '''
        self.subject = subject
        self.group = group
        self.session = session
        self.run = run
        self.directory = directory
        self.pupil = {}

    def load_pupil(self):
        '''
        Load pupil data and discard irrelevant data.
        '''
        path = join(self.directory, self.subject,
                    '2018_02_08_PSP_{0}_{1}'.format(self.subject,
                                                    self.session),
                    self.run, 'pupil_data')
        with open(path, 'rb') as f:
            file = pickle.load(f, encoding="latin-1")
        df = pd.DataFrame(file['pupil_positions'])
        self.pupil['left'] = df.loc[df.id == 0].drop(['norm_pos',
                                                      'ellipse',
                                                      'method'], axis=1)
        self.pupil['right'] = df.loc[df.id == 1].drop(['norm_pos',
                                                       'ellipse',
                                                       'method'], axis=1)

    def small_fragments(self, array, threshhold=200):
        '''
        Detects leftover fragments smaller than threshhold.

        Sets those detected fragments to NaN to make linear interpolation cleaner.
        '''
        convolved = np.convolve(array, [0.5, 1], 'same')
        ev_start = np.where(convolved == .5)[0]
        fragment_ends = ev_start
        if convolved[0] != 0:
            fragment_ends = fragment_ends[1:len(fragment_ends)]
        if convolved[len(convolved) - 1] == 0:
            fragment_ends = np.append(fragment_ends, len(array))
        ev_end = np.where(convolved == 1)[0]
        if convolved[0] == 0:
            fragment_starts = np.append(0, ev_end)
        else:
            fragment_starts = ev_end
        if (convolved[-2] == 1.5) & (convolved[-1] == 1):
            fragment_starts = fragment_starts[0:-1]
        assert len(fragment_ends) == len(fragment_starts)
        fragment_length = fragment_ends - fragment_starts
        wh = np.where(fragment_length < threshhold)
        smallfrag_ends = fragment_ends[wh]
        smallfrag_starts = fragment_starts[wh]
        for start, end in zip(smallfrag_starts, smallfrag_ends):
            array[start:end + 1] = True
        return array

    def interpol(self, array, orig_pupil, margin=50):
        '''
        Linear interpolation of blinks and artifacts.

        - Arguments:
            a) array (pupil with blinks set to NaN)
            b) orig_pupil (origninal pupil)
            c) margin used at star/end of interpolation
        '''
        convolved = np.convolve(array, [0.5, 1], 'same')                        # Use convolution to detect start and endpoint of interpolation
        ev_start = np.where(convolved == .5)[0]
        ev_end = np.where(convolved == 1)[0]
        if convolved[len(convolved) - 1] > 0:
            ev_end = np.append(ev_end, len(self.pupil_frame) - 1)
        pupil_interpolated = np.array(orig_pupil.copy())                        # Copy of original pupil to interpolate
        for b in range(len(ev_start)):
            if ev_start[b] < margin:
                start = 0
            else:
                start = ev_start[b] - margin + 1
            if ev_end[b] + margin + 1 > len(self.pupil_frame) - 1:
                end = len(self.pupil_frame) - 1
            else:
                end = ev_end[b] + margin + 1
            interpolated_signal = np.linspace(pupil_interpolated[start],        # Inteprolate
                                              pupil_interpolated[end],
                                              end - start,
                                              endpoint=False)
            pupil_interpolated[start:end] = interpolated_signal
        return pupil_interpolated

    def blink_interpol(self, side, crit_frags=200,
                       lower_blink_cutoff=30, blink_margin=50,
                       minor_margin=5, diff_cutoff=5):
        '''
        - detect blinks (using absolute diameter cutoff)
        - minor artifacts (using diameter velocity)
        - do linear interpolation of blinks and artifacts
        - optimize parameters manually by plotting and repeating

        -------------------------------------------------------------------

        INPUT:
        - pupil_frame: pd.dataframe with pupil data ('diameter')

        ARGUMENTS:
        - side: 'left' / 'right'
        - lower_blink_cutoff: absolute diameter --> below blink
        - crit_frags: no. samples as lower cut-off after
                    blink detection
        - blink_margin: interpolation margin for blinks
        - diff_cutoff: velocity cut-off for minor artifacts
        - minor_margin: interpolation margin for artifacts

        -------------------------------------------------------------------

        OUTPUT:
        - pd.DataFrame (as input) with new columns that include the clean pupil diameter data

        '''
        pupil = self.pupil[side]
        array = pupil['diameter'] < lower_blink_cutoff                  # detects blinks by absolute pupil diameter threshold
        array = self.small_fragments(array, threshhold=crit_frags)           # moreover, sets small fragments to NaN
        pupil_interpld =\
            self.interpol(array=array,
                          orig_pupil=pupil['diameter'],
                          margin=blink_margin)                               # interpolate detected blinks and small fragments linearly
        pupil['interpol'] = pupil_interpld
        pupil['diff'] = np.append(0,
                                  np.abs(np.diff(pupil.interpol.values)))
        array = pupil['diff'] > diff_cutoff
        pupil_interpld =\
            self.interpol(array=array,
                          orig_pupil=pupil['interpol'], margin=minor_margin)  # interpolate linearly
        pupil['interpol2'] = pupil_interpld
        self.pupil[side] = pupil

    def adjust_params_manual(self, side, lower_blink_cutoff=30,
                             blink_margin=50, minor_margin=5,
                             diff_cutoff=5):
        '''
        Function allows in interactive environment to manually report
        - samples that have been false negatively not reported as artifacts/blinks
        - samples that have been false positively been reported as artifacts/blinks

        Returns pd.DataFrame with
            - new column 'man_deblink'
            - updated column 'blink', which now tags every interpolated sample with True
        '''
        self.blink_interpol(side, lower_blink_cutoff=lower_blink_cutoff,
                            blink_margin=blink_margin,
                            minor_margin=minor_margin,
                            diff_cutoff=diff_cutoff)
        f, ax = plt.subplots(figsize=(250, 10))
        plt.plot(self.pupil[side].diameter.values + 5, color='grey',
                 alpha=.5)
        plt.plot(self.pupil[side].interpol.values, color='red',
                 alpha=.5)
        plt.plot(self.pupil[side].interpol2.values - 5, color='green',
                 alpha=.5)

    def filter(self, side, highpass=.01, lowpass=6, sample_rate=125):
        '''
        Apply 3rd-order Butterworth bandpass filter.
        '''

        pupil_interp = self.pupil[side].interpol                        # High pass:
        hp_cof_sample = highpass / (sample_rate / 2)
        bhp, ahp = signal.butter(3, hp_cof_sample, btype='high')
        pupil_interp_hp = signal.filtfilt(bhp, ahp, pupil_interp)
        lp_cof_sample = lowpass / (sample_rate / 2)                     # low pass
        blp, alp = signal.butter(3, lp_cof_sample)
        pupil_interp_bp = signal.filtfilt(blp, alp,
                                          pupil_interp_hp)        # band pass

        self.pupil[side]['bp_interp'] = pupil_interp_bp

    def z_score(self, side):
        '''
        Normalize
        '''
        self.pupil[side]['biz'] = (self.pupil[side].bp_interp -
                                   self.pupil[side].bp_interp.mean()) /\
            self.pupil[side].bp_interp.std()
