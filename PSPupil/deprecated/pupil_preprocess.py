import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pickle
from glob import glob
from os.path import join
from decim.adjuvant import slurm_submit as slu
from datetime import datetime


class PupilFrame(object):

    def __init__(self, subject, group, session, run,
                 directory='/Volumes/XKCD/PSP'):
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
        self.parameters = {}

    def load_pupil(self):
        '''
        Load pupil data and discard irrelevant data.
        '''
        path = glob(join(self.directory, 'raw_data', self.subject,
                         '20*_PSP_{0}_{1}'.format(self.subject,
                                                  self.session),
                         self.run, 'pupil_data'))

        with open(path[0], 'rb') as f:
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

    def interpol(self, array, orig_pupil, margin1=20, margin2=80):
        '''
        Linear interpolation of blinks and artifacts.

        - Arguments:
            a) array (pupil with blinks set to NaN)
            b) orig_pupil (origninal pupil)
            c) margin1 used at start of interpolation
            d) margin2 at end of interpolation
        '''
        convolved = np.convolve(array, [0.5, 1], 'same')                        # Use convolution to detect start and endpoint of interpolation
        ev_start = np.where(convolved == .5)[0]
        ev_end = np.where(convolved == 1)[0]
        if convolved[len(convolved) - 1] > 0:
            ev_end = np.append(ev_end, len(array) - 1)
        pupil_interpolated = np.array(orig_pupil.copy())                        # Copy of original pupil to interpolate
        for b in range(len(ev_start)):
            if ev_start[b] < margin1:
                start = 0
            else:
                start = ev_start[b] - margin1 + 1
            if ev_end[b] + margin2 + 1 > len(array) - 1:
                end = len(array) - 1
            else:
                end = ev_end[b] + margin2 + 1
            interpolated_signal = np.linspace(pupil_interpolated[start],        # Inteprolate
                                              pupil_interpolated[end],
                                              end - start,
                                              endpoint=False)
            pupil_interpolated[start:end] = interpolated_signal
        return pupil_interpolated

    def blink_interpol(self, side, crit_frags=200,
                       lower_blink_cutoff=30, blink_margin1=20, blink_margin2=80,
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
        - blink_margin1/2: interpolation margin for blinks
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
                          margin1=blink_margin1,
                          margin2=blink_margin2)                               # interpolate detected blinks and small fragments linearly
        pupil['interpol'] = pupil_interpld
        pupil['diff'] = np.append(0,
                                  np.abs(np.diff(pupil.interpol.values)))
        array = pupil['diff'] > diff_cutoff
        pupil_interpld =\
            self.interpol(array=array,
                          orig_pupil=pupil['interpol'], margin1=minor_margin,
                          margin2=minor_margin)  # interpolate linearly
        pupil['interpol2'] = pupil_interpld
        self.pupil[side] = pupil
        self.parameters['lower_blink_cutoff'] = lower_blink_cutoff
        self.parameters['blink_margin1'] = blink_margin1
        self.parameters['blink_margin2'] = blink_margin2
        self.parameters['minor_margin'] = minor_margin
        self.parameters['diff_cutoff'] = diff_cutoff

    def adjust_params_manual(self, side, lower_blink_cutoff=30,
                             blink_margin1=20, blink_margin2=80,
                             minor_margin=3, diff_cutoff=5):
        '''
        Function allows in interactive environment to manually report
        - samples that have been false negatively not reported as artifacts/blinks
        - samples that have been false positively been reported as artifacts/blinks

        Returns pd.DataFrame with
            - new column 'man_deblink'
            - updated column 'blink', which now tags every interpolated sample with True
        '''
        self.blink_interpol(side, lower_blink_cutoff=lower_blink_cutoff,
                            blink_margin1=blink_margin1,
                            blink_margin2=blink_margin2,
                            minor_margin=minor_margin,
                            diff_cutoff=diff_cutoff)
        f, ax = plt.subplots(figsize=(250, 10))
        plt.plot(self.pupil[side].diameter.values + 5, color='grey',
                 alpha=.5)
        plt.plot(self.pupil[side].interpol.values, color='red',
                 alpha=.5)
        plt.plot(self.pupil[side].interpol2.values - 5, color='green',
                 alpha=.5)
        plt.show()
        self.parameters['lower_blink_cutoff'] = lower_blink_cutoff
        self.parameters['blink_margin1'] = blink_margin1
        self.parameters['blink_margin2'] = blink_margin2
        self.parameters['minor_margin'] = minor_margin
        self.parameters['diff_cutoff'] = diff_cutoff

    def cut_end(self, side, frames):
        '''
        Cutoff frames at the end if there are artifacts.
        '''
        self.pupil[side] = self.pupil[side].iloc[:-frames, :]

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

    def save(self):
        out_dir = join(self.directory, 'Pupil_Preprocessed_{}'.
                       format(datetime.now().strftime("%Y-%m-%d")))
        slu.mkdir_p(out_dir)
        for side in ['left', 'right']:
            self.pupil[side].\
                to_hdf(join(out_dir, 'Pupil_Preprocessed_SUB-{0}_{1}_{2}.hdf'.
                            format(self.subject,
                                   self.session,
                                   self.run)), key=side)


def execute(subject, group, session, run, directory='/Volumes/XKCD/PSP'):
    p = PupilFrame(subject, group, session, run, directory)
    try:
        p.load_pupil()
        for side in ['left', 'right']:
            p.adjust_params_manual(side=side)
            adjust = input('Adjust?')
            while adjust == 'y':
                lbc = float(input('lower_blink_cutoff?'))
                bm1 = int(input('blink_margin1?'))
                bm2 = int(input('blink_margin2?'))
                mm = int(input('minor_margin?'))
                dc = float(input('diff_cutoff?'))
                p.adjust_params_manual(side=side, lower_blink_cutoff=lbc,
                                       blink_margin1=bm1, blink_margin2=bm2,
                                       minor_margin=mm, diff_cutoff=dc)
                adjust = input('Adjust?')
            cut_end = input('Cut End?')
            while cut_end == 'y':
                fr = int(input('frames? '))
                p.cut_end(side, fr)
                plt.plot(p.pupil[side].interpol2)
                plt.show()
                cut_end = input('more? ')
            p.filter(side)
            plt.plot(p.pupil[side].bp_interp.values)
            plt.show()
            p.z_score(side)
            plt.plot(p.pupil[side].biz.values)
            plt.show()
            p.save()
    except IndexError:
        print('File Error for', subject, session, run)                  #glob finds no files

# execute('001', 'patient', 'Followup', '000')
