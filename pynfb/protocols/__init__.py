from copy import deepcopy
import pickle as pkl
import numpy as np
from numpy import vstack
from numpy.random import randint

from ..helpers.beep import SingleBeep
from ..io.hdf5 import load_h5py_protocols_raw
from ..protocols.user_inputs import SelectSSDFilterWidget
from ..protocols.widgets import (CircleFeedbackProtocolWidgetPainter, BarFeedbackProtocolWidgetPainter,
                                     BaselineProtocolWidgetPainter,
                                     ThresholdBlinkFeedbackProtocolWidgetPainter, VideoProtocolWidgetPainter, TrialsProtocolWidgetPainter, CenterOutProtocolWidgetPainter)

from ..signals import CompositeSignal, DerivedSignal, BCISignal

from ..widgets.helpers import ch_names_to_2d_pos
from ..widgets.update_signals_dialog import SignalsSSDManager
import time
import random

class Protocol:
    def __init__(self, signals, source_signal_id=None, name='', duration=30, update_statistics_in_the_end=False,
                 mock_samples_path=(None, None), show_reward=False, reward_signal_id=0, reward_threshold=0.,
                 ssd_in_the_end=False, timer=None, freq=500, mock_previous=0, drop_outliers=0, stats_type='meanstd',
                 experiment=None, pause_after=False, reverse_mock_previous=False, m_signal_index=None,
                 shuffle_mock_previous=None, beep_after=False, as_mock=False, auto_bci_fit=False, montage=None,
                 random_over_time=0):
        """ Constructor
        :param signals: derived signals
        :param source_signal_id: base signal id, or None if 'All' signals using
        :param name: name of protocol
        :param duration: duration of protocol
        :param update_statistics_in_the_end: if true update mean and std scaling parameters of signals
        """
        self.show_reward = show_reward
        self.reward_signal_id = reward_signal_id
        self.reward_threshold = reward_threshold
        self.update_statistics_in_the_end = update_statistics_in_the_end
        self.stats_type = stats_type
        self.mock_samples_file_path, self.mock_samples_protocol = mock_samples_path
        self.name = name
        self.duration = duration
        self.random_over_time = random_over_time
        self.widget_painter = None
        self.signals = signals
        self.source_signal_id = source_signal_id
        self.ssd_in_the_end = ssd_in_the_end
        self.timer = timer
        self.freq = freq
        self.montage = montage
        self.mock_previous = mock_previous
        self.reverse_mock_previous = reverse_mock_previous
        self.drop_outliers = drop_outliers
        self.experiment = experiment
        self.pause_after = pause_after
        self.m_signal_id = m_signal_index
        self.shuffle_mock_previous = shuffle_mock_previous
        self.beep_after = beep_after
        self.as_mock = as_mock
        self.istrials = 0
        self.iscenterout = 0
        self.auto_bci_fit = auto_bci_fit

        pass

    def update_state(self, samples, reward, chunk_size=1, is_half_time=False, samples_counter=None):

        m_sample = None if self.m_signal_id is None else samples[self.m_signal_id]
        if self.source_signal_id is not None:
            if self.mock_previous == 0:
                mark = self.widget_painter.redraw_state(samples[self.source_signal_id], m_sample)
                reward.update(samples[reward.signal_ind], chunk_size)
            else:
                mock_chunk = self.mock_recordings[self.mock_samples_counter:self.mock_samples_counter + chunk_size]
                for signal in self.mock:
                    signal.update(mock_chunk)
                self.mock_samples_counter += chunk_size
                self.mock_samples_counter %= self.mock_recordings.shape[0]
                #mock_signals = self.mock_recordings_signals[self.mock_samples_counter - 1]
                #mark = self.widget_painter.redraw_state(mock_signals[self.source_signal_id], m_sample)
                #reward.update(mock_signals[reward.signal_ind], chunk_size)
                mark = self.widget_painter.redraw_state(self.mock[self.source_signal_id].current_chunk[-1], m_sample)
                reward.update(self.mock[reward.signal_ind].current_chunk[-1], chunk_size)
        else:
            mark = self.widget_painter.redraw_state(samples[0], m_sample)  # if source signal is 'ALL'
        return mark

    def update_statistics(self):
        pass

    def prepare_raw_mock_if_necessary(self, mock_raw, random_previous_fb_protocol_number, mock_signals):
        if self.shuffle_mock_previous:
            self.mock_previous = random_previous_fb_protocol_number
        if self.mock_previous:
            if self.source_signal_id is None:
                raise ValueError('If mock_previous is True, source signal should be single')
            self.mock_samples_counter = 0
            self.mock = deepcopy(self.signals)
            for signal in self.mock:
                if isinstance(signal, CompositeSignal):
                    signal.signals = [self.mock[j] for j in range(len(signal.signals))]
            rand_start_ind = randint(0, mock_raw.shape[0])
            self.mock_recordings = vstack((mock_raw[rand_start_ind:], mock_raw[:rand_start_ind]))
            self.mock_recordings_signals = vstack((mock_signals[rand_start_ind:], mock_signals[:rand_start_ind]))
            if self.reverse_mock_previous:
                self.mock_recordings = self.mock_recordings[::-1]
                self.mock_recordings_signals = self.mock_recordings_signals[::-1]

    def close_protocol(self, raw=None, signals=None, protocols=list(), protocols_seq=None, raw_file=None, marks=None):
        # action if ssd in the end checkbox was checked
        if self.beep_after:
            SingleBeep().try_to_play()

        if self.ssd_in_the_end or self.auto_bci_fit:

            # stop main timer
            if self.timer:
                self.timer.stop()

            # get recorded raw data
            if raw_file is not None and protocols_seq is not None:
                x = load_h5py_protocols_raw(raw_file, [j for j in range(len(protocols_seq)-1)])
                x.append(raw)
            else:
                raise AttributeError('Attributes protocol_seq and raw_file should be not a None')

        # automatic fit bci (protocol names should be in the bci_labels dictionary keys below)
        if self.auto_bci_fit:
            # prepare train data:
            bci_labels = {'Open': 0, 'Left': 1, 'Right': 2}
            X = np.vstack([x for x, name in zip(x, protocols_seq) if name in bci_labels])
            y = np.concatenate([np.ones(len(x), dtype=int) *  bci_labels[name]
                                for x, name in zip(x, protocols_seq) if name in bci_labels], 0)
            # find and fit first bci signal:
            bci_signal = [signal for signal in self.signals if isinstance(signal, BCISignal)][0]
            bci_signal.fit_model(X, y)

        if self.ssd_in_the_end:
            signal_manager = SignalsSSDManager(self.signals, x, self.montage, self, signals, protocols,
                                               sampling_freq=self.freq, protocol_seq=protocols_seq, marks=marks)
            signal_manager.test_signal.connect(lambda: self.experiment.start_test_protocol(
                protocols[signal_manager.combo_protocols.currentIndex()]
            ))
            signal_manager.test_closed_signal.connect(self.experiment.close_test_protocol)
            signal_manager.exec_()

        if self.ssd_in_the_end or self.auto_bci_fit:
            # run main timer
            if self.timer:
                self.timer.start(1000 * 1. / self.freq)

        self.update_mean_std(raw, signals)

        if self.pause_after:
            self.experiment.handle_channels_trouble_pause()

    def update_mean_std(self, raw, signals, must=False):
        # update statistics action
        if self.update_statistics_in_the_end or must:

            # firstly update DerivedSignals and collect updated signals data
            updated_derived_signals_recorder = []
            for s, signal in enumerate([signal for signal in self.signals if isinstance(signal, DerivedSignal)]):
                updated_derived_signals_recorder.append(
                    signal.update_statistics(raw=raw, emulate=self.ssd_in_the_end, signals_recorder=signals,
                                             stats_type=self.stats_type))
            updated_derived_signals_recorder = np.array(updated_derived_signals_recorder).T

            # secondly update CompositeSignals
            for signal in [signal for signal in self.signals if isinstance(signal, CompositeSignal)]:
                signal.update_statistics(updated_derived_signals_recorder, stats_type=self.stats_type)


class BaselineProtocol(Protocol):
    def __init__(self, signals, name='Baseline', update_statistics_in_the_end=True, text='Relax', half_time_text=None,
                 **kwargs):
        kwargs['name'] = name
        kwargs['update_statistics_in_the_end'] = update_statistics_in_the_end
        super().__init__(signals, **kwargs)
        self.text = text
        self.widget_painter = BaselineProtocolWidgetPainter(text=text, show_reward=self.show_reward)
        self.half_time_text_change = half_time_text is not None
        self.half_time_text = half_time_text
        self.is_half_time = False
        self.beep = SingleBeep()
        pass

    def update_state(self, samples, reward, chunk_size=1, is_half_time=False, samples_counter=None):
        if self.half_time_text_change:
            if is_half_time and not self.is_half_time:
                self.beep.try_to_play()
                self.is_half_time = True
                self.widget_painter.set_message(self.half_time_text)

    def close_protocol(self, **kwargs):
        self.is_half_time = False
        self.beep = SingleBeep()
        self.widget_painter.set_message('')
        super(BaselineProtocol, self).close_protocol(**kwargs)
        self.widget_painter.set_message(self.text)
        
class TrialsProtocol(Protocol):
    def __init__(self, signals, name='Trials', update_statistics_in_the_end=True, pic1_path='', #pic2_path, pic3_path, pic4_path,
                 **kwargs):
        kwargs['name'] = name
        kwargs['update_statistics_in_the_end'] = update_statistics_in_the_end
        super().__init__(signals, **kwargs)
        self.widget_painter = TrialsProtocolWidgetPainter()
        self.pic1_path = pic1_path
        self.is_half_time = False
        self.beep = SingleBeep()

        self.elapsed = 0
        
        self.is_first_update = True
        self.cur_state = 100
        self.istrials = 1
        
        # Construct events sequence with corresponding times
        
        before_red_arrow_t = 1
        red_arrow_on_t = 2
        after_red_arrow_t = 1
        
        all_events_seq = np.array([],dtype=np.int)
        all_events_times= np.array([],dtype=np.int)
        
        # start after 5 seconds
        cur_ev_time = 5
        
        for j in np.arange(1,6):
            for i in np.arange(1,5):
                
                dir_epoch_start_events = np.array([0, i+4, 0],dtype=np.int);
                dir_epoch_start_event_times = cur_ev_time + np.array([0, before_red_arrow_t, before_red_arrow_t+red_arrow_on_t]);
                
                all_events_seq = np.concatenate((all_events_seq, dir_epoch_start_events))
                all_events_times = np.concatenate((all_events_times, dir_epoch_start_event_times))
                
                cur_ev_time = all_events_times[-1] + after_red_arrow_t
                
                #[seq,times] = self.construct_dir_epoch(10,4,500,1000) #[np.zeros(40),np.zeros(40)]#
                [seq,times] = self.construct_dir_epoch_2(4,1000,0,i,8)
                
                times = times + cur_ev_time
                
                all_events_seq = np.concatenate((all_events_seq, seq))
                all_events_times = np.concatenate((all_events_times, times))
                
                cur_ev_time = all_events_times[-1] + before_red_arrow_t
            
            
        self.pos_in_events_times = 0
        self.events_seq = all_events_seq
        self.events_times = all_events_times
        with open("events_seq.pkl", "wb") as f:
            pkl.dump(all_events_seq, f)
        with open("all_events_times.pkl", "wb") as f:
            pkl.dump(all_events_times, f)

    def update_state(self, samples, reward, chunk_size=1, is_half_time=False, samples_counter=None):
        #if(samples_counter is not None):
        
        if(self.is_first_update):
            self.is_first_update = False
            self.protocol_start_time=time.time()
            self.elapsed = 0
        
        self.elapsed = time.time() - self.protocol_start_time

        self.check_times()
        
        return None, self.cur_state
                

    def check_times(self):
        if(self.pos_in_events_times < len(self.events_times)):

            if(self.elapsed > self.events_times[self.pos_in_events_times]):
                self.pos_in_events_times = self.pos_in_events_times + 1;
                
                if(self.pos_in_events_times < len(self.events_times)):
                    #self.widget_painter.img.setImage(self.widget_painter.image_0)
                    self.cur_state = self.events_seq[self.pos_in_events_times]

                    self.widget_painter.change_pic(self.cur_state)
                    #self.widget_painter.img.rotate(90)
                    #self.widget_painter.set_message(str(self.events_times[self.pos_in_events_times]) + ' seconds passed')
                    self.check_times()
#            if(self.pos_in_times > 0):
#                if(self.elapsed > self.times[self.pos_in_times-1]) + 0.2:
#                    self.widget_painter.img.setImage(self.widget_painter.up)
        
        
    def close_protocol(self, **kwargs):
        self.is_half_time = False
        self.beep = SingleBeep()
        self.widget_painter.set_message('')
        super(TrialsProtocol, self).close_protocol(**kwargs)
        #self.widget_painter.set_message(self.text)
        
    def construct_dir_epoch(self,num_trials_in_dir_epoch,num_states,stim_on_t,stim_off_t):
        num_arrow_flashes = num_states*num_trials_in_dir_epoch
        fullseq = np.zeros([num_arrow_flashes])
        
        for i in np.arange(num_trials_in_dir_epoch):
            
            this_trial = np.random.permutation(np.array([1, 2, 3, 4])).astype(int)
            #this_trial = np.array([1, 2, 3, 4]).astype(int)
            
            
            if(i>0):
                last_num_prev_trial = fullseq[(4*i - 1)]
                first_num_this_trial = this_trial[0]
                
                if(last_num_prev_trial == first_num_this_trial):
                    pos_to_switch_with = random.randint(1,3)
        
                    new_first = this_trial[pos_to_switch_with]
                    
                    this_trial[pos_to_switch_with] = this_trial[0] 
                    this_trial[0] = new_first

            fullseq[4*i:(4*i + 4)] = this_trial
        
        num_events = num_arrow_flashes*2
        event_seq_dir_epoch = np.zeros([num_events],dtype=np.int)
        
        event_times_dir_epoch = np.zeros([num_events])
        event_times_dir_epoch = np.arange(0,num_events*stim_on_t,stim_on_t)/float(1000)
        
        event_seq_dir_epoch[::2] = fullseq
        
        return event_seq_dir_epoch, event_times_dir_epoch
    
    def construct_dir_epoch_2(self,num_states,stim_on_t,stim_off_t,target,rarity):
        num_target_flashes = 5
        num_total_flashes_0 = rarity*num_target_flashes
        num_nontarget_flashes = (num_total_flashes_0 - num_target_flashes)//(num_states - 1)
        #num_total_flashes = num_nontarget_flashes*(num_states-1) + num_target_flashes
        
        seq = []
        
        for i in np.arange(1,num_states+1):
            if(i == target):
                seq = seq + [i]*num_target_flashes
            else:
                seq = seq + [i]*num_nontarget_flashes
                
        seq = np.random.permutation(np.array(seq)).astype(int)
        
        num_events = seq.shape[0]*2
        event_seq_dir_epoch = np.zeros([num_events],dtype=np.int)
        
        event_times_dir_epoch = np.zeros([num_events])
        event_times_dir_epoch = np.arange(0,num_events*stim_on_t,stim_on_t)/float(1000)
        
        event_seq_dir_epoch[::2] = seq
        
        return event_seq_dir_epoch, event_times_dir_epoch

class CenterOutProtocol(Protocol):
    def __init__(self, signals, name='CenterOut', update_statistics_in_the_end=True, pic1_path='', #pic2_path, pic3_path, pic4_path,
                 **kwargs):
        kwargs['name'] = name
        kwargs['update_statistics_in_the_end'] = update_statistics_in_the_end
        super().__init__(signals, **kwargs)
        self.widget_painter = CenterOutProtocolWidgetPainter()
        self.is_half_time = False
        self.beep = SingleBeep()

        
        
        self.is_first_update = True
        self.iscenterout = 1
        
        self.cur_state = 0
        self.elapsed = 0
        self.cur_par=0
        self.startHover=-1
        self.hoverEnough=0
        self.hoverCircle=-1
        evnts=[]
        tmp=[0,0,0]
        
        # Construct events sequence with corresponding times
        
#        (start)
#        - 2s -        
#        center on, outer passive 0 wait
#        - 2s -
#        center on, outer show (random) 1 showStart
#        - 1s -
#        center on, outer passive 0 wait
#        - 1s -
#        center angle (random), outer passive 2 showSpan
#        - 2s -
#        center dissapear, outer passive 3 getResponse
#        - 4s -
#        --

#           timings, types, time to wait on the guessed circle and starting pause (better no less then 4 secs) 
        timings=[2,1,1,1,3]
        #timings=[0.1,0.1,0.1,1,0.1]
        types=[0,1,0,2,3]
        self.onCircle=1
        start_pause=5
             
        for i in range(0,5*50,5):
            for j in range(5):
                if i==0 and j==0:
                    tmp[0]=0
                    tmp[1]=start_pause
                    tmp[2]=0
                else:
                    tmp[0]=types[j]
                    tmp[1]=evnts[i+j-1][1]+timings[j]
                    if types[j]==1:
                        tmp[2]=random.randint(0,7)
                    elif types[j]==2:
                        tmp[2]=random.randint(-7,7)
                    else:
                        tmp[2]=0
                evnts.append([tmp[k] for k in range(len(tmp))])
        #print(evnts)
        
        self.evnts=evnts    
        self.pos_in_events_times = 0

    def update_state(self, samples, reward, chunk_size=1, is_half_time=False, samples_counter=None):
        #if(samples_counter is not None):
        
        if(self.is_first_update):
            self.is_first_update = False
            self.protocol_start_time=time.time()
            self.elapsed = 0
            self.pos_in_events_times=0
            self.widget_painter.goFullScreen()
        
        self.elapsed = time.time() - self.protocol_start_time

        self.check_times()
        
        [self.posx, self.posy]=self.widget_painter.getMousePos()
        
        if self.cur_state==3:
            dat= self.widget_painter.checkHover(self.posx, self.posy)
            if self.hoverEnough==1:
                if dat[1]==0 or self.hoverCircle!=dat[0]:
                    self.hoverEnough=-1
                    self.startHover=-1
                else:
                    self.widget_painter.showCorrect(self.cur_par,self.hoverCircle,1)
                    
            elif self.hoverEnough==0:
                self.widget_painter.showCorrect(self.cur_par,self.hoverCircle,0)
                if dat[1]==1 and self.hoverCircle==dat[0]:
                    if self.startHover==-1:
                        self.startHover=self.elapsed
                    elif self.elapsed>=self.startHover+self.onCircle:
                        self.hoverEnough=1
                else:
                    self.startHover=-1
            self.hoverCircle=dat[0]                                       
                    
        return None, [self.cur_state, self.cur_par, self.posx, self.posy]
                

    def check_times(self):
        if(self.pos_in_events_times < len(self.evnts)):
            #print(self.elapsed)
            #print(self.evnts[self.pos_in_events_times][1])
            if(self.elapsed > self.evnts[self.pos_in_events_times][1]):
                self.pos_in_events_times = self.pos_in_events_times + 1;
                if(self.pos_in_events_times < len(self.evnts)):
                    #self.widget_painter.img.setImage(self.widget_painter.image_0)
                    self.cur_state = self.evnts[self.pos_in_events_times][0]
                    self.cur_par = self.evnts[self.pos_in_events_times][2]
                    if self.cur_state==1:
                        self.startSpan=self.cur_par
                    elif self.cur_state==2:
                        self.plusSpan=self.cur_par
                    elif self.cur_state==3:
                        self.cur_par=(8+self.plusSpan+self.startSpan)%8
                        self.startHover=-1
                        self.hoverEnough=0
                    self.widget_painter.doStuff(self.cur_state,self.cur_par)
                    self.widget_painter.prev_par=self.cur_par
                    self.widget_painter.prev_state=self.cur_state
                    self.check_times()

            
    def close_protocol(self, **kwargs):
        self.is_half_time = False
        self.beep = SingleBeep()
        self.widget_painter.set_message('')
        super(CenterOutProtocol, self).close_protocol(**kwargs)
        #self.widget_painter.set_message(self.text)

class FeedbackProtocol(Protocol):
    def __init__(self, signals, name='Feedback', circle_border=0, m_threshold=1, **kwargs):
        kwargs['name'] = name
        super().__init__(signals, **kwargs)
        if circle_border == 2:
            self.widget_painter = BarFeedbackProtocolWidgetPainter(show_reward=self.show_reward,
                                                                      circle_border=circle_border,
                                                                      m_threshold=m_threshold)
        else:
            self.widget_painter = CircleFeedbackProtocolWidgetPainter(show_reward=self.show_reward,
                                                                      circle_border=circle_border,
                                                                      m_threshold=m_threshold)
        pass


class ThresholdBlinkFeedbackProtocol(Protocol):
    def __init__(self, signals, name='ThresholdBlink', threshold=1000, time_ms=50, **kwargs):
        kwargs['name'] = name
        super().__init__(signals, **kwargs)
        self.widget_painter = ThresholdBlinkFeedbackProtocolWidgetPainter(threshold=threshold, time_ms=time_ms,
                                                                          show_reward=self.show_reward)


class VideoProtocol(Protocol):
    def __init__(self, signals, name='Video', video_path='', **kwargs):
        kwargs['name'] = name
        super().__init__(signals, **kwargs)
        self.widget_painter = VideoProtocolWidgetPainter(video_file_path=video_path)
        pass


class SSDProtocol(Protocol):
    def __init__(self, signals, text='Relax', **kwargs):
        kwargs['ssd_in_the_end'] = True
        super().__init__(signals, **kwargs)
        self.widget_painter = BaselineProtocolWidgetPainter(text=text, show_reward=self.show_reward)


def main():
    pass


if __name__ == '__main__':
    main()
