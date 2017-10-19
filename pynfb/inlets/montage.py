import mne
import pandas as pd


class Montage(pd.DataFrame):
    CHANNEL_TYPES = ['EEG', 'MAG', 'GRAD', 'OTHER']

    def __init__(self, names):
        super(Montage, self).__init__(columns=['name', 'type', 'pos_x', 'pos_y'])
        layout_eeg = Montage.load_layout('EEG1005')
        layout_mag = Montage.load_layout('Vectorview-mag')
        layout_grad = Montage.load_layout('Vectorview-grad')
        for name in names:
            if name.upper() in layout_eeg.names:
                ch_ind = layout_eeg.names.index(name.upper())
                self._add_channel(name, 'EEG', layout_eeg.pos[ch_ind][:2])
            elif name.upper() in layout_mag.names:
                ch_ind = layout_mag.names.index(name.upper())
                self._add_channel(name, 'MAG', layout_mag.pos[ch_ind][:2])
            elif name.upper() in layout_grad.names:
                ch_ind = layout_grad.names.index(name.upper())
                self._add_channel(name, 'GRAD', layout_grad.pos[ch_ind][:2])
            else:
                self._add_channel(name, 'OTHER', (None, None))

    def _add_channel(self, name, type, pos):
        self.loc[len(self)] = {'name': name, 'type': type, 'pos_x': pos[0], 'pos_y': pos[1]}

    def get_names(self, type='ALL'):
        if type in self.CHANNEL_TYPES:
            return list(self[self['type'] == type]['name'])
        elif type == 'ALL':
            return list(self['name'])
        else:
            raise TypeError('Bad channels type')

    def get_pos(self, type='ALL'):
        if type in self.CHANNEL_TYPES:
            return (self[self['type']==type][['pos_x', 'pos_y']]).as_matrix()
        elif type == 'ALL':
            return self[['pos_x', 'pos_y']].as_matrix()
        else:
            raise TypeError('Bad channels type')

    @staticmethod
    def load_layout(name):
        layout = mne.channels.read_layout(name)
        layout.names = list(map(str.upper, layout.names))
        return layout

if __name__ == '__main__':
    m = Montage(['cz', 'fp1', 'FP2', 'AUX1', 'MEG 2631', 'MEg 2632'])

    print(m.get_names())
    print(m.get_pos())