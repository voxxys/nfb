from collections import deque
from warnings import warn

import mne
import numpy as np
from matplotlib import cm
from pyqtgraph import opengl as gl

from ..protocols import Protocol
from ..protocols.widgets import Painter
from .settings import SourceSpaceWidgetPainterSettings as PainterSettings


class SourceSpaceRecontructor(Protocol):
    def __init__(self, signals, **kwargs):
        kwargs['ssd_in_the_end'] = True
        super().__init__(signals, **kwargs)
        inverse_operator = self.make_inverse_operator()
        self.mesh_data = self.get_mesh_data_from_inverse_operator(inverse_operator)
        self._forward_model_matrix = self._assemble_forward_model_matrix(inverse_operator)
        self.widget_painter = SourceSpaceWidgetPainter(self)

    @staticmethod
    def _assemble_forward_model_matrix(inverse_operator):
        from mne.datasets import sample

        warn('Currently info is read from the raw file. TODO: change to getting it from the stream')
        data_path = sample.data_path()
        fname_raw = data_path + '/MEG/sample/sample_audvis_raw.fif'
        raw = mne.io.read_raw_fif(fname_raw, verbose='ERROR')
        info = raw.info
        channel_cnt = info['nchan']
        I = np.identity(channel_cnt)
        dummy_raw = mne.io.RawArray(data=I, info=info, verbose='ERROR')
        dummy_raw.set_eeg_reference(verbose='ERROR');

        # Applying inverse modelling to an identity matrix gives us the forward model matrix
        snr = 1.0  # use smaller SNR for raw data
        lambda2 = 1.0 / snr ** 2
        method = "MNE"  # use sLORETA method (could also be MNE or dSPM)
        stc = mne.minimum_norm.apply_inverse_raw(dummy_raw, inverse_operator, lambda2, method)
        return stc.data

    def chunk_to_sources(self, chunk):
        F = self._forward_model_matrix
        return F.dot(chunk.T).T

    @staticmethod
    def make_inverse_operator():
        from mne.datasets import sample
        from mne.minimum_norm import read_inverse_operator
        data_path = sample.data_path()
        filename_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
        return read_inverse_operator(filename_inv, verbose='ERROR')

    def get_mesh_data_from_inverse_operator(self, inverse_operator):
        # Creates pyqtgraph.opengl.MeshData instance to store the cortical mesh used to create the inverse operator.

        # mne's inverse operator is a dict with the geometry information under the key 'src'.
        # inverse_operator['src'] is a list two items each of which corresponds to one hemisphere.
        left_hemi, right_hemi = inverse_operator['src']

        # Each hemisphere is represented by a dict that contains the list of all vertices from the original mesh
        # representaion of that representation (with default options in FreeSurfer that is ~150K vertices). These are
        # stored under the key 'rr'.

        # Only a small subset of these vertices was most likely during the construction of the inverse operator. The
        # mesh containing only the used vertices is represented by an array of faces sotred under the 'use_tris' key.
        # This submesh still contains some extra vertices so that it is still a manifold.

        # Each face is a row with the indices of the vertices of that face. The indexing is into the 'rr' array
        # containing all the vertices.

        # Let's now combine two meshes into one. Also save the indexes of the used vertices into vertex_idx property
        vertexes = np.r_[left_hemi['rr'], right_hemi['rr']]
        lh_vertex_cnt = left_hemi['rr'].shape[0]
        faces = np.r_[left_hemi['use_tris'], lh_vertex_cnt + right_hemi['use_tris']]
        self.vertex_idx = np.r_[left_hemi['vertno'], lh_vertex_cnt + right_hemi['vertno']]

        return gl.MeshData(vertexes=vertexes, faces=faces)

    def update_state(self, chunk):
        self.widget_painter.redraw_state(chunk)

    def close_protocol(self, **kwargs):
        self.widget_painter.close()
        super().close_protocol(**kwargs)


class SourceSpaceWidget(gl.GLViewWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward = None
        self.setMinimumSize(400, 400)
        self.clear_all()

    def clear_all(self):
        for item in self.items:
            self.removeItem(item)
        # self.addItem(self.reward)

    def update_reward(self, reward):
        pass

    def show_reward(self, flag):
        pass


class SourceSpaceWidgetPainter(Painter):
    # Settings constants

    COLORMAP_BUFFER_LENGTH_DEFAULT = 40000  # samples, if colormap limits are set to 'global' then 'global' means last
                                            # COLORMAP_BUFFER_LENGTH_DEFAULT samples

    class RangeBuffer:
        def __init__(self, buffer_length):
            self.min_buffer = deque(maxlen=buffer_length)
            self.max_buffer = deque(maxlen=buffer_length)
            self.vmin = None
            self.vmax = None

        def update(self, sources):
            self.min_buffer.extend(np.min(sources, axis=1))
            self.vmin = min(self.min_buffer)
            self.max_buffer.extend(np.max(sources, axis=1))
            self.vmax = max(self.max_buffer)

        def limits(self):
            return self.vmin, self.vmax

    def __init__(self, source_space_reconstructor, show_reward=False, params=None):
        super().__init__(show_reward=show_reward)

        self.settings = PainterSettings()
        self.connect_settings()

        self.protocol = source_space_reconstructor
        self.chunk_to_sources = source_space_reconstructor.chunk_to_sources

        self.cortex_mesh_data = None
        self.vertex_idx = None
        self.cortex_mesh_item = None

        self.brain_colormap = cm.Greys
        self.data_colormap = cm.viridis
        self.colormap_mode = self.settings.colormap.mode.value()
        self.lock_current_limits = self.settings.colormap.lock_current_limits.value()
        self.vmin, self.vmax = None, None

        self.colormap_buffer_length = self.COLORMAP_BUFFER_LENGTH_DEFAULT
        self.range_buffer = self.RangeBuffer(self.colormap_buffer_length)

    def prepare_widget(self, widget):
        super().prepare_widget(widget)

        self.cortex_mesh_data = self.protocol.mesh_data
        self.vertex_idx = self.protocol.vertex_idx

        # We will only be assigning colors to a subset of vertexes used for forward/inverse modelling. First, we need to
        # assign an initial color to all the vertices.
        total_vertex_cnt = self.cortex_mesh_data.vertexes().shape[0]
        brain_color = self.brain_colormap(0.5)
        brain_colors = np.tile(brain_color, (total_vertex_cnt, 1))
        self.cortex_mesh_data.setVertexColors(brain_colors)

        # Set the camera at twice the size of the mesh along the widest dimension
        max_ptp = max(np.ptp(self.cortex_mesh_data.vertexes(), axis=0))
        widget.setCameraPosition(distance=2*max_ptp)

        self.cortex_mesh_item = gl.GLMeshItem(meshdata=self.cortex_mesh_data, shader='shaded')
        widget.addItem(self.cortex_mesh_item)

        print('Widget prepared')

    def redraw_state(self, chunk):
        sources = self.chunk_to_sources(chunk)
        last_sources = sources[-1, :]
        self.range_buffer.update(sources)

        # settings.update_limits will result in sigValueChanged which will trigger update of self.vmin and self.vmax
        if self.colormap_mode == PainterSettings.COLORMAP_LIMITS_LOCAL:
            self.settings.update_limits(np.min(last_sources), np.max(last_sources))
        elif self.colormap_mode == PainterSettings.COLORMAP_LIMITS_GLOBAL and not self.lock_current_limits:
            self.settings.update_limits(*self.range_buffer.limits())
        elif self.colormap_mode == PainterSettings.COLORMAP_LIMITS_MANUAL:
            # In this case vmin, vmax are set by a slot connected to the changes from settings widget
            pass

        sources_normalized = self.normalize_to_01(last_sources)
        colors = self.colormap(sources_normalized)
        self.update_mesh_colors(colors)

    def update_mesh_colors(self, colors):
        # using cortex_mesh_data.setVertexColors() is much slower, bc we are coloring only a subset of vertices
        self.cortex_mesh_data._vertexColors[self.vertex_idx] = colors
        self.cortex_mesh_data._vertexColorsIndexedByFaces = None
        self.cortex_mesh_item.meshDataChanged()

    def normalize_to_01(self, values):
        return (values - self.vmin) / (self.vmax - self.vmin)

    def set_limits(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax

    def mode_changed(self, param, mode):
        self.colormap_mode = mode
        if mode == PainterSettings.COLORMAP_LIMITS_LOCAL:
            self.settings.colormap.lower_limit.setReadonly(True)
            self.settings.colormap.upper_limit.setReadonly(True)
            self.settings.colormap.lock_current_limits.setReadonly(True)
        elif mode == PainterSettings.COLORMAP_LIMITS_GLOBAL:
            self.settings.colormap.lower_limit.setReadonly(True)
            self.settings.colormap.upper_limit.setReadonly(True)
            self.settings.colormap.lock_current_limits.setReadonly(False)
        elif mode == PainterSettings.COLORMAP_LIMITS_MANUAL:
            self.settings.colormap.lower_limit.setReadonly(False)
            self.settings.colormap.upper_limit.setReadonly(False)
            self.settings.colormap.lock_current_limits.setReadonly(True)

    def lower_limit_changed(self, param, vmin):
        self.vmin = vmin

    def upper_limit_changed(self, param, vmax):
        self.vmax = vmax

    def connect_settings(self):
        cmap_settings = self.settings.colormap
        cmap_settings.mode.sigValueChanged.connect(self.mode_changed)
        cmap_settings.lower_limit.sigValueChanged.connect(self.lower_limit_changed)
        cmap_settings.upper_limit.sigValueChanged.connect(self.upper_limit_changed)