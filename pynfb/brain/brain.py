from collections import deque
from warnings import warn
import os

import mne
from mne.datasets import sample
import numpy as np
from matplotlib import cm
from pyqtgraph import opengl as gl
import nibabel as nib
from scipy import sparse
import matplotlib.colors as mpl_colors
from OpenGL.GL import (GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_FUNC_ADD, GL_MAX, GL_BLEND,
                       GL_ALPHA_TEST, GL_DEPTH_TEST,
                       GL_CULL_FACE, GL_FRONT, GL_FRONT_FACE, GL_BACK)

from ..protocols import Protocol
from ..protocols.widgets import Painter
from .settings import SourceSpaceWidgetPainterSettings as PainterSettings


class SourceSpaceRecontructor(Protocol):
    def __init__(self, signals, **kwargs):
        kwargs['ssd_in_the_end'] = True
        super().__init__(signals, **kwargs)
        inverse_operator = self.make_inverse_operator()
        self._forward_model_matrix = self._assemble_forward_model_matrix(inverse_operator)
        self.source_vertex_idx = self.extract_source_vertex_idx(inverse_operator)
        self.widget_painter = SourceSpaceWidgetPainter(self)

    @staticmethod
    def _assemble_forward_model_matrix(inverse_operator):
        warn('Currently info is read from the raw file. TODO: change to getting it from the stream')
        data_path = sample.data_path()
        fname_raw = data_path + '/MEG/sample/sample_audvis_raw.fif'
        raw = mne.io.read_raw_fif(fname_raw, verbose='ERROR')
        info = raw.info
        channel_cnt = info['nchan']
        I = np.identity(channel_cnt)
        dummy_raw = mne.io.RawArray(data=I, info=info, verbose='ERROR')
        dummy_raw.set_eeg_reference(verbose='ERROR')

        # Applying inverse modelling to an identity matrix gives us the forward model matrix
        snr = 1.0  # use smaller SNR for raw data
        lambda2 = 1.0 / snr ** 2
        method = "MNE"  # use sLORETA method (could also be MNE or dSPM)
        stc = mne.minimum_norm.apply_inverse_raw(dummy_raw, inverse_operator, lambda2, method)
        return stc.data

    @staticmethod
    def extract_source_vertex_idx(inverse_operator):
        left_hemi, right_hemi = inverse_operator['src']
        lh_vertex_cnt = left_hemi['rr'].shape[0]
        return np.r_[left_hemi['vertno'], lh_vertex_cnt + right_hemi['vertno']]

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


class RangeBuffer:
    def __init__(self, max_buffer_length):
        self.max_buffer = deque(maxlen=max_buffer_length)
        self.current_buffer_length = max_buffer_length
        self.vmax = None
        self.robust_max = SourceSpaceWidgetPainter.robust_max

    def update(self, sources, take_abs=True):
        if take_abs:
            sources = np.abs(sources)
        self.max_buffer.extend(self.robust_max(sources))
        self.vmax = max(self.max_buffer)

class SourceSpaceWidgetPainter(Painter):
    # Settings constants

    COLORMAP_BUFFER_LENGTH_MAX = 40000  # samples, if colormap limits are set to 'global' then 'global' means last
                                        # COLORMAP_BUFFER_LENGTH_DEFAULT samples
    ROBUST_PCT = 95 # Perecntage to use in robust_max

    def __init__(self, source_space_reconstructor, show_reward=False, params=None):
        super().__init__(show_reward=show_reward)

        self.settings = PainterSettings()
        self.connect_settings()

        self.protocol = source_space_reconstructor
        self.source_vertex_idx = self.protocol.source_vertex_idx
        self.chunk_to_sources = source_space_reconstructor.chunk_to_sources

        # Background grey brain
        self.cortex_mesh_data = None
        self.cortex_mesh_item = None
        self.brain_colormap = cm.Greys
        self.data_colormap = cm.seismic

        self.colormap_mode = self.settings.colormap.mode.value()
        self.lock_current_limits = self.settings.colormap.lock_current_limits.value()
        self.vmax = None

        self.colormap_buffer_length = self.COLORMAP_BUFFER_LENGTH_MAX
        self.colormap_threshold_pct = self.settings.colormap.threshold_pct.value()
        self.range_buffer = RangeBuffer(self.colormap_buffer_length)

        self.smoothing_matrix = self.read_smoothing_matrix()

    def prepare_widget(self, widget):
        super().prepare_widget(widget)

        # Background grey brain
        self.cortex_mesh_data = self.read_mesh()
        curvature = self.read_curvature()
        # Concave regions get the color 2/3 into the colormap and convex - 1/3
        self.background_colors = self.brain_colormap((curvature > 0) / 3 + 1/3)
        self.cortex_mesh_data.setVertexColors(self.background_colors)

        # Set the camera at twice the size of the mesh along the widest dimension
        max_ptp = max(np.ptp(self.cortex_mesh_data.vertexes(), axis=0))
        widget.setCameraPosition(distance=2*max_ptp)

        self.cortex_mesh_item = gl.GLMeshItem(meshdata=self.cortex_mesh_data, shader='shaded')
        widget.addItem(self.cortex_mesh_item)

        print('Widget prepared')

    def surfaces_dir(self):
        data_path = sample.data_path()
        return os.path.join(data_path, "subjects", "sample", "surf")

    def read_mesh(self, cortex_type='inflated'):
        surf_paths = [os.path.join(self.surfaces_dir(), '{}.{}'.format(h, cortex_type))
                      for h in ('lh', 'rh')]
        lh_mesh, rh_mesh = [nib.freesurfer.read_geometry(surf_path) for surf_path in surf_paths]
        lh_vertexes, lh_faces = lh_mesh
        rh_vertexes, rh_faces = rh_mesh

        # Move all the vertexes so that the lh has x (L-R) <= 0 and rh - >= 0
        lh_vertexes[:, 0] -= np.max(lh_vertexes[:, 0])
        rh_vertexes[:, 0] -= np.min(rh_vertexes[:, 0])

        # Combine two meshes
        vertexes = np.r_[lh_vertexes, rh_vertexes]
        lh_vertex_cnt = lh_vertexes.shape[0]
        faces = np.r_[lh_faces, lh_vertex_cnt + rh_faces]

        # Move the mesh so that the center of the brain is at (0, 0, 0) (kinda)
        vertexes[:, 1:2] -= np.mean(vertexes[:, 1:2])

        # Invert vertex normals for more reasonable lighting (I am not sure if the puqtgraph's shader has a bug or
        # gl.MeshData's calculation of normals does
        mesh_data = gl.MeshData(vertexes=vertexes, faces=faces)
        mesh_data._vertexNormals = mesh_data.vertexNormals() * (-1)

        return mesh_data

    def read_curvature(self):
        curv_paths = [os.path.join(self.surfaces_dir(),
                                   "{}.curv".format(h)) for h in ('lh', 'rh')]
        curvs = [nib.freesurfer.read_morph_data(curv_path) for curv_path in curv_paths]
        return np.concatenate(curvs)

    def redraw_state(self, chunk):
        sources = self.chunk_to_sources(chunk)
        last_sources = sources[-1, :]

        self.range_buffer.update(sources, take_abs=True)
        self.update_colormap_upper_limit(self.colormap_mode, last_sources, take_abs=True)
        colors = self.calculate_colors(last_sources, self.colormap_threshold_pct, take_abs=True)
        self.update_mesh_colors(colors)

    def update_colormap_upper_limit(self, colormap_mode, last_sources, take_abs=True):
        if self.colormap_mode == PainterSettings.COLORMAP_LIMITS_MANUAL:
            # In this case vmax is set by a slot connected to the changes from settings widget
            return

        elif colormap_mode == PainterSettings.COLORMAP_LIMITS_LOCAL:
            new_upper_limit = self.robust_max(np.abs(last_sources[np.newaxis, :]))[0]

        elif self.colormap_mode == PainterSettings.COLORMAP_LIMITS_GLOBAL and not self.lock_current_limits:
            new_upper_limit = self.range_buffer.vmax
        else:
            return

        # upper_limit.setValue() will result in sigValueChanged which will trigger the update of self.vmax
        self.settings.colormap.upper_limit.setValue(new_upper_limit)

    def update_mesh_colors(self, colors):
        self.cortex_mesh_data.setVertexColors(colors)
        self.cortex_mesh_item.meshDataChanged()

    def normalize_to_01(self, values):
        return values / self.vmax

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

    def upper_limit_changed(self, param, vmax):
        self.vmax = vmax

    def connect_settings(self):
        cmap_settings = self.settings.colormap
        cmap_settings.mode.sigValueChanged.connect(self.mode_changed)
        cmap_settings.upper_limit.sigValueChanged.connect(self.upper_limit_changed)

    @staticmethod
    def read_smoothing_matrix():
        lh_npz = np.load('playground/vs_pysurfer/smooth_mat_lh.npz')
        rh_npz = np.load('playground/vs_pysurfer/smooth_mat_rh.npz')

        smooth_mat_lh = sparse.coo_matrix((
            lh_npz['data'], (lh_npz['row'], lh_npz['col'])),
            shape=lh_npz['shape'] + rh_npz['shape'])

        lh_row_cnt, lh_col_cnt = lh_npz['shape']
        smooth_mat_rh = sparse.coo_matrix((
            rh_npz['data'], (rh_npz['row'] + lh_row_cnt, rh_npz['col'] + lh_col_cnt)),
            shape=rh_npz['shape'] + lh_npz['shape'])

        return smooth_mat_lh.tocsc() + smooth_mat_rh.tocsc()

    @classmethod
    def robust_max(cls, ndarray):
        pctl = np.percentile(ndarray, q=cls.ROBUST_PCT, axis=1, keepdims=True)
        return np.nanmean(
            np.where(ndarray >= pctl, ndarray, np.nan),
            axis=1)

    def calculate_colors(self, last_sources, threshold_pct, take_abs=True):
        sources_smoothed = self.smoothing_matrix.dot(last_sources)
        sources_normalized = self.normalize_to_01(np.abs(sources_smoothed))
        colors = self.data_colormap(sources_normalized)

        threshold = threshold_pct / 100

        invisible_mask = sources_normalized <= threshold
        colors[invisible_mask] = self.background_colors[invisible_mask]
        colors[~invisible_mask] *= self.background_colors[~invisible_mask, 0, np.newaxis]

        return colors