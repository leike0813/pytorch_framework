import os
import cv2
from pathlib import Path
import torch
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.pyplot import close as closefig
from typing import Optional
from datetime import datetime


class TS_Featuremap_Visualizer:
    colormap = cv2.COLORMAP_VIRIDIS
    data_dim = 4
    data_linewidth = 0.3
    data_alpha = 0.7
    data_color_list = [
        'xkcd:yellow',
        'xkcd:orangered',
        'xkcd:fuchsia',
        'xkcd:magenta',
    ]

    def __init__(self, basefld, figsize=(16 / 2.54, 10 / 2.54), dpi=300, block_size=24, colormap=None):
        self.basefld = Path(basefld)
        self.figsize = figsize
        self.dpi = dpi
        self.pixel_shape = (int(figsize[1] * dpi), int(figsize[0] * dpi))
        self.block_size = block_size
        self.run_id = None
        self.count = 0
        if colormap is not None:
            self.colormap = getattr(cv2, colormap, self.colormap)

    def __call__(self, features, orig_datas=None):
        assert isinstance(features[0], torch.Tensor)
        assert features[0].dim() == 2
        img_paths = []
        if orig_datas is not None:
            assert len(features) == len(orig_datas) # same batch size
        else:
            orig_datas = [None for _ in range(len(features))]
        for i in range(len(features)):
            feature_img = self.draw_feature(features[i])
            fig = self.draw_figure(feature_img, orig_datas[i])
            img_paths.append(self.save_figure(fig))

        return img_paths

    def initialize_run(self, run_id=None):
        if not run_id:
            run_id = datetime.now().strftime('%y%m%d%H%M%S')
        self.run_id = run_id
        self.count = 0

    def draw_feature(self, feature: torch.Tensor) -> np.ndarray:
        feature_numpy = feature.permute(1, 0).numpy()
        f_dim = feature_numpy.shape[0]
        f_length = feature_numpy.shape[1]
        f_max = feature_numpy.max()
        f_min = feature_numpy.min()

        aspect_ratio = 1.6 * f_dim / f_length

        img = (feature_numpy - f_min) / (f_max - f_min)
        img = (img * 255).astype(np.uint8)
        img = np.repeat(np.repeat(img, self.block_size, axis=0), int(self.block_size * aspect_ratio), axis=1)
        img = cv2.applyColorMap(img, self.colormap)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.pixel_shape[1], self.pixel_shape[0]))

        return img

    def draw_figure(self, feature_img: np.ndarray, orig_data: Optional[torch.Tensor] = None) -> matplotlib.figure.Figure:
        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])
        ax.set_xmargin(0)
        ax.set_ymargin(0)

        if orig_data is not None:
            orig_data_numpy = orig_data.permute(1, 0).numpy()
            extent = [0, orig_data.shape[0], 0, orig_data.max()]
            x = np.arange(orig_data_numpy.shape[1])
            for i in range(self.data_dim):
                ax.plot(x, orig_data_numpy[i], lw=self.data_linewidth, alpha=self.data_alpha, c=self.data_color_list[i])
        else:
            extent = [0, 1, 0, 1]

        ax.imshow(
            feature_img,
            extent=extent,
            origin='upper',
            aspect='auto',
            zorder=-10
        )

        return fig

    def save_figure(self, fig: matplotlib.figure.Figure):
        if not self.basefld.exists():
            os.makedirs(self.basefld)
        img_path = self.basefld / f'{f"{self.run_id}_" if self.run_id else ""}{self.count}.png'
        with open(img_path, 'wb') as f:
            fig.savefig(f)
        closefig(fig)
        self.count += 1
        return img_path