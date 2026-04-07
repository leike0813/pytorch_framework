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


class TS_Mask_Visualizer:
    data_dim = 4
    data_linewidth = 0.5
    data_alpha = 1.0
    data_color_list = [
        '#0C5DA5',
        '#00B945',
        '#FF9500',
        '#FF2C00',
    ]

    def __init__(self, basefld, figsize=(8 / 2.54, 5 / 2.54), dpi=300):
        self.basefld = Path(basefld)
        self.figsize = figsize
        self.dpi = dpi
        self.pixel_shape = (int(figsize[1] * dpi), int(figsize[0] * dpi))
        self.run_id = None
        self.count = 0

    def __call__(self, batch, batch_output, padding_mask=None):
        assert batch[0].dim() == batch_output[0].dim() == 2
        assert len(batch) == len(batch_output) == len(padding_mask)# same batch size
        img_paths = []
        for i in range(len(batch)):
            start_index, end_index, mask = self.process_mask(batch_output[i])
            if padding_mask is not None:
                act_length = (~padding_mask[i]).sum().item()
                orig_data = batch[i, :act_length, :]
                mask = mask[:act_length]
            else:
                orig_data = batch[i]
            fig = self.draw_figure(orig_data, start_index, end_index, mask)
            img_paths.append(self.save_figure(fig))

        return img_paths

    def initialize_run(self, run_id=None):
        if not run_id:
            run_id = datetime.now().strftime('%y%m%d%H%M%S')
        self.run_id = run_id
        self.count = 0

    def process_mask(self, output_logits: torch.Tensor) -> tuple:
        start_index = torch.argmax(output_logits[:, 0]).item()
        end_index = torch.argmax(output_logits[:, 1]).item()
        mask = output_logits[:, 2] > 0

        return start_index, end_index, mask

    def draw_figure(self, orig_data: torch.Tensor, start_index, end_index, mask) -> matplotlib.figure.Figure:
        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])
        ax.set_xmargin(0)
        ax.set_ymargin(0)

        orig_data_numpy = orig_data.permute(1, 0).numpy()
        x = np.arange(orig_data_numpy.shape[1])
        for i in range(self.data_dim):
            ax.plot(x, orig_data_numpy[i], lw=self.data_linewidth, alpha=self.data_alpha, c=self.data_color_list[i])

        # 绘制start_index和end_index对应位置的竖线
        ax.axvline(x=start_index, color='xkcd:purple', linewidth=1.5, alpha=0.8)  # 蓝色竖线表示起点
        ax.axvline(x=end_index, color='xkcd:olive', linewidth=1.5, alpha=0.8)    # 红色竖线表示终点

        # 在mask为True的部位绘制半透明灰色前景
        mask_numpy = mask.numpy()
        for i in range(len(mask_numpy)):
            if mask_numpy[i]:
                ax.axvspan(i-0.5, i+0.5, color='gray', alpha=0.2, zorder=-10)

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