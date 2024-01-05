from typing import Tuple, Dict, Any
from ipycanvas import Canvas, hold_canvas
from ipywidgets import widgets
import numpy as np


class FlowRenderer():

    def __init__(self,
                 org_image_sequence: np.ndarray,
                 u_image_sequence: np.ndarray,
                 v_image_sequence: np.ndarray) -> None:

        if len(set([org_image_sequence.shape, u_image_sequence.shape, v_image_sequence.shape])) > 1:
            raise ValueError("Input shapes must be the same!")

        self.org_image_sequence = org_image_sequence
        self.u_image_sequence = u_image_sequence
        self.v_image_sequence = v_image_sequence

        self.sequence_len, self.height, self.width = org_image_sequence.shape

        self.canvas_img = Canvas(width=self.width, height=self.height)
        self.canvas_flow = Canvas(width=self.width, height=self.height)
        self.canvas_img.stroke_style = "red"

        self.play_widget = widgets.Play(
            value=0,
            min=0,
            max=self.sequence_len-1,
            step=1,
            description="Play",
            disabled=False
        )
        self.slider_widget = widgets.IntSlider(value=0,
                                               min=0,
                                               max=self.sequence_len-1,
                                               step=1,)
        widgets.jslink((self.play_widget, 'value'), (self.slider_widget, 'value'))

        self.slider_widget.observe(self.callback, "value")
        self.callback(dict(new=0))

    def callback(self, value: Dict[str, Any]) -> None:
        step = value["new"]
        with hold_canvas():
            uv_norm = np.sqrt(self.u_image_sequence[step]**2 + self.v_image_sequence[step]**2)
            max_norm = np.maximum(uv_norm.max(), 1e-5)
            uv_norm = (uv_norm / max_norm * 255).astype(np.uint8)
            self.canvas_flow.put_image_data(
                np.repeat(np.expand_dims(uv_norm, -1), 3, axis=-1), 0, 0)

            x0s, y0s = np.meshgrid(np.arange(0, self.width, 10), np.arange(0, self.height, 10))
            u_s, v_s = self.u_image_sequence[step][y0s, x0s], self.v_image_sequence[step][y0s, x0s]
            self.canvas_img.put_image_data(np.repeat(np.expand_dims(
                self.org_image_sequence[step], -1), 3, axis=-1), 0, 0)
            vector_indices = np.argwhere(u_s**2 + v_s**2 > (max_norm / 10))
            for y_index, x_index in vector_indices:
                self.canvas_img.stroke_line(
                    int(x0s[y_index, x_index]),
                    int(y0s[y_index, x_index]),
                    int(x0s[y_index, x_index] + v_s[y_index, x_index] / max_norm * 25),
                    int(y0s[y_index, x_index] + u_s[y_index, x_index] / max_norm * 25))

    def make_box(self, canvas: Canvas, label: str) -> widgets.VBox:
        return widgets.VBox([
            widgets.HTML(f"<h3>{label}</h3>"),
            widgets.Box([
                canvas,
            ], layout=widgets.Layout(
                width=f"{self.width}px",
            ))
        ], layout=widgets.Layout(
            width=f"{self.width + 8}px",
            margin="4px"
        ))

    def __call__(self) -> widgets.VBox:
        return widgets.VBox([
            widgets.HBox([self.play_widget, self.slider_widget]),
            self.make_box(self.canvas_img, "original"),
            self.make_box(self.canvas_flow, "flow norm"),
        ])