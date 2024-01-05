from typing import Tuple
from ipycanvas import Canvas
from ipywidgets import widgets
from PIL import Image
import numpy as np

from visualizers.utils import image_to_array


class CornerRenderer():
    """ Interactive visualization for corner detection

    Args:
        image_path (str): Image path of shape (H, W)
        kernel_size (Tuple[int, int]): Height (K_h) and width (K_w) of the kernel
        canvas_size (Tuple[int, int]): Width (Q_w) and height (Q_h) of the kernel
        x_derivative (np.ndarray): I_x, x derivative of the image array of shape (H, W)
        y_derivative (np.ndarray): I_y, y derivative of the image array of shape (H, W)
        eigen_values (np.ndarray): Eigen values of the gradient covariance matrix (H, W, 2)
        eigen_vectors (np.ndarray): Eigen vectors of the gradient covariance matrix (H, W, 2, 2)
        corner_indices (np.ndarray): Indices of the corner points (C, 2)
    """

    def __init__(self,
                 image_path: str,
                 kernel_size: Tuple[int, int],
                 canvas_size: Tuple[int, int],
                 x_derivative: np.ndarray,
                 y_derivative: np.ndarray,
                 eigen_values: np.ndarray,
                 eigen_vectors: np.ndarray,
                 corner_indices: np.ndarray,
                 ) -> None:

        for derivative in [x_derivative, y_derivative]:
            if (np.abs(derivative).max() > 255):
                raise ValueError("Derivative arrays must not exceed 255")

        if corner_indices.shape[-1] != 2:
            raise ValueError("Parameter corner_indices must be an array of shape [C, 2]")

        if eigen_values.ndim != 3:
            raise ValueError("Eigen value array must be 3D of shape [H, W, M]")

        if eigen_vectors.ndim != 4:
            raise ValueError("Eigen vectors array must be 4D of shape [H, W, M, M]")

        self.canvas_width, self.canvas_height = canvas_size
        self.kernel_height, self.kernel_width = kernel_size
        self.x_derivative = x_derivative.astype(np.float32)
        self.y_derivative = y_derivative.astype(np.float32)
        self.eigen_values = eigen_values
        self.eigen_vectors = eigen_vectors
        self.corner_indices = corner_indices

        image = Image.open(image_path)
        file_obj = open(image_path, "rb")
        self.img_widget = widgets.Image(value=file_obj.read())
        self.image_width, self.image_height = image.size
        self.image_array = image_to_array(image)

        self.canvas_plot = Canvas(width=self.canvas_width, height=self.canvas_height)
        self.canvas_focus = Canvas(width=self.kernel_width, height=self.kernel_height)
        self.canvas_fresh = Canvas(width=self.canvas_width, height=self.canvas_height)
        self.canvas_full = Canvas(width=self.canvas_width, height=self.canvas_height)
        self.canvas_corner = Canvas(width=self.canvas_width, height=self.canvas_height)

        self.prev_x_index = None
        self.prev_y_index = None
        self.prepare()

    def prepare(self):
        self.canvas_fresh.draw_image(
            self.img_widget, width=self.canvas_width, height=self.canvas_height)
        self.canvas_full.draw_image(self.canvas_fresh, 0, 0,
                                    width=self.canvas_width, height=self.canvas_height)
        self.canvas_corner.draw_image(self.canvas_fresh, 0, 0,
                                      width=self.canvas_width, height=self.canvas_height)
        self.canvas_corner.fill_style = "red"
        self.canvas_corner.fill_rects(
            self.corner_indices[:, 1] // (self.image_width / self.canvas_width),
            self.corner_indices[:, 0] // (self.image_height / self.canvas_height),
            width=4)
        self.canvas_full.stroke_style = "blue"
        self.canvas_full.on_mouse_move(self.handle_mouse_move)

    def _crop_array(self,
                    image_array: np.ndarray,
                    x_index: int,
                    y_index: int,
                    ) -> np.ndarray:
        k_half_height = self.kernel_height // 2
        k_half_width = self.kernel_width // 2
        return np.repeat(
            np.expand_dims(image_array[
                y_index-k_half_height-1:y_index+k_half_height,
                x_index-k_half_width-1:x_index+k_half_width
            ], axis=-1), 3, axis=-1)

    def plot_derivatives(self,
                         x_derivative: np.ndarray,
                         y_derivative: np.ndarray,
                         x_index: int,
                         y_index: int
                         ) -> None:
        self.canvas_plot.clear_rect(0, 0, width=self.canvas_width, height=self.canvas_height)
        self.canvas_plot.stroke_style = "red"
        self.canvas_plot.stroke_line(self.canvas_width // 2, 0,
                                     self.canvas_width // 2, self.canvas_height)
        self.canvas_plot.stroke_line(0, self.canvas_height // 2,
                                     self.canvas_width, self.canvas_height // 2)

        scaled_x_der_array = (x_derivative.reshape(-1) / 255) * \
            (self.canvas_width // 2) + (self.canvas_width // 2)
        scaled_y_der_array = (y_derivative.reshape(-1) / 255) * \
            (self.canvas_height // 2) + (self.canvas_height // 2)
        self.canvas_plot.fill_style = "#555555"
        self.canvas_plot.fill_rects(scaled_x_der_array, scaled_y_der_array, width=1)

        self.canvas_plot.stroke_style = "green"
        for index in range(2):
            eigen_vec = (self.eigen_vectors[y_index, x_index, :, index] *
                         np.sqrt(self.eigen_values)[y_index, x_index, index]).astype(np.int32)
            self.canvas_plot.stroke_line(
                self.canvas_width // 2,
                self.canvas_height // 2,
                self.canvas_width // 2 + eigen_vec[0].item(),
                self.canvas_height // 2 + eigen_vec[1].item())

    def handle_mouse_move(self, x_index: float, y_index: float):
        x_index, y_index = int(x_index), int(y_index)
        if self.prev_x_index == x_index and self.prev_y_index == y_index:
            return
        self.prev_x_index, self.prev_y_index = x_index, y_index
        k_half_height = int(self.kernel_height * (self.canvas_height / self.image_height) / 2)
        k_half_width = int(self.kernel_width * (self.canvas_width / self.image_width) / 2)

        if ((k_half_width < x_index < self.canvas_width - k_half_width) and
                (k_half_height < y_index < self.canvas_height - k_half_height)):
            self.canvas_full.draw_image(self.canvas_fresh, 0, 0,
                                        width=self.canvas_width, height=self.canvas_height)
            self.canvas_full.stroke_rect(x_index-k_half_width,
                                         y_index - k_half_height,
                                         k_half_width,
                                         k_half_height)
            y_arr_index = int(y_index * (self.image_height / self.canvas_height))
            x_arr_index = int(x_index * (self.image_width / self.canvas_width))

            self.canvas_focus.put_image_data(
                self._crop_array(self.image_array, x_arr_index, y_arr_index), 0, 0)
            self.plot_derivatives(
                self._crop_array(self.x_derivative, x_arr_index, y_arr_index),
                self._crop_array(self.y_derivative, x_arr_index, y_arr_index),
                x_arr_index,
                y_arr_index)

    def make_box(self, canvas: Canvas, label: str):
        return widgets.VBox([
            widgets.HTML(f"<h3>{label}</h3>"),
            widgets.Box([
                canvas,
            ], layout=widgets.Layout(
                width="300px",
                height="300px"
            ))
        ], layout=widgets.Layout(
            width="308px",
            margin="4px"
        ))

    def __call__(self) -> widgets.VBox:
        return widgets.VBox([
            widgets.HBox(
                [
                    self.make_box(self.canvas_full, "Hover the image below"),
                    self.make_box(self.canvas_focus, "Kernel"),
                ],

            ),
            widgets.HBox(
                [
                    self.make_box(self.canvas_corner, "Corners"),
                    self.make_box(self.canvas_plot, "Derivatives and eigen vectors plot"),
                ],
            )
        ])
