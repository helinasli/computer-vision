from typing import Tuple, Any, Callable, List
from ipycanvas import Canvas
from ipywidgets import widgets
import numpy as np


class HoughRenderer():

    def __init__(self,
                 image_array: np.ndarray,
                 binary_edges: np.ndarray,
                 n_rho: int,
                 n_theta) -> None:
        """ Interactive visualization for Line detection via Hough transform

        Args:
            image_array (np.ndarray): 2D array of the image (H, W)
            binary_edges (np.ndarray): 2D binary array of edges (H, W)
            n_rho (int): Number of rhos in the Hough space
            n_theta (_type_): Number of thetas in the Hough space

        Raises:
            ValueError: If <binary_edges> array has a different shape from that of <image_array>
            ValueError: If data type of the <binary_edges> array is not boolean or integer
        """

        if (binary_edges.shape != image_array.shape):
            raise ValueError(
                f"Shape of <binary_edges>: {binary_edges.shape} is not the same as the shape of <image_array>: {image_array.shape}")
        if (binary_edges.dtype not in (np.bool, np.int32, np.int64, np.uint8)):
            raise ValueError("Data type of <binary_edges> bust be int or boolean")

        height, width = binary_edges.shape
        self.binary_edges = binary_edges
        self.image_array = image_array
        self.n_rho = n_rho
        self.n_theta = n_theta
        self.width = width
        self.height = height

        self.canvas_img = Canvas(width=width, height=height)
        self.canvas_img_fresh = Canvas(width=width, height=height)
        self.canvas_edge = Canvas(width=width, height=height)
        self.canvas_edge_fresh = Canvas(width=width, height=height)
        self.canvas_hough = Canvas(width=width + 32, height=height + 32)
        self.prepare()

    def prepare(self):
        # Edge initial
        binary_edge_arr = (self.binary_edges * 255)
        edge_img = np.expand_dims(binary_edge_arr, -1).repeat(3, axis=-1)
        self.canvas_edge_fresh.put_image_data(edge_img, 0, 0)
        self.canvas_edge.draw_image(self.canvas_edge_fresh, 0, 0)
        # Hough space axes
        self.canvas_hough.fill_style = "#777777"
        self.canvas_hough.font = "24px sans-serif"
        self.canvas_hough.fill_text("-π", self.width + 8, 32)
        self.canvas_hough.fill_text("π", self.width + 8, self.height)

        self.canvas_hough.fill_text("-d", 0 + 8, self.height + 24)
        self.canvas_hough.fill_text("d", self.width - 16, self.height + 24)

        self.canvas_hough.font = "32px sans-serif"
        self.canvas_hough.fill_text("θ", self.width, self.height // 2 + 16)
        self.canvas_hough.fill_text("ρ", self.width // 2, self.height + 24)

        # Hough Space initial
        self.canvas_hough.scale(self.width/self.n_rho, y=self.height/self.n_theta)
        hough_image = np.random.randint(
            0, 255, (self.n_theta, self.n_rho, 1), dtype=np.int32).repeat(3, axis=-1)
        self.canvas_hough.put_image_data(hough_image, 0, 0)
        # Image initial
        self.canvas_img_fresh.put_image_data(np.expand_dims(
            self.image_array, -1).repeat(3, axis=-1), 0, 0)
        self.canvas_img.draw_image(self.canvas_img_fresh, 0, 0)
        self.canvas_img.stroke_style = "#22DD33"
        # Edge style
        self.canvas_edge.stroke_style = "#22DD33"
        self.canvas_edge.line_width = 4

    def add_edge_to_hough_callback(self,
                                   edge_to_hough: Callable[[np.ndarray, np.ndarray], np.ndarray]
                                   ) -> Any:
        prev_x, prev_y = -1, -1

        def callback(x_index, y_index):
            nonlocal prev_x, prev_y
            x_index, y_index = int(x_index), int(y_index)
            if (prev_x == x_index) and (y_index == prev_y):
                return
            prev_x, prev_y = x_index, y_index

            hough_array = edge_to_hough(
                x_index * np.ones((1,), dtype=np.int32),
                y_index * np.ones((1,), dtype=np.int32),
            )

            self.canvas_hough.put_image_data(
                np.expand_dims(hough_array / max(1, hough_array.max()), axis=-1).repeat(3, axis=-1) * 255)

        self.canvas_edge.on_mouse_move(callback)

    def add_hough_to_edge_callback(self,
                                   hough_to_edge: Callable[[np.ndarray, np.ndarray], np.ndarray]
                                   ) -> Any:
        half_width = self.width // 2
        half_height = self.height // 2
        half_diagonal = np.sqrt(half_width**2 + half_height**2)
        prev_rho, prev_theta = -1, -1

        def callback(rho_index, theta_index):
            if rho_index > self.width or theta_index > self.height:
                return
            nonlocal prev_rho, prev_theta
            rho_index, theta_index = int(rho_index), int(theta_index)
            if (prev_rho == rho_index) and (theta_index == prev_theta):
                return
            prev_rho, prev_theta = rho_index, theta_index

            rho = ((rho_index - half_width) / half_width) * half_diagonal
            theta = ((theta_index - half_height) / half_height) * np.pi
            self.canvas_edge.draw_image(self.canvas_edge_fresh, 0, 0)
            self.canvas_edge.stroke_line(*hough_to_edge(rho, theta))
        self.canvas_hough.on_mouse_move(callback)

    def batch_draw_lines(self, line_end_points: List[Tuple[int, int, int, int]]) -> None:
        """ Draw batch of lines on the edge image.

        Args:
            line_end_points (np.ndarray): List of line segment endpoints
                - [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        """
        self.canvas_edge.stroke_style = "#22DD33"
        self.canvas_edge.line_width = 1
        self.canvas_edge.draw_image(self.canvas_edge_fresh, 0, 0)
        self.canvas_img.draw_image(self.canvas_img_fresh, 0, 0)

        for x1, y1, x2, y2 in line_end_points:
            self.canvas_edge.stroke_line(x1, y1, x2, y2)
            self.canvas_img.stroke_line(x1, y1, x2, y2)

    def draw_hough_space(self, hough_array: np.ndarray) -> None:
        """ Draw the hough space with the given array

        Args:
            hough_array (np.ndarray): 2D array of shape (H, W) (original image shape)
        """
        image = np.expand_dims((hough_array / hough_array.max()) * 255,
                               axis=-1).repeat(3, axis=-1).astype(np.int32)
        self.canvas_hough.put_image_data(image)

    def n_points_selector(self, callback: Callable[[int], None]) -> widgets.HBox:
        slider = widgets.IntSlider(
            value=20,
            min=1,
            max=50,
            step=1,
            description="",
            orientation="horizontal",
            readout=True,
            readout_format="d"
        )
        slider.observe(callback, "value")
        return widgets.HBox(
            [widgets.HTML("<p> Use n maximum points </p>"),
             slider]
        )

    def __call__(self) -> Any:
        return widgets.HBox([
            self.make_box("Original Image", self.canvas_img, width=self.width, height=self.height),
            self.make_box("Binary Edge Image", self.canvas_edge,
                          width=self.width, height=self.height),
            self.make_box("Hough Space", self.canvas_hough, width=self.width, height=self.height),
        ],
        )

    def make_box(self, title, image_canvas: Canvas, width: int, height: int) -> widgets.HBox:
        return widgets.VBox([
            widgets.HTML(f"<h3>{title}</h3>"),
            widgets.Box(
                [
                    image_canvas
                ],
                layout=widgets.Layout(
                    width=f"{width}px",
                    height=f"{height}px"
                )
            ),
        ],
            layout=widgets.Layout(width=f"{width + 6}px", border="2px solid #AAAAAA")
        )
