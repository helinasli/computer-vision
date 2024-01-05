from typing import List, Callable, Any, Tuple
from IPython.display import display, clear_output
import ipywidgets as widgets
from PIL.Image import Image as ImageType


class ImageRenderer():

    def __init__(self, image: ImageType, control_height: str = "200px") -> None:
        self.output = widgets.Output()
        self.org_image = image
        self.control_height = control_height

    def render_image(self, image: ImageType):
        with self.output:
            clear_output(wait=True)
            display(image)

    def control_layout(self, children: List[widgets.Widget]):
        return widgets.VBox(
            children,
            layout=widgets.Layout(height=self.control_height, width="320px")
        )

    def control_context(self):
        self.render_image(self.org_image)
        return [widgets.HTML("<h2>Original Image</h2>")]

    def __call__(self):
        return widgets.VBox(
            [
                self.control_layout(self.control_context()),
                self.output,
            ],
            layout=widgets.Layout(width="325px", border="2px solid #aaaaaa", margin="2px")
        )


class GaussianRenderer(ImageRenderer):

    def __init__(self, image: ImageType, gaussian_filter: Callable[[ImageType, List[int], float], ImageType]) -> None:
        super().__init__(image)
        self.gaussian_filter = gaussian_filter

    def control_context(self):
        kernel_size_slider = widgets.IntSlider(
            value=3,
            step=2,
            min=1,
            max=11,
        )
        sigma_slider = widgets.FloatSlider(
            value=1,
            min=0.5,
            max=4.0,
            step=0.5,
        )
        controls = [
            widgets.HTML("<h2>Gaussian Filter</h2>"),
            widgets.HTML("<b>Sigma</b>"),
            sigma_slider,
            widgets.HTML("<b>kernel size</b>"),
            kernel_size_slider,
        ]

        def slider_callback(change: Any) -> None:
            kernel_shape = (int(kernel_size_slider.value), int(kernel_size_slider.value))
            sigma = sigma_slider.value
            image = self.gaussian_filter(self.org_image, kernel_shape, sigma)
            self.render_image(image)

        sigma_slider.observe(slider_callback, names="value")
        kernel_size_slider.observe(slider_callback, names="value")
        slider_callback(None)

        return controls


class BoxRenderer(ImageRenderer):

    def __init__(self, image: ImageType, box_filter: Callable[[ImageType, List[int]], ImageType]) -> None:
        super().__init__(image)
        self.box_filter = box_filter

    def control_context(self):
        kernel_x_slider = widgets.IntSlider(
            value=3,
            step=2,
            min=1,
            max=11,
        )
        kernel_y_slider = widgets.IntSlider(
            value=3,
            step=2,
            min=1,
            max=11,
        )
        controls = [
            widgets.HTML("<h2>Box Filter</h2>"),
            widgets.HTML("<b>Kernel width</b>"),
            kernel_x_slider,
            widgets.HTML("<b>kernel height</b>"),
            kernel_y_slider,
        ]

        def slider_callback(change: Any) -> None:
            kernel_shape = (int(kernel_y_slider.value), int(kernel_x_slider.value))
            image = self.box_filter(self.org_image, kernel_shape)
            self.render_image(image)

        kernel_x_slider.observe(slider_callback, names="value")
        kernel_y_slider.observe(slider_callback, names="value")
        slider_callback(None)

        return controls


def noise_renderers(image: ImageType,
                    gaussian_filter: Callable[[ImageType, List[int], float], ImageType],
                    box_filter: Callable[[ImageType, List[int]], ImageType]
                    ):
    return widgets.HBox(
        [
            ImageRenderer(image)(),
            GaussianRenderer(image, gaussian_filter)(),
            BoxRenderer(image, box_filter)(),
        ],
        layout=widgets.Layout()
    )


class EdgeRenderer(ImageRenderer):

    def __init__(self, image: ImageType, name: str, control_height: str = "70px") -> None:
        super().__init__(image, control_height)
        self.name = name

    def control_context(self):
        self.render_image(self.org_image)
        return [widgets.HTML(f"<h2>{self.name}</h2>")]


def edge_renderers(*image_name_pairs: List[Tuple[ImageType, str]]):
    return widgets.HBox(
        [
            EdgeRenderer(image, name)()
            for image, name in image_name_pairs
        ],
        layout=widgets.Layout()
    )
