from typing import Callable, List
from PIL.Image import Image as ImageType
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output
from functools import partial
import plotly.graph_objects as go
import numpy as np


def image_to_array(image: ImageType) -> np.ndarray:
    """ Helper function """
    return np.asarray(image)


def array_to_image(image_array: np.ndarray) -> ImageType:
    """ Helper function """
    return Image.fromarray(np.uint8(image_array))


def render_question_1(gray_image_i: ImageType, gray_image_ii: ImageType) -> widgets.HBox:
    output_gray_i = widgets.Output()
    output_gray_ii = widgets.Output()

    with output_gray_i:
        display(gray_image_i)
    with output_gray_ii:
        display(gray_image_ii)

    return widgets.HBox([output_gray_i, output_gray_ii])


def plot_density(density: np.ndarray) -> go.FigureWidget:
    fig = go.FigureWidget()
    fig.add_trace(go.Bar(x=np.arange(len(density)), y=density))
    fig.update_layout(template="plotly_white", width=600, height=350)
    return fig


def render_question_2(original_image: ImageType,
                      original_density: np.ndarray,
                      transformed_image: ImageType,
                      transformed_density: np.ndarray
                      ) -> widgets.HBox:

    output_original = widgets.Output(layout={"width": "600px"})
    output_transformed = widgets.Output(layout={"width": "600px"})

    with output_original:
        display(original_image)
    with output_transformed:
        display(transformed_image)

    return widgets.VBox([
        widgets.Label(value="Original Image"),
        output_original,
        widgets.Label(value="Original Image density"),
        plot_density(original_density),
        widgets.Label(value="Transformed Image"),
        output_transformed,
        widgets.Label(value="Transformed Image Density"),
        plot_density(transformed_density)],
        layout=widgets.Layout(width="610px")
    )


def render_question_3(original_image: ImageType,
                      scaled_image: ImageType,
                      reference_image: ImageType,
                      ) -> widgets.HBox:

    output_original = widgets.Output(layout={"width": "600px"})
    output_scaled = widgets.Output(layout={"width": "600px"})
    output_reference = widgets.Output(layout={"width": "600px"})

    with output_original:
        display(original_image)
    with output_scaled:
        display(scaled_image)
    with output_reference:
        display(reference_image)

    return widgets.HBox([
        widgets.VBox([widgets.Label(value="Reference Image"), output_reference]),
        widgets.VBox([widgets.Label(value="Input Image"), output_original]),
        widgets.VBox([widgets.Label(value="Scaled Image"), output_scaled]),
    ],
        layout=widgets.Layout()
    )


def render_question_4(local_filter: Callable[[ImageType, int, Callable[[np.ndarray], np.ndarray]], ImageType],
                      images: List[ImageType],
                      ) -> widgets.HBox:
    mean_output = widgets.Output()
    max_output = widgets.Output()

    def slider_callback(output, reducer, change):
        kernel_size = change["new"]
        with output:
            clear_output()
            for image in images:
                display(local_filter(image, kernel_size, reducer))

    slider_callback(mean_output, np.mean, {"new": 0})
    slider_callback(max_output, np.max, {"new": 0})

    max_slider = widgets.IntSlider(value=0, min=0, max=5, step=1, description="k")
    max_slider.observe(partial(slider_callback, max_output, np.max), "value")
    mean_slider = widgets.IntSlider(value=0, min=0, max=5, step=1, description="k")
    mean_slider.observe(partial(slider_callback, mean_output, np.mean), "value")

    return widgets.HBox([
        widgets.VBox([widgets.Label(value="Local max filter"), max_slider, max_output]),
        widgets.VBox([widgets.Label(value="Local mean filter"), mean_slider, mean_output]),
    ])
