from itertools import zip_longest
from typing import NoReturn

import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

DEFAULT_COLORS = {"bbox": (128, 0, 0), "text": (255, 255, 255)}


def draw_bounding_box_on_image(
    image: Image,
    x_min,
    y_min,
    x_max,
    y_max,
    color,
    thickness=4,
    display_str=(),
    use_normalized_coordinates=True,
    fontsize=20,
) -> NoReturn:
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (x_min * im_width, x_max * im_width, y_min * im_height, y_max * im_height)
    else:
        (left, right, top, bottom) = (x_min, x_max, y_min, y_max)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color["bbox"]
    )

    try:
        font = ImageFont.truetype("/data/DejaVuSansMono.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    text_left = left
    # Reverse list and print from bottom to top.
    for display_str in display_str:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(text_left, top - text_height), (text_left + text_width + 2 * margin, top)], fill=color["bbox"])
        draw.text((text_left, top - text_height), display_str, fill=color["text"], font=font)
        text_left += text_width - 2 * margin


def draw_bounding_boxes_on_image(
    image,
    bboxes,
    labels=(),
    label2colors=None,
    thickness=4,
    display_str_list=(),
    use_normalized_coordinates=True,
    fontsize=20,
):
    if label2colors is None:
        label2colors = {}
    image_pil = Image.fromarray(image)
    for bbox, label, display_str in zip_longest(bboxes, labels, display_str_list):
        draw_bounding_box_on_image(
            image=image_pil,
            x_min=bbox[0],
            y_min=bbox[1],
            x_max=bbox[2],
            y_max=bbox[3],
            color=label2colors.get(label, DEFAULT_COLORS),
            thickness=thickness,
            display_str=[] if display_str is None else display_str,
            use_normalized_coordinates=use_normalized_coordinates,
            fontsize=fontsize,
        )
    np.copyto(image, np.array(image_pil))
