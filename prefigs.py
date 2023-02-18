""" misc. Premade configurations and preprocessor functions for Atari games. """

import cv2
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import crop


breakout = {
    "environment_name": "Breakout-v0",
    "preprocessor": lambda state: cv2.resize(
        state[:, :, 0], (84, 84), interpolation=cv2.INTER_NEAREST
    ),
    "state_dimensionality": 7056,
    "action_dimensionality": 4,
}

space_invaders = {
    "environment_name": "SpaceInvaders-v0",
    "preprocessor": lambda state: cv2.resize(
        crop(state, ((13, 13), (15, 25), (0, 0)))[:, :, 0],
        (84, 84),
        interpolation=cv2.INTER_NEAREST,
    ),
    "state_dimensionality": 7056,
    "action_dimensionality": 6,
    "_notes": "The resizing function is a bit wonky, as we're not cropping the image to be fully square, but we are resizing it to be",
}

tennis = {
    "environment_name": "Tennis-v0",
    "preprocessor": lambda state: cv2.resize(
        state[:, :, 0], (84, 84), interpolation=cv2.INTER_NEAREST
    ),
    "state_dimensionality": 7056,
    "action_dimensionality": 18,
}

configs = {"breakout": breakout, "space_invaders": space_invaders, "tennis": tennis}
