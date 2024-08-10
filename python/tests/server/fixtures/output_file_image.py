import os
import tempfile

from cog import BasePredictor, File
from PIL import Image


class Predictor(BasePredictor):
    def predict(self) -> File:
        return Image.new("RGB", (255, 255), "red").tobytes()
