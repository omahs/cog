import os
import tempfile
from typing import Iterator

from cog import BasePredictor, File
from PIL import Image


class Predictor(BasePredictor):
    def predict(self) -> Iterator[File]:
        colors = ["red", "blue", "yellow"]
        for i, color in enumerate(colors):
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, f"prediction-{i}.bmp")
            img = Image.new("RGB", (8, 8), color)
            yield img.tobytes()
