import io

from cog import BasePredictor, File


class Predictor(BasePredictor):
    def predict(self) -> File:
        return b"hello"
