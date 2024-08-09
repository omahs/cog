from typing import Iterator

from cog import BasePredictor, File


class Predictor(BasePredictor):
    def predict(self, file: File) -> Iterator[str]:
        with file.open() as f:
            prefix = f.read().decode("utf-8")
        predictions = ["foo", "bar", "baz"]
        for prediction in predictions:
            yield prefix + " " + prediction
