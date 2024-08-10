from cog import BasePredictor, Input, File


class Predictor(BasePredictor):
    def predict(
        self,
        text: str,
        file: File,
        num1: int,
        num2: int = Input(default=10),
    ) -> str:
        with open(file, "r", encoding="utf-8") as fh:
            return text + " " + str(num1 * num2) + " " + fh.read()
