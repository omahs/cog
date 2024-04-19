from typing import Iterator, List

from cog import BasePredictor
from cog._vendor.pydantic import BaseModel


class Output(BaseModel):
    text: str


class Predictor(BasePredictor):
    def predict(self) -> Iterator[List[Output]]:
        yield [Output(text="hello")]
