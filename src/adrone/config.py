from pydantic import BaseModel

class TrainConfig(BaseModel):
    seed: int = 42
