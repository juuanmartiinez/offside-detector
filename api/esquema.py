from pydantic import BaseModel

class PeticionAnalisis(BaseModel):

    id: str
    correspondencias: list[tuple[tuple[float, float], tuple[float, float]]]
    iAtacante: int
    iDefensor: int
    direccion: str
    tolerancia: float = 0.5

