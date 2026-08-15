from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    password: str


class DocumentCreate(BaseModel):
    title: str
    category: str


class DocumentUpdate(BaseModel):
    title: str
    category: str


class DocumentResponse(BaseModel):
    id: int
    title: str
    category: str

    class Config:
        from_attributes = True

class AdminCreate(BaseModel):
    username: str
    password: str