import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserCreate(APIModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=128)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError("username contains invalid characters")
        return value


class DocumentCreate(APIModel):
    title: str = Field(min_length=1, max_length=255)
    category: str = Field(min_length=1, max_length=100)
    body: str | None = Field(default=None, max_length=100_000)
    is_private: bool = False

    @field_validator("title", "category")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("body")
    @classmethod
    def non_empty_body(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("body must not be empty")
        return value


class DocumentUpdate(APIModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    category: str | None = Field(default=None, min_length=1, max_length=100)
    body: str | None = Field(default=None, max_length=100_000)
    is_private: bool | None = None


class DocumentResponse(BaseModel):

    id: int
    title: str
    category: str
    body: str | None = None
    is_private: bool = False
    owner_id: int | None = None

    model_config = ConfigDict(
        from_attributes=True
    )

class RoleUpdate(APIModel):
    role: str

    @field_validator("role")
    @classmethod
    def valid_role(cls, value: str) -> str:
        if value not in {"user", "admin"}:
            raise ValueError("role must be user or admin")
        return value




class AIGenerateRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("message cannot be empty")

        return value


class AIGenerateResponse(BaseModel):
    answer: str



class AIChatRequest(APIModel):
    messages: list[AIMessage] = Field(min_length=1)

class AIMessage(APIModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("message content cannot be empty")

        return value