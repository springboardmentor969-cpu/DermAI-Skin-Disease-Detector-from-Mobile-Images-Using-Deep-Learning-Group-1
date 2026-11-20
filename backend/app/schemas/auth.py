from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)
    role: str = Field(default="patient")

    @field_validator("password")
    @classmethod
    def validate_password_bytes(cls, password: str) -> str:
        if len(password.encode("utf-8")) > 72:
            raise ValueError("Password must not exceed 72 UTF-8 bytes.")
        return password

    @field_validator("full_name")
    @classmethod
    def normalize_full_name(cls, full_name: str) -> str:
        normalized = full_name.strip()
        if len(normalized) < 2:
            raise ValueError("Full name must contain at least two non-space characters.")
        return normalized

    @field_validator("role")
    @classmethod
    def normalize_role(cls, role: str) -> str:
        r = role.lower().strip()
        if r in ("doctor", "clinician", "dermatologist"):
            return "doctor"
        return "patient"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str = "patient"
    full_name: str = ""


class UserRead(BaseModel):
    id: int
    email: str
    full_name: str
    role: str

    class Config:
        from_attributes = True
