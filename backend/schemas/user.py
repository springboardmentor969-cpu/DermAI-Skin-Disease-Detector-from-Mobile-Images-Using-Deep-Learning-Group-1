from typing import Literal

from pydantic import BaseModel, EmailStr, Field


# ============================================================
# REGISTRATION
# ============================================================

class UserRegister(BaseModel):

    full_name: str = Field(
        ...,
        min_length=2,
        max_length=100
    )

    email: EmailStr

    password: str = Field(
        ...,
        min_length=8,
        max_length=128
    )

    role: Literal[
        "patient",
        "doctor",
        "admin"
    ] = "patient"


# ============================================================
# LOGIN
# ============================================================

class UserLogin(BaseModel):

    email: EmailStr

    password: str = Field(
        ...,
        min_length=1,
        max_length=128
    )


# ============================================================
# USER RESPONSE
# ============================================================

class UserResponse(BaseModel):

    id: str

    full_name: str

    email: EmailStr

    role: str


# ============================================================
# TOKEN RESPONSE
# ============================================================

class TokenResponse(BaseModel):

    access_token: str

    token_type: str