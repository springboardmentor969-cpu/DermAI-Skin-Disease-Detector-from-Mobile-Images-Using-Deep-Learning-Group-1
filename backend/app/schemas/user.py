"""
Pydantic schemas for User registration, login, and profile responses.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, ConfigDict, model_validator
from app.models.user import UserRole


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole = UserRole.PATIENT


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: str

    @model_validator(mode="after")
    def validate_identifier(self):
        if not self.email and not self.username:
            raise ValueError("Either 'email' or 'username' must be provided.")
        return self


class UserResponse(UserBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user_id: int


class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
    role: Optional[str] = None
