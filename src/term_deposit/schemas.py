"""Pydantic models defining API request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

JobType = Literal[
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]

MaritalStatus = Literal["married", "single", "divorced", "unknown"]
EducationLevel = Literal["primary", "secondary", "tertiary", "unknown"]
BinaryFlag = Literal["yes", "no", "unknown"]
ContactType = Literal["cellular", "telephone"]
Month = Literal[
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    balance: float
    day: int = Field(..., ge=1, le=31)
    duration: float = Field(..., ge=0)
    campaign: int = Field(..., ge=1)
    job: JobType
    marital: MaritalStatus
    education: EducationLevel
    default: BinaryFlag
    housing: BinaryFlag
    loan: BinaryFlag
    contact: ContactType
    month: Month


class PredictionResponse(BaseModel):
    subscribe_proba: float
    subscribe_label: Literal["yes", "no"]
