import re
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, field_validator

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.IGNORECASE)

_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ResearchQuestion(BaseModel):
    text: _Text


class SearchQuery(BaseModel):
    query: _Text
    angle: str


class SearchPlan(BaseModel):
    queries: Annotated[list[SearchQuery], Field(min_length=2, max_length=4)]


class Paper(BaseModel):
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    abstract: str | None
    url: str

    @field_validator("doi")
    @classmethod
    def _normalize_doi(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _DOI_PREFIX.sub("", value.strip()).lower()


class ResearchResult(BaseModel):
    papers: list[Paper]
    # Queries whose search failed, so callers can tell a narrowed result from a complete one.
    failed_queries: list[str]
