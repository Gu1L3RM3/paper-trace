from typing import Protocol

from papertrace.domain.models import Paper


class PaperSource(Protocol):
    async def search(self, query: str) -> list[Paper]: ...
