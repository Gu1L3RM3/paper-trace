import sys

import uvicorn
from pydantic import ValidationError

from papertrace.settings import Settings


def main() -> int:
    try:
        Settings()
    except ValidationError as exc:
        fields = ", ".join(str(e["loc"][0]).upper() for e in exc.errors())
        print(f"error: invalid configuration, check: {fields}", file=sys.stderr)
        return 1
    uvicorn.run("papertrace.api.app:app", host="127.0.0.1", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
