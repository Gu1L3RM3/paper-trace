from fastapi import FastAPI

app = FastAPI(title="PaperTrace AI")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
