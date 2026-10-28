class PaperTraceError(Exception):
    """Base for errors adapters raise so the pipeline can turn them into `error` events."""


class LLMError(PaperTraceError):
    pass


class SourceError(PaperTraceError):
    pass
