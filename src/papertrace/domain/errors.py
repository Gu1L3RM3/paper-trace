class PaperTraceError(Exception):
    """Base for errors adapters raise so the pipeline can turn them into `error` events."""


class LLMError(PaperTraceError):
    pass


class InvalidReportError(LLMError):
    """The model answered, but its report does not hold against the retrieved evidence."""


class SourceError(PaperTraceError):
    pass
