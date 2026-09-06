class AppError(Exception):
    """An application failure safe to expose at the API or CLI boundary."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.safe_message = message
        super().__init__(message)


class LightRAGQueryError(AppError):
    def __init__(self, message: str = "LightRAG query failed") -> None:
        super().__init__("LIGHTRAG_QUERY_FAILED", message)
