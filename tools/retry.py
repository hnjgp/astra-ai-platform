from typing import Type


class RetryPolicy:
    """
    Defines when a failed tool execution should be retried.
    """

    def __init__(
        self,
        max_retries: int = 2,
        retryable_exceptions: tuple[Type[Exception], ...] = (
            TimeoutError,
            ConnectionError,
        ),
    ):
        if max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative"
            )

        self.max_retries = max_retries
        self.retryable_exceptions = retryable_exceptions

    def should_retry(
        self,
        exception: Exception,
        retries_used: int,
    ) -> bool:
        """
        Return True when the exception is retryable
        and the retry limit has not been reached.
        """

        if retries_used >= self.max_retries:
            return False

        return isinstance(
            exception,
            self.retryable_exceptions,
        )