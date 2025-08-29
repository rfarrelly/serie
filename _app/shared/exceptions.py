# shared/exceptions.py
class DomainException(Exception):
    """Base exception for domain-related errors"""

    pass


class InvalidOddsException(DomainException):
    """Raised when odds values are invalid"""

    pass


class InvalidProbabilityException(DomainException):
    """Raised when probability values are invalid"""

    pass


class MatchNotCompletedException(DomainException):
    """Raised when trying to access result of uncompleted match"""

    pass


class MatchAlreadyCompletedException(DomainException):
    """Raised when trying to record result for already completed match"""

    pass


class InsufficientDataException(DomainException):
    """Raised when there's not enough data for calculations"""

    pass


class ModelNotTrainedException(DomainException):
    """Raised when trying to use untrained model"""

    pass


class InvalidPredictionException(DomainException):
    """Raised when prediction data is invalid"""

    pass
