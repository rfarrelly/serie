class DomainException(Exception):
    pass


class InvalidMatchDataException(DomainException):
    pass


class InsufficientDataException(DomainException):
    pass


class ModelNotFittedException(DomainException):
    pass
