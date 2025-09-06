class DomainError(Exception):
    """Base exception for domain-related errors"""

    pass


class InfrastructureError(Exception):
    """Base exception for infrastructure-related errors"""

    pass


class ApplicationError(Exception):
    """Base exception for application-related errors"""

    pass


class TeamNotFoundError(DomainError):
    """Raised when a team is not found"""

    pass


class InvalidProbabilityError(DomainError):
    """Raised when probability values are invalid"""

    pass


class InvalidOddsError(DomainError):
    """Raised when odds values are invalid"""

    pass


class DataSourceError(InfrastructureError):
    """Raised when external data source fails"""

    pass


class ConfigurationError(ApplicationError):
    """Raised when configuration is invalid"""

    pass
