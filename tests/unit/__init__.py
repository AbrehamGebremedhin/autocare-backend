"""
Unit tests for the AutoCare backend application.

This package contains unit tests for:
- Models/Schemas: User, Car, and other data models
- Services: Business logic and service layer components
- Utilities: Helper functions, formatters, and validators
- CRUD Operations: Database interaction layer
- Controllers: Request handling and API endpoints

Test Categories:
- Models: Data validation, serialization, business rules
- Services: Business logic, caching, calculations
- Utilities: Helper functions, formatters, validators
- CRUD: Database operations with mocked dependencies
- Controllers: Request handling with mocked services

Usage:
    # Run all unit tests
    pytest tests/unit/

    # Run specific test category
    pytest tests/unit/test_models_*.py
    pytest tests/unit/test_services_*.py
    pytest tests/unit/test_utils_*.py
    pytest tests/unit/test_crud_*.py

    # Run with coverage
    pytest tests/unit/ --cov=app --cov-report=html
"""