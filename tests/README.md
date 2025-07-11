# AutoCare Backend Unit Tests

This directory contains comprehensive unit tests for the AutoCare backend application. The tests are organized by component type and follow best practices for testing Python applications.

## Test Structure

```
tests/
├── conftest.py                    # Test configuration and fixtures
├── unit/
│   ├── __init__.py               # Unit test package init
│   ├── test_models_user.py       # User model/schema tests
│   ├── test_models_car.py        # Car model/schema tests
│   ├── test_services_base.py     # Base service tests
│   ├── test_utils_json.py        # JSON utility tests
│   ├── test_utils_message_formatter.py  # Message formatter tests
│   ├── test_utils_message_types.py      # Message types enum tests
│   ├── test_crud_user.py         # User CRUD tests
│   └── test_main_application.py  # Main application tests
├── integration/                  # Integration tests (separate)
└── README.md                     # This file
```

## Test Categories

### 1. Models/Entities Tests (`test_models_*.py`)

- **Purpose**: Test data models, schemas, and business rules
- **Coverage**:
  - Field validation
  - Serialization/deserialization
  - Business rule enforcement
  - Edge cases and error conditions
- **Examples**:
  - User email validation
  - Car year validation
  - Required field enforcement
  - Data type validation

### 2. Services Tests (`test_services_*.py`)

- **Purpose**: Test business logic and service layer
- **Coverage**:
  - Business logic calculations
  - Service method behavior
  - Caching mechanisms
  - Error handling
- **Examples**:
  - BaseService caching functionality
  - Rate limiting behavior
  - WebSocket integration
  - Service lifecycle management

### 3. Utilities Tests (`test_utils_*.py`)

- **Purpose**: Test utility functions and helpers
- **Coverage**:
  - Helper functions
  - Formatters and validators
  - Enum functionality
  - Type conversions
- **Examples**:
  - JSON datetime serialization
  - Message formatting
  - Enum value validation
  - String manipulation utilities

### 4. CRUD Tests (`test_crud_*.py`)

- **Purpose**: Test database operations with mocked dependencies
- **Coverage**:
  - Database queries
  - Data validation
  - Error handling
  - Transaction management
- **Examples**:
  - User CRUD operations
  - Database connection handling
  - Query parameter validation
  - Error response handling

### 5. Controllers Tests (`test_main_*.py`)

- **Purpose**: Test API endpoints and request handling
- **Coverage**:
  - Request/response handling
  - Authentication/authorization
  - Error handling middleware
  - WebSocket connections
- **Examples**:
  - API endpoint responses
  - Error handler behavior
  - Middleware functionality
  - WebSocket message handling

## Running Tests

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you're in the project root directory
cd /path/to/autocare-backend
```

### Quick Commands

```bash
# Run all unit tests
python run_tests.py all

# Run specific test categories
python run_tests.py models      # Model/schema tests
python run_tests.py services    # Service layer tests
python run_tests.py utils       # Utility function tests
python run_tests.py crud        # CRUD operation tests
python run_tests.py controllers # Controller/API tests

# Run with coverage report
python run_tests.py coverage

# Fast feedback mode (stop on first failure)
python run_tests.py fast
```

### Direct pytest Commands

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_models_user.py -v

# Run specific test class
pytest tests/unit/test_models_user.py::TestUserBase -v

# Run specific test method
pytest tests/unit/test_models_user.py::TestUserBase::test_user_creation_with_minimal_data -v

# Run with coverage
pytest tests/unit/ --cov=app --cov-report=html --cov-report=term-missing

# Run with markers
pytest tests/unit/ -m "not slow" -v

# Run in parallel (if pytest-xdist is installed)
pytest tests/unit/ -n auto
```

## Test Configuration

### pytest.ini Settings (in pyproject.toml)

- Async test support enabled
- Coverage configuration
- Test discovery patterns
- Timeout settings
- Marker definitions

### Fixtures (conftest.py)

- **Mock Dependencies**: Pre-configured mocks for services
- **Sample Data**: Test data for models and operations
- **Async Support**: Event loop and async fixtures
- **Environment Setup**: Mock environment variables
- **Error Simulation**: Utilities for testing error conditions

## Writing New Tests

### Test Naming Convention

- Test files: `test_<component>_<module>.py`
- Test classes: `Test<ComponentName>`
- Test methods: `test_<functionality>_<condition>`

### Example Test Structure

```python
"""
Unit tests for <Component> <Module>.
Tests <functionality description>.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.module.component import ComponentClass


class TestComponentClass:
    """Test cases for ComponentClass."""

    @pytest.fixture
    def mock_dependency(self):
        """Create mock dependency."""
        return AsyncMock()

    @pytest.fixture
    def component_instance(self, mock_dependency):
        """Create component instance."""
        return ComponentClass(mock_dependency)

    def test_functionality_success(self, component_instance):
        """Test successful functionality."""
        # Arrange
        test_data = {"key": "value"}

        # Act
        result = component_instance.method(test_data)

        # Assert
        assert result is not None
        assert result["key"] == "value"

    @pytest.mark.asyncio
    async def test_async_functionality(self, component_instance):
        """Test async functionality."""
        result = await component_instance.async_method()
        assert result is not None

    def test_error_handling(self, component_instance):
        """Test error handling."""
        with pytest.raises(ValueError) as exc_info:
            component_instance.method(None)

        assert "Invalid input" in str(exc_info.value)
```

### Best Practices

1. **Test Isolation**: Each test should be independent
2. **Mock External Dependencies**: Use mocks for databases, APIs, etc.
3. **Test Edge Cases**: Include boundary conditions and error cases
4. **Clear Test Names**: Test names should describe what they test
5. **Arrange-Act-Assert**: Structure tests clearly
6. **Use Fixtures**: Reuse test data and setup
7. **Async Testing**: Use `@pytest.mark.asyncio` for async tests
8. **Parametrized Tests**: Use `@pytest.mark.parametrize` for multiple inputs

## Coverage Goals

### Target Coverage

- **Overall**: 80% minimum
- **Models**: 95% (high coverage for data validation)
- **Services**: 85% (business logic coverage)
- **Utilities**: 90% (pure functions, easy to test)
- **CRUD**: 80% (with mocked dependencies)
- **Controllers**: 75% (complex integration points)

### Coverage Reports

```bash
# Generate HTML coverage report
python run_tests.py coverage

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html # Windows
```

## Continuous Integration

### GitHub Actions / CI Pipeline

```yaml
- name: Run Unit Tests
  run: |
    python run_tests.py coverage

- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Run tests before commit
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Ensure you're in the project root
   cd /path/to/autocare-backend

   # Check PYTHONPATH
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

2. **Async Test Failures**

   ```python
   # Use pytest-asyncio
   @pytest.mark.asyncio
   async def test_async_function():
       result = await async_function()
       assert result is not None
   ```

3. **Mock Issues**

   ```python
   # Use proper mock imports
   from unittest.mock import AsyncMock, MagicMock, patch

   # Mock async functions with AsyncMock
   mock_async_func = AsyncMock(return_value="test")
   ```

4. **Fixture Scope Issues**
   ```python
   # Use appropriate fixture scope
   @pytest.fixture(scope="function")  # Default
   @pytest.fixture(scope="class")     # Shared across test class
   @pytest.fixture(scope="session")   # Shared across test session
   ```

### Debug Mode

```bash
# Run with debug output
pytest tests/unit/ -v --tb=long --capture=no

# Run single test with pdb
pytest tests/unit/test_models_user.py::test_specific -v --pdb
```

## Test Data Management

### Sample Data

- Located in `conftest.py` fixtures
- Realistic test data for each model
- Edge case data for boundary testing
- Error condition data for negative testing

### Data Factories

```python
@pytest.fixture
def user_factory():
    """Factory for creating test users."""
    def _create_user(**kwargs):
        defaults = {
            "email": "test@example.com",
            "role": "user",
            "created_at": datetime.now(timezone.utc)
        }
        defaults.update(kwargs)
        return UserBase(**defaults)
    return _create_user
```

## Performance Testing

### Timing Tests

```python
@pytest.mark.performance
def test_function_performance(performance_timer):
    """Test function performance."""
    performance_timer.start()

    # Run function
    result = expensive_function()

    performance_timer.stop()

    assert performance_timer.elapsed < 1.0  # Should complete in under 1 second
    assert result is not None
```

### Memory Testing

```python
def test_memory_usage():
    """Test memory usage doesn't grow unbounded."""
    import tracemalloc

    tracemalloc.start()

    # Run operations
    for i in range(1000):
        create_and_destroy_object()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert current < 1024 * 1024  # Less than 1MB
```

## Contributing

### Adding New Tests

1. Create test file following naming convention
2. Write comprehensive test cases
3. Include both positive and negative test cases
4. Add appropriate fixtures and mocks
5. Ensure tests pass locally
6. Update documentation if needed

### Test Review Checklist

- [ ] Test names are descriptive
- [ ] Tests are isolated and independent
- [ ] External dependencies are mocked
- [ ] Edge cases are covered
- [ ] Error conditions are tested
- [ ] Async tests use proper decorators
- [ ] Coverage meets minimum requirements
- [ ] Tests pass consistently

## Resources

### Documentation

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)

### Tools

- **pytest**: Test framework
- **pytest-asyncio**: Async test support
- **pytest-mock**: Mock fixtures
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel testing (optional)

### Best Practices

- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Pydantic Testing](https://docs.pydantic.dev/latest/concepts/testing/)
