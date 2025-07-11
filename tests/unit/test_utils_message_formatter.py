"""
Unit tests for Message Formatter utility.
Tests message formatting, validation, and type handling.
"""
import pytest
from datetime import datetime, timezone
from app.utils.message_formatter import MessageFormatter
from app.utils.message_types import MessageType, MessageSource


class TestMessageFormatter:
    """Test cases for MessageFormatter class."""
    
    def test_format_basic_message(self):
        """Test formatting a basic message with required fields."""
        result = MessageFormatter.format(
            type=MessageType.INFO,
            source=MessageSource.CHAT_SERVICE,
            content="Hello, how are you?"
        )
        
        assert isinstance(result, dict)
        assert result["type"] == MessageType.INFO.value
        assert result["source"] == MessageSource.CHAT_SERVICE.value
        assert result["content"] == "Hello, how are you?"
        assert "timestamp" in result
        assert result["timestamp"].endswith("Z")
    
    def test_format_message_with_session_id(self):
        """Test formatting a message with session ID."""
        result = MessageFormatter.format(
            type=MessageType.INFO,
            source=MessageSource.DIAGNOSTIC_AGENT,
            content="I'm doing well, thank you!",
            session_id="session-123"
        )
        
        assert result["session_id"] == "session-123"
        assert result["type"] == MessageType.INFO.value
        assert result["source"] == MessageSource.DIAGNOSTIC_AGENT.value
        assert result["content"] == "I'm doing well, thank you!"
    
    def test_format_message_with_progress(self):
        """Test formatting a message with progress indicator."""
        result = MessageFormatter.format(
            type=MessageType.PROGRESS,
            source=MessageSource.ORCHESTRATOR,
            content="Processing your request...",
            progress=0.75
        )
        
        assert result["progress"] == 0.75
        assert result["type"] == MessageType.PROGRESS.value
        assert result["source"] == MessageSource.ORCHESTRATOR.value
        assert result["content"] == "Processing your request..."
    
    def test_format_message_with_details(self):
        """Test formatting a message with additional details."""
        details = {
            "error_code": "E001",
            "retry_count": 3,
            "additional_info": "Database connection failed"
        }
        
        result = MessageFormatter.format(
            type=MessageType.ERROR,
            source=MessageSource.ORCHESTRATOR,
            content="An error occurred",
            details=details
        )
        
        assert result["details"] == details
        assert result["details"]["error_code"] == "E001"
        assert result["details"]["retry_count"] == 3
        assert result["details"]["additional_info"] == "Database connection failed"
    
    def test_format_message_with_custom_timestamp(self):
        """Test formatting a message with custom timestamp."""
        custom_timestamp = "2025-01-01T12:00:00Z"
        
        result = MessageFormatter.format(
            type=MessageType.INFO,
            source=MessageSource.CHAT_SERVICE,
            content="Custom timestamp message",
            timestamp=custom_timestamp
        )
        
        assert result["timestamp"] == custom_timestamp
        assert result["type"] == MessageType.INFO.value
        assert result["source"] == MessageSource.CHAT_SERVICE.value
        assert result["content"] == "Custom timestamp message"
    
    def test_format_message_with_all_fields(self):
        """Test formatting a message with all possible fields."""
        details = {"key": "value"}
        custom_timestamp = "2025-01-01T12:00:00Z"
        
        result = MessageFormatter.format(
            type=MessageType.RESULT,
            source=MessageSource.ORCHESTRATOR,
            content="Complete message",
            session_id="session-456",
            progress=0.5,
            details=details,
            timestamp=custom_timestamp
        )
        
        assert result["type"] == MessageType.RESULT.value
        assert result["source"] == MessageSource.ORCHESTRATOR.value
        assert result["content"] == "Complete message"
        assert result["session_id"] == "session-456"
        assert result["progress"] == 0.5
        assert result["details"] == details
        assert result["timestamp"] == custom_timestamp
    
    def test_format_message_auto_timestamp_format(self):
        """Test that auto-generated timestamp has correct format."""
        result = MessageFormatter.format(
            type=MessageType.INFO,
            source=MessageSource.CHAT_SERVICE,
            content="Test message"
        )
        
        timestamp = result["timestamp"]
        assert isinstance(timestamp, str)
        assert timestamp.endswith("Z")
        assert "T" in timestamp
        assert len(timestamp) >= 19  # YYYY-MM-DDTHH:MM:SS format minimum
        
        # Verify it's a valid ISO format
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail("Generated timestamp is not in valid ISO format")
    
    def test_format_message_types_enum_values(self):
        """Test formatting messages with different MessageType enum values."""
        message_types = [
            MessageType.INFO,
            MessageType.WARNING,
            MessageType.ERROR,
            MessageType.PROGRESS,
            MessageType.STAGE,
            MessageType.RESULT,
            MessageType.DEBUG
        ]
        
        for msg_type in message_types:
            result = MessageFormatter.format(
                type=msg_type,
                source=MessageSource.CHAT_SERVICE,
                content=f"Test {msg_type.value} message"
            )
            
            assert result["type"] == msg_type.value
            assert result["content"] == f"Test {msg_type.value} message"
    
    def test_format_message_sources_enum_values(self):
        """Test formatting messages with different MessageSource enum values."""
        message_sources = [
            MessageSource.CHAT_SERVICE,
            MessageSource.DIAGNOSTIC_AGENT,
            MessageSource.SYMPTOM_EXTRACTION,
            MessageSource.ORCHESTRATOR
        ]
        
        for msg_source in message_sources:
            result = MessageFormatter.format(
                type=MessageType.INFO,
                source=msg_source,
                content=f"Test message from {msg_source.value}"
            )
            
            assert result["source"] == msg_source.value
            assert result["content"] == f"Test message from {msg_source.value}"
    
    def test_format_message_with_none_optional_fields(self):
        """Test formatting message with None values for optional fields."""
        result = MessageFormatter.format(
            type=MessageType.INFO,
            source=MessageSource.CHAT_SERVICE,
            content="Test message",
            session_id=None,
            progress=None,
            details=None,
            timestamp=None
        )
        
        # None values should not be included in the result
        assert "session_id" not in result
        assert "progress" not in result
        assert "details" not in result
        # timestamp should be auto-generated when None
        assert "timestamp" in result
        assert result["timestamp"].endswith("Z")
    
    def test_format_message_with_empty_content(self):
        """Test formatting message with empty content."""
        result = MessageFormatter.format(
            type=MessageType.INFO,
            source=MessageSource.CHAT_SERVICE,
            content=""
        )
        
        assert result["content"] == ""
        assert result["type"] == MessageType.INFO.value
        assert result["source"] == MessageSource.CHAT_SERVICE.value
    
    def test_format_message_with_complex_details(self):
        """Test formatting message with complex details object."""
        complex_details = {
            "error": {
                "code": "E001",
                "message": "Database error",
                "stack_trace": ["line1", "line2", "line3"]
            },
            "context": {
                "user_id": "user-123",
                "session_id": "session-456",
                "timestamp": "2025-01-01T12:00:00Z"
            },
            "metadata": {
                "version": "1.0",
                "environment": "test"
            }
        }
        
        result = MessageFormatter.format(
            type=MessageType.ERROR,
            source=MessageSource.ORCHESTRATOR,
            content="Complex error occurred",
            details=complex_details
        )
        
        assert result["details"] == complex_details
        assert result["details"]["error"]["code"] == "E001"
        assert result["details"]["context"]["user_id"] == "user-123"
        assert result["details"]["metadata"]["version"] == "1.0"
    
    def test_format_message_progress_boundary_values(self):
        """Test formatting messages with progress boundary values."""
        progress_values = [0.0, 0.5, 1.0, 0.0001, 0.9999]
        
        for progress in progress_values:
            result = MessageFormatter.format(
                type=MessageType.PROGRESS,
                source=MessageSource.ORCHESTRATOR,
                content=f"Progress: {progress}",
                progress=progress
            )
            
            assert result["progress"] == progress
            assert result["content"] == f"Progress: {progress}"
    
    def test_format_message_static_method(self):
        """Test that format is a static method and can be called without instance."""
        # Should be able to call without creating an instance
        result = MessageFormatter.format(
            type=MessageType.INFO,
            source=MessageSource.CHAT_SERVICE,
            content="Static method test"
        )
        
        assert isinstance(result, dict)
        assert result["content"] == "Static method test"
    
    def test_format_message_immutability(self):
        """Test that original objects are not modified during formatting."""
        original_details = {"key": "value"}
        original_content = "Original content"
        
        result = MessageFormatter.format(
            type=MessageType.INFO,
            source=MessageSource.CHAT_SERVICE,
            content=original_content,
            details=original_details
        )
        
        # Original objects should remain unchanged
        assert original_details == {"key": "value"}
        assert original_content == "Original content"
        
        # Result should contain copies
        assert result["details"] == original_details
        assert result["content"] == original_content
    
    def test_format_message_special_characters_content(self):
        """Test formatting message with special characters in content."""
        special_contents = [
            "Message with emoji 😊",
            "Message with Unicode: café",
            "Message with newlines\\nand tabs\\t",
            "Message with quotes: \"hello\" and 'world'",
            "Message with symbols: !@#$%^&*()",
            "Message with HTML: <div>content</div>",
            "Message with JSON: {\"key\": \"value\"}"
        ]
        
        for content in special_contents:
            result = MessageFormatter.format(
                type=MessageType.INFO,
                source=MessageSource.CHAT_SERVICE,
                content=content
            )
            
            assert result["content"] == content
            assert isinstance(result["content"], str)
    
    def test_format_message_large_content(self):
        """Test formatting message with large content."""
        large_content = "A" * 10000  # 10KB of content
        
        result = MessageFormatter.format(
            type=MessageType.INFO,
            source=MessageSource.CHAT_SERVICE,
            content=large_content
        )
        
        assert result["content"] == large_content
        assert len(result["content"]) == 10000
    
    def test_format_message_consistent_structure(self):
        """Test that formatted messages have consistent structure."""
        messages = []
        
        # Create various message types
        test_cases = [
            (MessageType.INFO, MessageSource.CHAT_SERVICE, "Hello"),
            (MessageType.PROGRESS, MessageSource.ORCHESTRATOR, "Loading...", 0.5),
            (MessageType.ERROR, MessageSource.ORCHESTRATOR, "Error occurred", None, {"code": "E001"}),
            (MessageType.INFO, MessageSource.DIAGNOSTIC_AGENT, "Information", None, None, "session-123")
        ]
        
        for case in test_cases:
            msg_type, source, content = case[:3]
            progress = case[3] if len(case) > 3 else None
            details = case[4] if len(case) > 4 else None
            session_id = case[5] if len(case) > 5 else None
            
            result = MessageFormatter.format(
                type=msg_type,
                source=source,
                content=content,
                progress=progress,
                details=details,
                session_id=session_id
            )
            
            messages.append(result)
        
        # Verify all messages have required fields
        for msg in messages:
            assert "type" in msg
            assert "source" in msg
            assert "content" in msg
            assert "timestamp" in msg
            assert isinstance(msg["timestamp"], str)
            assert msg["timestamp"].endswith("Z")
