"""
Unit tests for Message Types enums.
Tests enum values, serialization, and usage.
"""
import pytest
from app.utils.message_types import MessageType, MessageSource


class TestMessageType:
    """Test cases for MessageType enum."""
    
    def test_message_type_values(self):
        """Test that all MessageType values are correct."""
        assert MessageType.INFO.value == "info"
        assert MessageType.WARNING.value == "warning"
        assert MessageType.ERROR.value == "error"
        assert MessageType.PROGRESS.value == "progress"
        assert MessageType.STAGE.value == "stage"
        assert MessageType.RESULT.value == "result"
        assert MessageType.DEBUG.value == "debug"
    
    def test_message_type_string_inheritance(self):
        """Test that MessageType inherits from str."""
        assert isinstance(MessageType.INFO, str)
        assert isinstance(MessageType.ERROR, str)
        assert isinstance(MessageType.PROGRESS, str)
        
        # Test string operations
        assert MessageType.INFO == "info"
        assert MessageType.ERROR.upper() == "ERROR"
        assert MessageType.PROGRESS.title() == "Progress"
    
    def test_message_type_enumeration(self):
        """Test MessageType enumeration."""
        all_types = list(MessageType)
        
        expected_types = [
            MessageType.INFO,
            MessageType.WARNING,
            MessageType.ERROR,
            MessageType.PROGRESS,
            MessageType.STAGE,
            MessageType.RESULT,
            MessageType.DEBUG
        ]
        
        assert len(all_types) == len(expected_types)
        for msg_type in expected_types:
            assert msg_type in all_types
    
    def test_message_type_uniqueness(self):
        """Test that all MessageType values are unique."""
        values = [msg_type.value for msg_type in MessageType]
        assert len(values) == len(set(values))
    
    def test_message_type_json_serialization(self):
        """Test MessageType JSON serialization."""
        import json
        
        # Test individual values
        assert json.dumps(MessageType.INFO) == '"info"'
        assert json.dumps(MessageType.ERROR) == '"error"'
        
        # Test in data structures
        data = {
            "type": MessageType.PROGRESS,
            "types": [MessageType.INFO, MessageType.ERROR]
        }
        
        serialized = json.dumps(data)
        assert '"type": "progress"' in serialized
        assert '"types": ["info", "error"]' in serialized
    
    def test_message_type_comparison(self):
        """Test MessageType comparison operations."""
        # Test equality
        assert MessageType.INFO == MessageType.INFO
        assert MessageType.INFO != MessageType.ERROR
        
        # Test string comparison
        assert MessageType.INFO == "info"
        assert MessageType.ERROR == "error"
        assert MessageType.INFO != "error"
        
        # Test in collections
        assert MessageType.INFO in [MessageType.INFO, MessageType.ERROR]
        assert MessageType.INFO in ["info", "error"]
    
    def test_message_type_hash(self):
        """Test MessageType hash functionality."""
        # Should be hashable (can be used as dict keys)
        type_dict = {
            MessageType.INFO: "Information message",
            MessageType.ERROR: "Error message",
            MessageType.PROGRESS: "Progress message"
        }
        
        assert type_dict[MessageType.INFO] == "Information message"
        assert type_dict[MessageType.ERROR] == "Error message"
        assert type_dict[MessageType.PROGRESS] == "Progress message"
    
    def test_message_type_string_methods(self):
        """Test MessageType string methods."""
        # Test string methods work
        assert MessageType.INFO.startswith("in")
        assert MessageType.ERROR.endswith("or")
        assert MessageType.PROGRESS.replace("gre", "XXX") == "proXXXss"
        assert len(MessageType.WARNING) == 7
    
    def test_message_type_case_sensitivity(self):
        """Test MessageType case sensitivity."""
        # Values should be lowercase
        for msg_type in MessageType:
            assert msg_type.value.islower()
        
        # Comparison should be case-sensitive
        assert MessageType.INFO != "INFO"
        assert MessageType.ERROR != "Error"


class TestMessageSource:
    """Test cases for MessageSource enum."""
    
    def test_message_source_values(self):
        """Test that all MessageSource values are correct."""
        assert MessageSource.CHAT_SERVICE.value == "chat_service"
        assert MessageSource.DIAGNOSTIC_AGENT.value == "diagnostic_agent"
        assert MessageSource.SYMPTOM_EXTRACTION.value == "symptom_extraction"
        assert MessageSource.ORCHESTRATOR.value == "orchestrator"
    
    def test_message_source_string_inheritance(self):
        """Test that MessageSource inherits from str."""
        assert isinstance(MessageSource.CHAT_SERVICE, str)
        assert isinstance(MessageSource.DIAGNOSTIC_AGENT, str)
        assert isinstance(MessageSource.ORCHESTRATOR, str)
        
        # Test string operations
        assert MessageSource.CHAT_SERVICE == "chat_service"
        assert MessageSource.DIAGNOSTIC_AGENT.upper() == "DIAGNOSTIC_AGENT"
        assert MessageSource.ORCHESTRATOR.title() == "Orchestrator"
    
    def test_message_source_enumeration(self):
        """Test MessageSource enumeration."""
        all_sources = list(MessageSource)
        
        expected_sources = [
            MessageSource.CHAT_SERVICE,
            MessageSource.DIAGNOSTIC_AGENT,
            MessageSource.SYMPTOM_EXTRACTION,
            MessageSource.ORCHESTRATOR
        ]
        
        assert len(all_sources) == len(expected_sources)
        for msg_source in expected_sources:
            assert msg_source in all_sources
    
    def test_message_source_uniqueness(self):
        """Test that all MessageSource values are unique."""
        values = [msg_source.value for msg_source in MessageSource]
        assert len(values) == len(set(values))
    
    def test_message_source_json_serialization(self):
        """Test MessageSource JSON serialization."""
        import json
        
        # Test individual values
        assert json.dumps(MessageSource.CHAT_SERVICE) == '"chat_service"'
        assert json.dumps(MessageSource.ORCHESTRATOR) == '"orchestrator"'
        
        # Test in data structures
        data = {
            "source": MessageSource.DIAGNOSTIC_AGENT,
            "sources": [MessageSource.CHAT_SERVICE, MessageSource.ORCHESTRATOR]
        }
        
        serialized = json.dumps(data)
        assert '"source": "diagnostic_agent"' in serialized
        assert '"sources": ["chat_service", "orchestrator"]' in serialized
    
    def test_message_source_comparison(self):
        """Test MessageSource comparison operations."""
        # Test equality
        assert MessageSource.CHAT_SERVICE == MessageSource.CHAT_SERVICE
        assert MessageSource.CHAT_SERVICE != MessageSource.ORCHESTRATOR
        
        # Test string comparison
        assert MessageSource.CHAT_SERVICE == "chat_service"
        assert MessageSource.ORCHESTRATOR == "orchestrator"
        assert MessageSource.CHAT_SERVICE != "orchestrator"
        
        # Test in collections
        assert MessageSource.CHAT_SERVICE in [MessageSource.CHAT_SERVICE, MessageSource.ORCHESTRATOR]
        assert MessageSource.CHAT_SERVICE in ["chat_service", "orchestrator"]
    
    def test_message_source_naming_convention(self):
        """Test MessageSource naming conventions."""
        # All values should use snake_case
        for msg_source in MessageSource:
            assert "_" in msg_source.value or msg_source.value.islower()
            assert msg_source.value.islower()
            assert " " not in msg_source.value
    
    def test_message_source_descriptive_names(self):
        """Test that MessageSource names are descriptive."""
        # Check that source names are meaningful
        source_names = [source.value for source in MessageSource]
        
        # Should contain service or agent or component identifiers
        descriptive_keywords = ["service", "agent", "extraction", "orchestrator"]
        
        for source_name in source_names:
            assert any(keyword in source_name for keyword in descriptive_keywords), \
                f"Source name '{source_name}' should contain descriptive keywords"


class TestMessageTypeSourceCombinations:
    """Test cases for MessageType and MessageSource combinations."""
    
    def test_type_source_combinations(self):
        """Test that MessageType and MessageSource can be combined."""
        combinations = [
            (MessageType.INFO, MessageSource.CHAT_SERVICE),
            (MessageType.ERROR, MessageSource.DIAGNOSTIC_AGENT),
            (MessageType.PROGRESS, MessageSource.ORCHESTRATOR),
            (MessageType.RESULT, MessageSource.SYMPTOM_EXTRACTION)
        ]
        
        for msg_type, msg_source in combinations:
            # Should be able to create message structures
            message = {
                "type": msg_type,
                "source": msg_source,
                "content": "Test message"
            }
            
            assert message["type"] == msg_type
            assert message["source"] == msg_source
            assert isinstance(message["type"], str)
            assert isinstance(message["source"], str)
    
    def test_type_source_json_serialization(self):
        """Test JSON serialization of combined types and sources."""
        import json
        
        message_data = {
            "messages": [
                {
                    "type": MessageType.INFO,
                    "source": MessageSource.CHAT_SERVICE,
                    "content": "Chat message"
                },
                {
                    "type": MessageType.ERROR,
                    "source": MessageSource.DIAGNOSTIC_AGENT,
                    "content": "Diagnostic error"
                },
                {
                    "type": MessageType.PROGRESS,
                    "source": MessageSource.ORCHESTRATOR,
                    "content": "Processing..."
                }
            ]
        }
        
        serialized = json.dumps(message_data)
        parsed = json.loads(serialized)
        
        # Verify serialization/deserialization
        assert len(parsed["messages"]) == 3
        assert parsed["messages"][0]["type"] == "info"
        assert parsed["messages"][0]["source"] == "chat_service"
        assert parsed["messages"][1]["type"] == "error"
        assert parsed["messages"][1]["source"] == "diagnostic_agent"
        assert parsed["messages"][2]["type"] == "progress"
        assert parsed["messages"][2]["source"] == "orchestrator"
    
    def test_enum_completeness(self):
        """Test that enums are complete for common use cases."""
        # MessageType should cover common message categories
        required_types = ["info", "error", "progress", "result"]
        actual_types = [msg_type.value for msg_type in MessageType]
        
        for required_type in required_types:
            assert required_type in actual_types, f"Missing required MessageType: {required_type}"
        
        # MessageSource should cover system components
        required_sources = ["chat_service", "diagnostic_agent", "orchestrator"]
        actual_sources = [msg_source.value for msg_source in MessageSource]
        
        for required_source in required_sources:
            assert required_source in actual_sources, f"Missing required MessageSource: {required_source}"
    
    def test_enum_extensibility(self):
        """Test that enums can be extended."""
        # Verify that new enum values can be added
        # (This is more of a design verification test)
        
        # Check that the enums use consistent naming patterns
        type_pattern = all(msg_type.value.islower() for msg_type in MessageType)
        source_pattern = all(msg_source.value.islower() and "_" in msg_source.value or len(msg_source.value) < 15 
                           for msg_source in MessageSource)
        
        assert type_pattern, "MessageType values should follow lowercase pattern"
        assert source_pattern, "MessageSource values should follow snake_case pattern"
    
    def test_backwards_compatibility(self):
        """Test backwards compatibility of enum values."""
        # These values should remain stable for API compatibility
        stable_types = {
            "info": MessageType.INFO,
            "error": MessageType.ERROR,
            "progress": MessageType.PROGRESS
        }
        
        for value, enum_item in stable_types.items():
            assert enum_item.value == value, f"Enum value changed: {enum_item} should be {value}"
        
        stable_sources = {
            "chat_service": MessageSource.CHAT_SERVICE,
            "diagnostic_agent": MessageSource.DIAGNOSTIC_AGENT,
            "orchestrator": MessageSource.ORCHESTRATOR
        }
        
        for value, enum_item in stable_sources.items():
            assert enum_item.value == value, f"Enum value changed: {enum_item} should be {value}"
    
    def test_enum_membership_testing(self):
        """Test enum membership testing."""
        # Test value membership
        assert "info" in [msg_type.value for msg_type in MessageType]
        assert "chat_service" in [msg_source.value for msg_source in MessageSource]
        
        # Test enum membership
        assert MessageType.INFO in MessageType
        assert MessageSource.CHAT_SERVICE in MessageSource
        
        # Test non-membership
        assert "invalid_type" not in [msg_type.value for msg_type in MessageType]
        assert "invalid_source" not in [msg_source.value for msg_source in MessageSource]
