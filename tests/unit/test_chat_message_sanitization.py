"""
Integration tests for chat message sanitization.
"""
import pytest
from app.api.v1.chat_route import ChatMessageRequest, ChatSessionMessageRequest
from pydantic import ValidationError


class TestChatMessageSanitization:
    """Test message sanitization in chat API endpoints."""
    
    def test_chat_message_request_sanitization(self):
        """Test that ChatMessageRequest sanitizes 'my car's' messages."""
        # Test cases with expected sanitized outputs
        test_cases = [
            ("my car's engine is making noise", "the car's engine is making noise"),
            ("My car's brakes are squeaking", "the car's brakes are squeaking"), 
            ("my vehicle's transmission failed", "the vehicle's transmission failed"),
            ("I think my truck's problem is serious", "I think the truck's problem is serious"),
        ]
        
        for original_message, expected_sanitized in test_cases:
            request = ChatMessageRequest(
                user_id="test_user_123",
                message=original_message,
                context=None
            )
            
            # The validator should have sanitized the message
            assert request.message == expected_sanitized, \
                f"Expected '{expected_sanitized}', got '{request.message}' for input '{original_message}'"
    
    def test_chat_session_message_request_sanitization(self):
        """Test that ChatSessionMessageRequest sanitizes 'my car's' messages.""" 
        test_cases = [
            ("my car's engine won't start", "the car's engine won't start"),
            ("my SUV's air conditioning is broken", "the SUV's air conditioning is broken"),
            ("checking my motorcycle's brakes", "checking the motorcycle's brakes"),
        ]
        
        for original_message, expected_sanitized in test_cases:
            request = ChatSessionMessageRequest(
                message=original_message,
                context=None
            )
            
            assert request.message == expected_sanitized, \
                f"Expected '{expected_sanitized}', got '{request.message}' for input '{original_message}'"
    
    def test_non_car_references_preserved(self):
        """Test that non-car 'my' references are not sanitized."""
        test_cases = [
            "my house is nice",
            "I love my family", 
            "my phone's battery is dead",
            "my dog is barking",
        ]
        
        for message in test_cases:
            request = ChatMessageRequest(
                user_id="test_user_123",
                message=message,
                context=None
            )
            
            # Message should remain unchanged
            assert request.message == message, \
                f"Message '{message}' should not be changed but got '{request.message}'"
    
    def test_invalid_messages_still_raise_errors(self):
        """Test that invalid messages still raise validation errors."""
        # Test empty message
        with pytest.raises(ValidationError, match="Message must be between 1 and 5000 characters"):
            ChatMessageRequest(
                user_id="test_user_123",
                message="",
                context=None
            )
        
        # Test HTML tags
        with pytest.raises(ValidationError, match="HTML tags are not allowed"):
            ChatMessageRequest(
                user_id="test_user_123", 
                message="<script>alert('test')</script>",
                context=None
            )
        
        # Test dangerous URLs
        with pytest.raises(ValidationError, match="Potentially dangerous URLs"):
            ChatMessageRequest(
                user_id="test_user_123",
                message="javascript:alert('test')",
                context=None
            )
    
    def test_mixed_car_and_non_car_references(self):
        """Test messages with both car and non-car references."""
        test_cases = [
            ("my car's engine and my dog are both loud", "the car's engine and my dog are both loud"),
            ("I took my vehicle to my friend's house", "I took the vehicle to my friend's house"),
            ("my truck's problem started after my vacation", "the truck's problem started after my vacation"),
        ]
        
        for original_message, expected_sanitized in test_cases:
            request = ChatMessageRequest(
                user_id="test_user_123",
                message=original_message,
                context=None
            )
            
            assert request.message == expected_sanitized, \
                f"Expected '{expected_sanitized}', got '{request.message}' for input '{original_message}'"
    
    def test_edge_cases(self):
        """Test edge cases in message sanitization."""
        test_cases = [
            # Extra whitespace
            ("   my car's   engine   is   loud   ", "the car's engine is loud"),
            # Multiple car references  
            ("my car's engine and my car's brakes", "the car's engine and the car's brakes"),
            # Case insensitive
            ("My Car's Engine", "the car's Engine"),
        ]
        
        for original_message, expected_sanitized in test_cases:
            request = ChatMessageRequest(
                user_id="test_user_123",
                message=original_message,
                context=None
            )
            
            assert request.message == expected_sanitized, \
                f"Expected '{expected_sanitized}', got '{request.message}' for input '{original_message}'"
