"""
Unit tests for text sanitization functionality.
"""
import pytest
from app.utils.text_sanitizer import TextSanitizer, sanitize_user_message, normalize_car_references


class TestTextSanitizer:
    """Test cases for TextSanitizer class."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.sanitizer = TextSanitizer()
    
    def test_sanitize_my_cars_possessive(self):
        """Test sanitization of 'my car's' possessive form."""
        test_cases = [
            ("my car's engine is making noise", "the car's engine is making noise"),
            ("My car's brakes are squeaking", "the car's brakes are squeaking"),
            ("MY CAR'S transmission is slipping", "the car's transmission is slipping"),
            ("my cars engine is loud", "the car's engine is loud"),  # Missing apostrophe
        ]
        
        for input_text, expected in test_cases:
            result = self.sanitizer.sanitize_message(input_text)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_text}'"
    
    def test_sanitize_multiple_vehicle_types(self):
        """Test sanitization of different vehicle types."""
        test_cases = [
            ("my truck's engine won't start", "the truck's engine won't start"),
            ("my SUV's air conditioning failed", "the SUV's air conditioning failed"),
            ("my motorcycle's brakes need repair", "the motorcycle's brakes need repair"),
            ("my vehicle's dashboard lights are on", "the vehicle's dashboard lights are on"),
            ("my bike's chain is loose", "the bike's chain is loose"),
        ]
        
        for input_text, expected in test_cases:
            result = self.sanitizer.sanitize_message(input_text)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_text}'"
    
    def test_sanitize_car_references_without_possessive(self):
        """Test sanitization of car references without possessive."""
        test_cases = [
            ("my car is not starting", "the car is not starting"),
            ("my vehicle makes strange sounds", "the vehicle makes strange sounds"),
            ("my truck needs an oil change", "the truck needs an oil change"),
            ("I need to fix my SUV", "I need to fix the SUV"),
            ("my motorcycle won't turn on", "the motorcycle won't turn on"),
        ]
        
        for input_text, expected in test_cases:
            result = self.sanitizer.sanitize_message(input_text)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_text}'"
    
    def test_preserve_non_car_references(self):
        """Test that non-car 'my' references are preserved."""
        test_cases = [
            ("my house is nice", "my house is nice"),
            ("my dog is barking", "my dog is barking"),
            ("I love my family", "I love my family"),
            ("my phone's battery is dead", "my phone's battery is dead"),
        ]
        
        for input_text, expected in test_cases:
            result = self.sanitizer.sanitize_message(input_text)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_text}'"
    
    def test_mixed_references(self):
        """Test messages with both car and non-car references."""
        test_cases = [
            ("my car's engine and my dog are both loud", "the car's engine and my dog are both loud"),
            ("I took my car to my friend's house", "I took the car to my friend's house"),
            ("my vehicle's problem started after my vacation", "the vehicle's problem started after my vacation"),
        ]
        
        for input_text, expected in test_cases:
            result = self.sanitizer.sanitize_message(input_text)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_text}'"
    
    def test_edge_cases(self):
        """Test edge cases and special scenarios."""
        test_cases = [
            ("", ""),  # Empty string
            ("   my car's   engine   ", "the car's engine"),  # Extra whitespace
            ("mycars engine", "mycars engine"),  # No space (should not match)
            ("my friend's car's engine", "my friend's car's engine"),  # Should not change non-"my" possessives
        ]
        
        for input_text, expected in test_cases:
            result = self.sanitizer.sanitize_message(input_text)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_text}'"
    
    def test_case_insensitive_matching(self):
        """Test that pattern matching is case insensitive."""
        test_cases = [
            ("My Car's engine", "the car's engine"),
            ("MY CAR'S ENGINE", "the car's ENGINE"),
            ("mY cAr'S problem", "the car's problem"),
        ]
        
        for input_text, expected in test_cases:
            result = self.sanitizer.sanitize_message(input_text)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_text}'"
    
    def test_extract_car_mentions(self):
        """Test extraction of car-related terms."""
        test_cases = [
            ("my car's engine is loud", ["car", "engine"]),
            ("the brakes and transmission need work", ["brakes", "transmission"]),
            ("checking the dashboard lights", ["dashboard"]),
            ("my dog is barking", []),  # No car terms
        ]
        
        for input_text, expected_terms in test_cases:
            result = self.sanitizer.extract_car_mentions(input_text)
            # Convert to lowercase for comparison
            result_lower = [term.lower() for term in result]
            expected_lower = [term.lower() for term in expected_terms]
            
            for term in expected_lower:
                assert term in result_lower, f"Expected term '{term}' not found in result {result_lower}"
    
    def test_is_car_related_message(self):
        """Test detection of car-related messages."""
        car_related = [
            "my car's engine is making noise",
            "the brakes are squeaking", 
            "dashboard warning lights",
            "transmission problems"
        ]
        
        not_car_related = [
            "hello there",
            "my dog is barking",
            "what's the weather like?",
            "I need help with my computer"
        ]
        
        for message in car_related:
            assert self.sanitizer.is_car_related_message(message), f"'{message}' should be detected as car-related"
        
        for message in not_car_related:
            assert not self.sanitizer.is_car_related_message(message), f"'{message}' should not be detected as car-related"
    
    def test_normalize_contractions(self):
        """Test normalization of contractions."""
        test_cases = [
            ("my car won't start", "my car will not start"),
            ("it can't be fixed", "it cannot be fixed"),
            ("the engine doesn't work", "the engine does not work"),
            ("I haven't checked it", "I have not checked it"),
        ]
        
        for input_text, expected in test_cases:
            result = self.sanitizer.normalize_contractions(input_text)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_text}'"
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        input_text = "my car's engine won't start"
        
        # Test sanitize_user_message function
        result1 = sanitize_user_message(input_text)
        expected1 = "the car's engine won't start"
        assert result1 == expected1, f"Expected '{expected1}', got '{result1}'"
        
        # Test normalize_car_references function  
        result2 = normalize_car_references(input_text)
        expected2 = "the car's engine won't start"
        assert result2 == expected2, f"Expected '{expected2}', got '{result2}'"
    
    def test_multiple_same_references(self):
        """Test handling of multiple instances of the same reference."""
        test_cases = [
            ("my car's engine and my car's brakes", "the car's engine and the car's brakes"),
            ("my truck needs work and my truck is old", "the truck needs work and the truck is old"),
        ]
        
        for input_text, expected in test_cases:
            result = self.sanitizer.sanitize_message(input_text)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_text}'"
    
    def test_disabled_normalization(self):
        """Test that normalization can be selectively disabled."""
        input_text = "my car's engine in my car is loud"
        
        # With both enabled (default)
        result_both = self.sanitizer.sanitize_message(input_text, normalize_possessives=True, normalize_references=True)
        expected_both = "the car's engine in the car is loud"
        assert result_both == expected_both
        
        # With only possessives enabled
        result_possessive_only = self.sanitizer.sanitize_message(input_text, normalize_possessives=True, normalize_references=False)
        expected_possessive_only = "the car's engine in my car is loud"
        assert result_possessive_only == expected_possessive_only
        
        # With only references enabled
        result_reference_only = self.sanitizer.sanitize_message(input_text, normalize_possessives=False, normalize_references=True)
        expected_reference_only = "my car's engine in the car is loud"
        assert result_reference_only == expected_reference_only
        
        # With both disabled
        result_none = self.sanitizer.sanitize_message(input_text, normalize_possessives=False, normalize_references=False)
        expected_none = input_text.strip()
        assert result_none == expected_none
