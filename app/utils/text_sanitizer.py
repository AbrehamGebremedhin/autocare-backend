"""
Text sanitization utilities for user input processing.
"""
import re
from typing import Dict, List, Optional
from app.utils.logger import get_logger_instance

logger = get_logger_instance(__name__)

class TextSanitizer:
    """Utility class for sanitizing and normalizing user text input."""
    
    # Common possessive patterns that should be normalized
    POSSESSIVE_PATTERNS = {
        # Pattern: normalized replacement (using character class for apostrophes)
        r"\bmy\s+car[\'\u2019]?s\b": "the car's",
        r"\bmy\s+vehicle[\'\u2019]?s\b": "the vehicle's", 
        r"\bmy\s+truck[\'\u2019]?s\b": "the truck's",
        r"\bmy\s+suv[\'\u2019]?s\b": "the SUV's",
        r"\bmy\s+motorcycle[\'\u2019]?s\b": "the motorcycle's",
        r"\bmy\s+bike[\'\u2019]?s\b": "the bike's",
    }
    
    # Additional car-related normalizations
    CAR_REFERENCE_PATTERNS = {
        r"\bmy\s+car(?![\'\u2019])\b": "the car",  # Match "my car" but not "my car's"
        r"\bmy\s+vehicle(?![\'\u2019])\b": "the vehicle",  # Match "my vehicle" but not "my vehicle's"
        r"\bmy\s+truck(?![\'\u2019])\b": "the truck",  # Match "my truck" but not "my truck's"
        r"\bmy\s+suv(?![\'\u2019])\b": "the SUV",  # Match "my SUV" but not "my SUV's"
        r"\bmy\s+motorcycle(?![\'\u2019])\b": "the motorcycle",  # Match "my motorcycle" but not "my motorcycle's"
        r"\bmy\s+bike(?![\'\u2019])\b": "the bike",  # Match "my bike" but not "my bike's"
    }
    
    def __init__(self):
        self.logger = get_logger_instance("TextSanitizer")
    
    def sanitize_message(self, message: str, normalize_possessives: bool = True, normalize_references: bool = True) -> str:
        """
        Sanitize and normalize a user message.
        
        Args:
            message: The raw user message
            normalize_possessives: Whether to normalize possessive references (e.g., "my car's" -> "the car's")
            normalize_references: Whether to normalize car references (e.g., "my car" -> "the car")
        
        Returns:
            Sanitized and normalized message
        """
        if not message:
            return message
        
        sanitized = message.strip()
        original = sanitized
        
        try:
            # Normalize possessive patterns first (more specific)
            if normalize_possessives:
                sanitized = self._apply_pattern_replacements(sanitized, self.POSSESSIVE_PATTERNS)
            
            # Then normalize general car references
            if normalize_references:
                sanitized = self._apply_pattern_replacements(sanitized, self.CAR_REFERENCE_PATTERNS)
            
            # Clean up any double spaces that might have been introduced
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()
            
            # Log sanitization if changes were made
            if sanitized != original:
                self.logger.info(f"Sanitized message: '{original}' -> '{sanitized}'")
            
            return sanitized
            
        except Exception as e:
            self.logger.error(f"Error sanitizing message: {str(e)}")
            # Return original message if sanitization fails
            return original
    
    def _apply_pattern_replacements(self, text: str, patterns: Dict[str, str]) -> str:
        """Apply pattern-based replacements to text."""
        for pattern, replacement in patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def extract_car_mentions(self, message: str) -> List[str]:
        """
        Extract mentions of car-related terms from a message.
        
        Returns:
            List of car-related terms found in the message
        """
        car_terms = []
        car_patterns = [
            r"\b(car|vehicle|truck|suv|motorcycle|bike)s?\b",
            r"\b(engine|motor|transmission|brakes?|tire|wheel)s?\b",
            r"\b(dashboard|steering|accelerator|clutch|gear)s?\b"
        ]
        
        for pattern in car_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            car_terms.extend(matches)
        
        return list(set(car_terms))  # Remove duplicates
    
    def is_car_related_message(self, message: str) -> bool:
        """Check if a message contains car-related content."""
        car_mentions = self.extract_car_mentions(message)
        return len(car_mentions) > 0
    
    def normalize_contractions(self, text: str) -> str:
        """Normalize common contractions in text."""
        contractions = {
            r"\bwon't\b": "will not",
            r"\bcan't\b": "cannot", 
            r"\bdon't\b": "do not",
            r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not",
            r"\bisn't\b": "is not",
            r"\baren't\b": "are not",
            r"\bwasn't\b": "was not",
            r"\bweren't\b": "were not",
            r"\bhaven't\b": "have not",
            r"\bhasn't\b": "has not",
            r"\bhadn't\b": "had not",
            r"\bwouldn't\b": "would not",
            r"\bshouldn't\b": "should not",
            r"\bcouldn't\b": "could not",
            r"\bmustn't\b": "must not",
            r"\bmightn't\b": "might not",
            r"\bneedn't\b": "need not"
        }
        
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

# Global instance for easy access
text_sanitizer = TextSanitizer()

def sanitize_user_message(message: str) -> str:
    """
    Convenience function to sanitize user messages.
    
    Args:
        message: Raw user message
    
    Returns:
        Sanitized message
    """
    return text_sanitizer.sanitize_message(message)

def normalize_car_references(message: str) -> str:
    """
    Convenience function to normalize car references in messages.
    
    Args:
        message: Message with potential car references
    
    Returns:
        Message with normalized car references
    """
    return text_sanitizer.sanitize_message(message, normalize_possessives=True, normalize_references=True)
