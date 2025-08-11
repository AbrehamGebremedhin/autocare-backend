# User Message Sanitization

## Overview

The AutoCare backend now includes automatic user message sanitization that normalizes possessive references to vehicles. When users input messages containing "my car's", "my vehicle's", etc., these are automatically converted to more generic references like "the car's", "the vehicle's".

## Features

### Text Sanitization

- **Possessive Normalization**: Converts "my car's" → "the car's"
- **Reference Normalization**: Converts "my car" → "the car" (when not followed by an apostrophe)
- **Vehicle Type Support**: Works with car, vehicle, truck, SUV, motorcycle, bike
- **Case Insensitive**: Handles any case combination (e.g., "My Car's", "MY CAR'S")
- **Selective**: Only affects vehicle-related references, preserves other "my" references

### Supported Patterns

The sanitizer handles the following vehicle-related patterns:

#### Possessive Forms

- `my car's` → `the car's`
- `my vehicle's` → `the vehicle's`
- `my truck's` → `the truck's`
- `my SUV's` → `the SUV's`
- `my motorcycle's` → `the motorcycle's`
- `my bike's` → `the bike's`

#### Non-Possessive Forms

- `my car` → `the car`
- `my vehicle` → `the vehicle`
- `my truck` → `the truck`
- `my SUV` → `the SUV`
- `my motorcycle` → `the motorcycle`
- `my bike` → `the bike`

### Integration Points

The sanitization is integrated into:

- `ChatMessageRequest` validation (for general chat messages)
- `ChatSessionMessageRequest` validation (for session-based chat)

### Examples

```
Input:  "my car's engine is making noise"
Output: "the car's engine is making noise"

Input:  "My SUV's brakes are squeaking"
Output: "the SUV's brakes are squeaking"

Input:  "I think my truck's transmission failed"
Output: "I think the truck's transmission failed"

Input:  "my car won't start"
Output: "the car won't start"

# Non-vehicle references are preserved:
Input:  "my house needs repair"
Output: "my house needs repair" (unchanged)
```

### Additional Features

- **Car Mention Detection**: Can identify if a message is car-related
- **Contraction Normalization**: Optional normalization of contractions (won't → will not)
- **Apostrophe Safety**: Handles different apostrophe characters (', ') safely in regex patterns
- **JSON Safe Output**: All sanitized text is safe for JSON serialization
- **Error Handling**: Graceful fallback to original message if sanitization fails
- **Logging**: Logs sanitization actions for monitoring

## Usage

### Direct Usage

```python
from app.utils.text_sanitizer import sanitize_user_message

# Basic sanitization
sanitized = sanitize_user_message("my car's engine is loud")
# Result: "the car's engine is loud"
```

### API Integration

The sanitization is automatically applied to all incoming chat messages through the Pydantic validators:

```python
# When a request comes in:
request = ChatMessageRequest(
    user_id="user123",
    message="my car's engine won't start",  # This gets sanitized
    context=None
)
# request.message will be: "the car's engine won't start"
```

## Files Added/Modified

### New Files

- `app/utils/text_sanitizer.py` - Main sanitization utility
- `tests/unit/test_text_sanitizer.py` - Unit tests for sanitization
- `tests/unit/test_chat_message_sanitization.py` - Integration tests

### Modified Files

- `app/api/v1/chat_route.py` - Integrated sanitization into message validators

## Testing

Comprehensive test coverage includes:

- Basic possessive normalization
- Multiple vehicle types
- Case sensitivity
- Mixed car and non-car references
- Edge cases and error handling
- Integration with Pydantic validators

Run tests with:

```bash
python -m pytest tests/unit/test_text_sanitizer.py -v
python -m pytest tests/unit/test_chat_message_sanitization.py -v
```

## Technical Notes

### Regex Pattern Safety

The regex patterns used in the sanitizer are designed to be safe from injection attacks and handle special characters properly:

- **Apostrophe Handling**: Uses character classes `[\'\u2019]` to match both ASCII apostrophes (`'`) and Unicode right single quotation marks (`'`)
- **Word Boundaries**: Uses `\b` to ensure exact word matching and prevent partial matches
- **Negative Lookahead**: Uses `(?![\'\u2019])` to avoid conflicts between possessive and non-possessive patterns
- **JSON Safety**: All output is guaranteed to be JSON-serializable without escaping issues

### Pattern Examples

```regex
# Possessive patterns (more specific, processed first)
r"\bmy\s+car[\'\u2019]?s\b"  → "the car's"

# Reference patterns (processed second)
r"\bmy\s+car(?![\'\u2019])\b"  → "the car"
```

This ensures "my car's engine" becomes "the car's engine" and "my car is broken" becomes "the car is broken", while avoiding double-processing.
