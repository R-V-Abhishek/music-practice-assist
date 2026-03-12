"""
Feedback Generator - Human-Readable Error Messages

Converts grammar validation errors into educational feedback for students.
Supports multiple languages and provides constructive correction suggestions.
"""

from typing import Dict, List, Optional
from enum import Enum

from .grammar_validator import ValidationEvent, ErrorType, Direction
from .raga_database import get_raga_info

class Language(Enum):
    """Supported feedback languages"""
    ENGLISH = "english"
    KANNADA = "kannada"
    TAMIL = "tamil"
    TELUGU = "telugu"

class FeedbackGenerator:
    """
    Generate educational feedback messages from validation events.
    
    Provides context-aware explanations of raga grammar rules and
    specific suggestions for correcting detected errors.
    """
    
    def __init__(self, language: Language = Language.ENGLISH):
        """
        Initialize feedback generator.
        
        Args:
            language: Language for feedback messages
        """
        self.language = language
        
        # Error message templates by language
        self.templates = {
            Language.ENGLISH: {
                ErrorType.FORBIDDEN_NOTE: {
                    'title': "Forbidden Note Detected",
                    'message': "Note {swara} is varja (forbidden) in raga {raga} during {direction}",
                    'suggestion': "In {raga}, note {swara} should only be used in {allowed_direction}",
                    'explanation': "This raga's {direction} pattern ({sequence_name}) does not include {swara}"
                },
                ErrorType.SEQUENCE_VIOLATION: {
                    'title': "Sequence Order Violation", 
                    'message': "Note {swara} breaks {raga}'s {sequence_name} pattern",
                    'suggestion': "Expected {expected_swara} next in {sequence_name}",
                    'explanation': "The {sequence_name} of {raga} follows: {sequence}"
                },
                ErrorType.WRONG_DIRECTION: {
                    'title': "Directional Error",
                    'message': "Note {swara} sung in wrong direction for {raga}",
                    'suggestion': "Note {swara} should be sung while {correct_direction} in this raga",
                    'explanation': "Check the Arohana and Avarohana patterns for proper usage"
                },
                ErrorType.UNEXPECTED_JUMP: {
                    'title': "Unexpected Note Jump",
                    'message': "Large interval jump to {swara} may violate {raga} grammar",
                    'suggestion': "Consider using intermediate notes: {intermediate_notes}",
                    'explanation': "Carnatic music typically uses stepwise motion with occasional sanctioned jumps"
                }
            },
            # Add support for Indian languages (basic templates)
            Language.KANNADA: {
                ErrorType.FORBIDDEN_NOTE: {
                    'title': "ನಿಷಿದ್ಧ ಸ್ವರ ಪತ್ತೆಯಾಗಿದೆ",
                    'message': "{swara} ಸ್ವರವು {raga} ರಾಗದಲ್ಲಿ {direction} ಸಮಯದಲ್ಲಿ ವರ್ಜ್ಯ (ನಿಷಿದ್ಧ)",
                    'suggestion': "{raga}ದಲ್ಲಿ {swara} ಸ್ವರವನ್ನು {allowed_direction}ದಲ್ಲಿ ಮಾತ್ರ ಬಳಸಬೇಕು",
                    'explanation': "ಈ ರಾಗದ {direction} ಕ್ರಮ ({sequence_name}) {swara} ಸ್ವರವನ್ನು ಒಳಗೊಂಡಿಲ್ಲ"
                }
            }
        }
    
    def generate_feedback(self, event: ValidationEvent, raga_name: str) -> Dict[str, str]:
        """
        Generate comprehensive feedback for a validation event.
        
        Args:
            event: ValidationEvent with error details
            raga_name: Name of raga being validated
            
        Returns:
            Dict with title, message, suggestion, and explanation
        """
        if event.error_type is None:
            return self._generate_positive_feedback(event, raga_name)
        
        templates = self.templates.get(self.language, self.templates[Language.ENGLISH])
        error_template = templates.get(event.error_type, {})
        
        if not error_template:
            return self._generate_generic_error(event, raga_name)
        
        # Get raga information for context
        raga_info = get_raga_info(raga_name)
        
        # Format template with event data
        feedback = {}
        
        # Common format parameters
        format_params = {
            'swara': event.swara,
            'raga': raga_name,
            'direction': event.direction.value,
            'expected_swara': event.expected_swara or "unknown",
        }
        
        # Add raga-specific context
        if raga_info:
            format_params.update({
                'sequence': ' -> '.join(raga_info.arohana) if event.direction == Direction.ASCENDING 
                           else ' -> '.join(raga_info.avarohana),
                'sequence_name': 'Arohana' if event.direction == Direction.ASCENDING else 'Avarohana',
                'arohana': ' -> '.join(raga_info.arohana),
                'avarohana': ' -> '.join(raga_info.avarohana)
            })
            
            # Add error-specific context
            if event.error_type == ErrorType.FORBIDDEN_NOTE:
                allowed_direction = self._get_allowed_direction(event.swara, raga_info)
                format_params['allowed_direction'] = allowed_direction.value if allowed_direction else "neither direction"
        
        # Format each feedback component
        for key, template in error_template.items():
            try:
                feedback[key] = template.format(**format_params)
            except KeyError as e:
                feedback[key] = f"Template error: missing {e}"
        
        # Add corrective suggestions
        feedback['correction'] = self._generate_correction_suggestion(event, raga_info)
        
        # Add pitch accuracy feedback if relevant
        if abs(event.cents_deviation) > 10:
            pitch_feedback = self._generate_pitch_feedback(event)
            feedback['pitch_accuracy'] = pitch_feedback
        
        return feedback
    
    def _generate_positive_feedback(self, event: ValidationEvent, raga_name: str) -> Dict[str, str]:
        """Generate encouraging feedback for correct notes"""
        return {
            'title': "Correct Note",
            'message': f"✓ {event.swara} sung correctly in {raga_name}",
            'suggestion': "Good! Continue with proper raga grammar",
            'explanation': f"This note fits the {event.direction.value} pattern well"
        }
    
    def _generate_generic_error(self, event: ValidationEvent, raga_name: str) -> Dict[str, str]:
        """Generate generic feedback for unknown error types"""
        return {
            'title': "Raga Grammar Issue",
            'message': f"Issue detected with {event.swara} in {raga_name}",
            'suggestion': "Review the Arohana and Avarohana patterns for this raga",
            'explanation': "Consult your teacher or reference materials"
        }
    
    def _get_allowed_direction(self, swara: str, raga_info) -> Optional[Direction]:
        """Determine which direction(s) allow a given swara"""
        in_arohana = swara in raga_info.arohana
        in_avarohana = swara in raga_info.avarohana
        
        if in_arohana and not in_avarohana:
            return Direction.ASCENDING
        elif in_avarohana and not in_arohana:
            return Direction.DESCENDING
        elif in_arohana and in_avarohana:
            return None  # Allowed in both directions
        else:
            return None  # Not allowed in either direction
    
    def _generate_correction_suggestion(self, event: ValidationEvent, raga_info) -> str:
        """Generate specific correction suggestions based on error type"""
        if not raga_info:
            return "Refer to standard raga reference materials"
        
        if event.error_type == ErrorType.FORBIDDEN_NOTE:
            # Suggest nearby allowed notes
            if event.direction == Direction.ASCENDING:
                allowed_notes = [s for s in raga_info.arohana if s not in raga_info.varja_arohana]
            else:
                allowed_notes = [s for s in raga_info.avarohana if s not in raga_info.varja_avarohana]
            
            return f"Try using these allowed notes instead: {', '.join(allowed_notes[:5])}"
        
        elif event.error_type == ErrorType.SEQUENCE_VIOLATION:
            if event.expected_swara:
                return f"Continue with {event.expected_swara} to follow proper sequence"
            else:
                return "Return to Sa and restart the sequence"
        
        elif event.error_type == ErrorType.WRONG_DIRECTION:
            return f"Practice the proper {event.direction.value} sequence: {' -> '.join(raga_info.arohana if event.direction == Direction.ASCENDING else raga_info.avarohana)}"
        
        return "Practice slowly to internalize the raga grammar"
    
    def _generate_pitch_feedback(self, event: ValidationEvent) -> str:
        """Generate feedback about pitch accuracy (apashruthi)"""
        deviation = event.cents_deviation
        
        if abs(deviation) <= 5:
            return f"Excellent pitch accuracy ({deviation:+.1f} cents)"
        elif abs(deviation) <= 15:
            return f"Good pitch ({deviation:+.1f} cents from perfect)"
        elif abs(deviation) <= 25:
            direction = "sharp" if deviation > 0 else "flat"
            return f"Slightly {direction} ({abs(deviation):.1f} cents) - adjust tuning"
        else:
            direction = "sharp" if deviation > 0 else "flat"
            return f"Significantly {direction} ({abs(deviation):.1f} cents) - needs correction"
    
    def generate_session_summary(self, events: List[ValidationEvent], 
                               raga_name: str) -> Dict[str, str]:
        """
        Generate summary feedback for an entire practice session.
        
        Args:
            events: List of all validation events
            raga_name: Raga being practiced
            
        Returns:
            Dict with session-level feedback and recommendations
        """
        if not events:
            return {'summary': 'No performance data available'}
        
        # Calculate statistics
        total_notes = len(events)
        errors = [e for e in events if e.error_type is not None]
        error_rate = len(errors) / total_notes if total_notes > 0 else 0
        
        # Error type breakdown
        error_types = {}
        for error in errors:
            error_type = error.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Calculate average pitch accuracy
        deviations = [abs(e.cents_deviation) for e in events if e.cents_deviation is not None]
        avg_deviation = np.mean(deviations) if deviations else 0
        
        # Generate summary
        summary = {
            'session_title': f"{raga_name} Practice Session Summary",
            'performance_score': max(0, 100 - (error_rate * 100)),  # Simple scoring
            'total_notes': total_notes,
            'error_count': len(errors),
            'error_rate': f"{error_rate:.1%}",
            'pitch_accuracy': f"±{avg_deviation:.1f} cents average",
        }
        
        # Performance assessment
        if error_rate < 0.05:
            summary['assessment'] = "Excellent! Your raga grammar is very strong."
        elif error_rate < 0.15:
            summary['assessment'] = "Good performance with minor grammar issues."
        elif error_rate < 0.30:
            summary['assessment'] = "Moderate performance. Focus on specific problem areas."
        else:
            summary['assessment'] = "Needs improvement. Review fundamentals."
        
        # Specific recommendations
        recommendations = []
        if 'forbidden_note' in error_types:
            recommendations.append("Review which notes are varja (forbidden) in each direction")
        if 'sequence_violation' in error_types:
            recommendations.append("Practice Arohana and Avarohana sequences slowly")
        if avg_deviation > 20:
            recommendations.append("Work on pitch accuracy with tanpura reference")
        
        summary['recommendations'] = recommendations
        summary['most_common_error'] = max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        
        return summary

# Import numpy for calculations
import numpy as np