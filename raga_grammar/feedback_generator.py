"""
Feedback Generator - Human-Readable Error Messages

Converts grammar validation errors into educational feedback for students.
Supports multiple languages and provides constructive correction suggestions.
"""

import random
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .grammar_validator import ValidationEvent, ErrorType, Direction
from .raga_database import get_raga_info, SWARA_CENTS


_PRAYOGA_CONGRATS = [
    "Excellent! ",
    "Well done! ",
    "Beautiful! ",
    "Perfect phrasing! ",
    "Superb! ",
]


def _nearest_allowed(sung: str, candidates: List[str]) -> Optional[Tuple[str, int]]:
    """Return (best_allowed_swara, cents_diff) closest to `sung`.
    cents_diff > 0 means sung is flat (needs to go higher).
    cents_diff < 0 means sung is sharp (needs to go lower).
    """
    sung_c = SWARA_CENTS.get(sung)
    if sung_c is None or not candidates:
        return None
    best: Optional[str] = None
    best_diff = 0
    best_dist = float('inf')
    for s in candidates:
        sc = SWARA_CENTS.get(s)
        if sc is None:
            continue
        dist = abs(sc - sung_c)
        if dist < best_dist:
            best_dist = dist
            best = s
            best_diff = sc - sung_c
    return (best, best_diff) if best is not None else None

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
                    'message': "{swara} is not part of raga {raga}{direction_context}",
                    'suggestion': "{allowed_suggestion}",
                    'explanation': "{detailed_explanation}"
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
                },
                ErrorType.FORBIDDEN_PHRASE: {
                    'title': "Forbidden Skip Detected",
                    'message': "You skipped directly from {from_swara} to {to_swara} in {raga}. The vakra detour is required.",
                    'suggestion': "Include the required intermediate swara between {from_swara} and {to_swara}",
                    'explanation': "Raga {raga} uses a vakra (zigzag) pattern that mandates passing through an intermediate note"
                }
            },
            # Add support for Indian languages (basic templates)
            Language.KANNADA: {
                ErrorType.FORBIDDEN_NOTE: {
                    'title': "ನಿಷಿದ್ಧ ಸ್ವರ ಪತ್ತೆಯಾಗಿದೆ",
                    'message': "{swara} ಸ್ವರವು {raga} ರಾಗದಲ್ಲಿ{direction_context} ವರ್ಜ್ಯ (ನಿಷಿದ್ಧ)",
                    'suggestion': "{allowed_suggestion}",
                    'explanation': "{detailed_explanation}"
                },
                ErrorType.FORBIDDEN_PHRASE: {
                    'title': "ನಿಷಿದ್ಧ ಸ್ವರ ಜಿಗಿತ ಪತ್ತೆಯಾಗಿದೆ",
                    'message': "{raga} ರಾಗದಲ್ಲಿ {from_swara} ನಿಂದ {to_swara} ಗೆ ನೇರವಾಗಿ ಜಿಗಿದಿದೆ. ವಕ್ರ ಮಾರ್ಗ ಅಗತ್ಯ.",
                    'suggestion': "{from_swara} ಮತ್ತು {to_swara} ನಡುವೆ ಅಗತ್ಯ ಮಧ್ಯಂತರ ಸ್ವರವನ್ನು ಸೇರಿಸಿ",
                    'explanation': "{raga} ರಾಗವು ವಕ್ರ ಮಾದರಿಯನ್ನು ಬಳಸುತ್ತದೆ"
                }
            },
            Language.TAMIL: {
                ErrorType.FORBIDDEN_NOTE: {
                    'title': "தடைசெய்யப்பட்ட ஸ்வரம் கண்டறியப்பட்டது",
                    'message': "{swara} ஸ்வரம் {raga} ராகத்தில்{direction_context} வர்ஜ்ய (தடை)",
                    'suggestion': "{allowed_suggestion}",
                    'explanation': "{detailed_explanation}"
                },
                ErrorType.FORBIDDEN_PHRASE: {
                    'title': "தடைசெய்யப்பட்ட தாவல் கண்டறியப்பட்டது",
                    'message': "{raga} ராகத்தில் {from_swara} லிருந்து {to_swara} க்கு நேரடியாக தாவியுள்ளீர்கள். வக்ர பாதை அவசியம்.",
                    'suggestion': "{from_swara} மற்றும் {to_swara} இடையே தேவையான இடைநிலை ஸ்வரத்தை சேர்க்கவும்",
                    'explanation': "{raga} ராகம் வக்ர (வளைவு) முறையைப் பயன்படுத்துகிறது"
                }
            },
            Language.TELUGU: {
                ErrorType.FORBIDDEN_NOTE: {
                    'title': "నిషిద్ధ స్వరం గుర్తించబడింది",
                    'message': "{swara} స్వరం {raga} రాగంలో{direction_context} వర్జ్య (నిషిద్ధం)",
                    'suggestion': "{allowed_suggestion}",
                    'explanation': "{detailed_explanation}"
                },
                ErrorType.FORBIDDEN_PHRASE: {
                    'title': "నిషిద్ధ స్వర దూకం గుర్తించబడింది",
                    'message': "{raga} రాగంలో {from_swara} నుండి {to_swara} కు నేరుగా దూకారు. వక్ర మార్గం అవసరం.",
                    'suggestion': "{from_swara} మరియు {to_swara} మధ్య అవసరమైన మధ్యంతర స్వరాన్ని చేర్చండి",
                    'explanation': "{raga} రాగం వక్ర (మెలికలు) నమూనాను ఉపయోగిస్తుంది"
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
                in_arohana = event.swara in set(raga_info.arohana)
                in_avarohana = event.swara in set(raga_info.avarohana)

                # Compute nearest correct swara + pitch correction
                if event.direction == Direction.ASCENDING:
                    allowed_pool = [s for s in raga_info.arohana if s not in raga_info.varja_arohana]
                elif event.direction == Direction.DESCENDING:
                    allowed_pool = [s for s in raga_info.avarohana if s not in raga_info.varja_avarohana]
                else:
                    allowed_pool = list(
                        (set(raga_info.arohana) | set(raga_info.avarohana))
                        - raga_info.varja_arohana - raga_info.varja_avarohana
                    )

                nearest = _nearest_allowed(event.swara, allowed_pool)
                if nearest:
                    correct_swara, cent_diff = nearest
                    format_params['correct_swara'] = correct_swara
                    format_params['cents_diff'] = abs(cent_diff)
                    if cent_diff > 0:
                        pitch_hint = (
                            f"You are singing {event.swara} which is {abs(cent_diff)} cents too low. "
                            f"Raise your pitch by {abs(cent_diff)} cents to reach {correct_swara}."
                        )
                    elif cent_diff < 0:
                        pitch_hint = (
                            f"You are singing {event.swara} which is {abs(cent_diff)} cents too high. "
                            f"Lower your pitch by {abs(cent_diff)} cents to reach {correct_swara}."
                        )
                    else:
                        pitch_hint = f"Use {correct_swara} here instead of {event.swara}."
                    format_params['pitch_hint'] = pitch_hint
                else:
                    format_params['correct_swara'] = ''
                    format_params['cents_diff'] = 0
                    format_params['pitch_hint'] = ''

                if not in_arohana and not in_avarohana:
                    format_params['direction_context'] = ''
                    ph = format_params.get('pitch_hint', '')
                    format_params['allowed_suggestion'] = (
                        f"{ph} Raga {raga_name} does not use {event.swara} at all."
                        if ph else
                        f"Raga {raga_name} does not use {event.swara} at all. Use only the swaras in its scale"
                    )
                    format_params['detailed_explanation'] = (
                        f"Arohana: {' \u2192 '.join(raga_info.arohana)}  |  "
                        f"Avarohana: {' \u2192 '.join(raga_info.avarohana)}"
                    )
                elif in_arohana and not in_avarohana:
                    format_params['direction_context'] = ' while descending'
                    ph = format_params.get('pitch_hint', '')
                    format_params['allowed_suggestion'] = (
                        f"{event.swara} is only in the Arohana of {raga_name}. Avoid it while descending. {ph}"
                        if ph else
                        f"{event.swara} appears only in the Arohana (ascending scale) of {raga_name}. Avoid it while descending"
                    )
                    format_params['detailed_explanation'] = (
                        f"Avarohana: {' \u2192 '.join(raga_info.avarohana)}"
                    )
                elif in_avarohana and not in_arohana:
                    format_params['direction_context'] = ' while ascending'
                    ph = format_params.get('pitch_hint', '')
                    format_params['allowed_suggestion'] = (
                        f"{event.swara} is only in the Avarohana of {raga_name}. Avoid it while ascending. {ph}"
                        if ph else
                        f"{event.swara} appears only in the Avarohana (descending scale) of {raga_name}. Avoid it while ascending"
                    )
                    format_params['detailed_explanation'] = (
                        f"Arohana: {' \u2192 '.join(raga_info.arohana)}"
                    )
                else:
                    format_params['direction_context'] = ''
                    format_params['allowed_suggestion'] = format_params.get('pitch_hint') or 'Check your intonation'
                    format_params['detailed_explanation'] = (
                        f"Arohana: {' \u2192 '.join(raga_info.arohana)}  |  "
                        f"Avarohana: {' \u2192 '.join(raga_info.avarohana)}"
                    )
        
        # Fallback: ensure FORBIDDEN_NOTE placeholders are always set
        if event.error_type == ErrorType.FORBIDDEN_NOTE:
            format_params.setdefault('direction_context', '')
            format_params.setdefault('allowed_suggestion', 'This note may not belong in this raga')
            format_params.setdefault('detailed_explanation', 'Review the arohana and avarohana of this raga')

        # Add forbidden-phrase-specific context (from_swara / to_swara from description)
        if event.error_type == ErrorType.FORBIDDEN_PHRASE:
            # Description has format: "Forbidden skip A→B in RagaName"
            # Parse from_swara and to_swara from the description
            desc = event.description
            if '\u2192' in desc:
                skip_part = desc.split('Forbidden skip ')[-1].split(' in ')[0]
                parts = skip_part.split('\u2192')
                if len(parts) == 2:
                    format_params['from_swara'] = parts[0]
                    format_params['to_swara'] = parts[1]
            format_params.setdefault('from_swara', event.swara)
            format_params.setdefault('to_swara', event.swara)
        
        # Format each feedback component
        for key, template in error_template.items():
            try:
                feedback[key] = template.format(**format_params)
            except KeyError as e:
                feedback[key] = f"Template error: missing {e}"
        
        # Add corrective suggestions
        feedback['correction'] = self._generate_correction_suggestion(event, raga_info)

        # Attach pitch correction hint directly if computed
        if event.error_type == ErrorType.FORBIDDEN_NOTE and format_params.get('pitch_hint'):
            feedback['pitch_correction'] = format_params['pitch_hint']
            feedback['correct_swara'] = format_params.get('correct_swara', '')
            feedback['cents_diff'] = format_params.get('cents_diff', 0)

        # Add pitch accuracy feedback if relevant
        if abs(event.cents_deviation) > 10:
            pitch_feedback = self._generate_pitch_feedback(event)
            feedback['pitch_accuracy'] = pitch_feedback
        
        return feedback
    
    def generate_phrase_feedback(self, phrase: List[str], raga_name: str, is_positive: bool) -> Dict[str, str]:
        """Generate feedback for phrase-level events.

        Args:
            phrase: List of swara names forming the phrase
            raga_name: Name of the raga
            is_positive: True for characteristic prayoga match, False for forbidden skip

        Returns:
            Dict with title, message, suggestion keys
        """
        phrase_str = ' \u2192 '.join(phrase)
        if is_positive:
            congrats = random.choice(_PRAYOGA_CONGRATS)
            return {
                'title': f"{congrats}Prayoga Recognized!",
                'message': f"You sang the characteristic phrase {phrase_str} — a signature prayoga of {raga_name}!",
                'suggestion': f"This is a hallmark phrase of {raga_name}. Keep building on it!",
            }
        # Negative: forbidden skip — delegate to template
        templates = self.templates.get(self.language, self.templates[Language.ENGLISH])
        tpl = templates.get(ErrorType.FORBIDDEN_PHRASE, {})
        params = {'from_swara': phrase[0], 'to_swara': phrase[-1], 'raga': raga_name}
        result: Dict[str, str] = {}
        for key, template in tpl.items():
            try:
                result[key] = template.format(**params)
            except KeyError:
                result[key] = template
        return result

    def _generate_positive_feedback(self, event: ValidationEvent, raga_name: str) -> Dict[str, str]:
        """Generate encouraging feedback for correct notes"""
        if event.matched_phrase is not None:
            phrase_str = ' \u2192 '.join(event.matched_phrase)
            congrats = random.choice(_PRAYOGA_CONGRATS)
            return {
                'title': f"{congrats}Prayoga Recognized!",
                'message': f"You sang the characteristic phrase: {phrase_str}",
                'suggestion': f"This is a signature prayoga of {raga_name} \u2014 keep building on it!",
                'explanation': f"{phrase_str} is a well-known melodic phrase (prayoga) of {raga_name}."
            }
        return {
            'title': "Correct Note",
            'message': f"\u2713 {event.swara} sung correctly in {raga_name}",
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
        
        if event.error_type == ErrorType.FORBIDDEN_PHRASE:
            return "Follow the vakra (zigzag) pattern — do not skip the intermediate swara"
        
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
        
        # Phrase-level stats
        forbidden_phrase_count = error_types.get('forbidden_phrase', 0)
        characteristic_phrases = [e for e in events if e.matched_phrase is not None]
        characteristic_phrase_count = len(characteristic_phrases)
        
        # Calculate average pitch accuracy
        deviations = [abs(e.cents_deviation) for e in events if e.cents_deviation is not None]
        avg_deviation = np.mean(deviations) if deviations else 0
        
        # Scoring: base 100 - error_rate*100, forbidden phrases deduct 5 each,
        # characteristic phrase matches add 3 bonus each
        base_score = 100 - (error_rate * 100)
        phrase_penalty = forbidden_phrase_count * 5
        phrase_bonus = characteristic_phrase_count * 3
        performance_score = max(0, min(100, base_score - phrase_penalty + phrase_bonus))
        
        # Generate summary
        summary = {
            'session_title': f"{raga_name} Practice Session Summary",
            'performance_score': performance_score,
            'total_notes': total_notes,
            'error_count': len(errors),
            'error_rate': f"{error_rate:.1%}",
            'pitch_accuracy': f"±{avg_deviation:.1f} cents average",
            'forbidden_phrase_count': forbidden_phrase_count,
            'characteristic_phrases_completed': characteristic_phrase_count,
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
        if 'forbidden_phrase' in error_types:
            recommendations.append("Practice the vakra patterns slowly — avoid skipping the required zigzag swaras")
        if 'sequence_violation' in error_types:
            recommendations.append("Practice Arohana and Avarohana sequences slowly")
        if avg_deviation > 20:
            recommendations.append("Work on pitch accuracy with tanpura reference")
        if characteristic_phrase_count > 0:
            recommendations.append(f"Great prayoga usage! You completed {characteristic_phrase_count} characteristic phrase(s)")
        
        summary['recommendations'] = recommendations
        summary['most_common_error'] = max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        
        return summary

# Import numpy for calculations
import numpy as np