"""
Raga Grammar Validation - Example Usage and Testing

Demonstrates the complete pipeline:
1. Tonic (Sa) detection from audio
2. Real-time pitch extraction and swara quantization  
3. Grammar validation against raga rules
4. Educational feedback generation

Test with audio files from the drive-download folder.
"""

import os
import sys
import time
import numpy as np

# Add parent directory to import tonic_sa_detection
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from raga_grammar import (
    RAGA_DB, list_available_ragas,
    RealTimeGrammarPipeline, 
    FeedbackGenerator, Language
)

def demonstrate_raga_database():
    """Show available ragas and their properties"""
    print("🎵 RAGA DATABASE DEMO")
    print("=" * 50)
    
    available_ragas = list_available_ragas()
    print(f"Available ragas: {len(available_ragas)}")
    
    # Show first 5 ragas with details
    for i, raga_name in enumerate(available_ragas[:5]):
        raga_info = RAGA_DB[raga_name]
        print(f"\n{i+1}. {raga_name}")
        print(f"   Arohana:  {' → '.join(raga_info.arohana)}")
        print(f"   Avarohana: {' → '.join(raga_info.avarohana)}")
        print(f"   Parent Mela: {raga_info.parent_mela}")
        print(f"   Vakra: Arohana={raga_info.is_vakra_arohana}, Avarohana={raga_info.is_vakra_avarohana}")
        if raga_info.varja_arohana:
            print(f"   Forbidden in ascent: {list(raga_info.varja_arohana)}")
        if raga_info.varja_avarohana:
            print(f"   Forbidden in descent: {list(raga_info.varja_avarohana)}")

def test_swara_quantization():
    """Test swara quantization with known frequencies"""
    print("\n🎼 SWARA QUANTIZATION TEST")
    print("=" * 50)
    
    from raga_grammar.swara_quantizer import SwaraQuantizer
    
    # Test with standard Sa = 260 Hz
    sa_freq = 260.0
    quantizer = SwaraQuantizer(sa_freq)
    
    # Test known frequency → swara mappings
    test_frequencies = [
        (260.0, "Sa"),      # Perfect Sa
        (292.5, "Ri2"),     # Ri2 = 9/8 × Sa
        (325.0, "Ga2"),     # Ga2 = 5/4 × Sa  
        (346.7, "Ma1"),     # Ma1 = 4/3 × Sa
        (390.0, "Pa"),      # Pa = 3/2 × Sa
        (433.3, "Dha2"),    # Dha2 = 5/3 × Sa
        (487.5, "Ni2"),     # Ni2 = 15/8 × Sa
        (520.0, "Sa")       # Upper Sa = 2 × Sa
    ]
    
    print(f"Sa frequency: {sa_freq} Hz\n")
    print("Frequency →  Swara   Octave  Deviation")
    print("-" * 40)
    
    for freq, expected in test_frequencies:
        result = quantizer.to_swara(freq)
        if result:
            print(f"{freq:8.1f} → {result.swara:>6} {result.octave:>7}  {result.cents_deviation:+7.1f}¢")
        else:
            print(f"{freq:8.1f} → {'None':>6} {'—':>7}  {'—':>7}")

def test_grammar_validation():
    """Test grammar validation with known sequences"""
    print("\n🎯 GRAMMAR VALIDATION TEST")
    print("=" * 50)
    
    from raga_grammar.grammar_validator import RagaGrammarValidator
    from raga_grammar.swara_quantizer import SwaraResult
    
    # Test Kambhoji raga (Ni is varja in arohana, allowed in avarohana)
    validator = RagaGrammarValidator("Kāṁbhōji")
    
    print("Testing Kambhoji raga:")
    print("Arohana:  Sa → Ri2 → Ga2 → Ma1 → Pa → Dha2 → Sa")
    print("Avarohana: Sa → Ni2 → Dha2 → Pa → Ma1 → Ga2 → Ri2 → Sa")
    print("Ni2 is varja (forbidden) in arohana\n")
    
    # Simulate correct arohana sequence
    correct_arohana = ["Sa", "Ri2", "Ga2", "Ma1", "Pa", "Dha2", "Sa"]
    
    print("Testing correct arohana:")
    for i, swara in enumerate(correct_arohana):
        swara_result = SwaraResult(swara=swara, octave=0, cents_deviation=0, confidence=1.0)
        event = validator.validate_swara(swara_result, timestamp_ms=i*500)
        
        status = "✓" if event.error_type is None else "✗"
        print(f"  {status} {swara} ({event.direction.value})")
    
    # Test forbidden note (Ni2 in ascent)
    print(f"\nTesting forbidden note (Ni2 in ascent):")
    validator.reset()  # Reset for new test
    
    # Set up ascending context
    for swara in ["Sa", "Ri2", "Ga2"]:
        swara_result = SwaraResult(swara=swara, octave=0, cents_deviation=0, confidence=1.0)
        validator.validate_swara(swara_result)
    
    # Now try forbidden Ni2
    ni_result = SwaraResult(swara="Ni2", octave=0, cents_deviation=0, confidence=1.0)
    event = validator.validate_swara(ni_result)
    
    status = "✓" if event.error_type is None else "✗"
    print(f"  {status} Ni2 ({event.direction.value}) - Error: {event.error_type}")

def test_feedback_generation():
    """Test feedback message generation"""
    print("\n💬 FEEDBACK GENERATION TEST")
    print("=" * 50)
    
    from raga_grammar.feedback_generator import FeedbackGenerator, Language
    from raga_grammar.grammar_validator import ValidationEvent, ErrorType, Direction
    
    feedback_gen = FeedbackGenerator(Language.ENGLISH)
    
    # Create a test validation event (forbidden note error)
    error_event = ValidationEvent(
        timestamp_ms=1500,
        swara="Ni2",
        octave=0,
        frequency_hz=487.5,
        cents_deviation=+5.2,
        error_type=ErrorType.FORBIDDEN_NOTE,
        direction=Direction.ASCENDING,
        description="Forbidden note detected"
    )
    
    feedback = feedback_gen.generate_feedback(error_event, "Kāṁbhōji")
    
    print("Generated feedback for forbidden note error:")
    print(f"Title: {feedback['title']}")
    print(f"Message: {feedback['message']}")
    print(f"Suggestion: {feedback['suggestion']}")
    print(f"Explanation: {feedback['explanation']}")
    if 'correction' in feedback:
        print(f"Correction: {feedback['correction']}")

def analyze_audio_file(file_path: str, raga_name: str):
    """Analyze a real audio file"""
    print(f"\n🎤 AUDIO ANALYSIS: {os.path.basename(file_path)}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        # Initialize pipeline
        pipeline = RealTimeGrammarPipeline(raga_name)
        
        # Analyze file
        report = pipeline.analyze_file(file_path, verbose=True)
        
        if report['success']:
            print(f"\n📊 SUMMARY REPORT")
            print(f"  Sa frequency: {report['detected_sa_hz']:.2f} Hz")
            print(f"  Duration: {report['duration_sec']:.1f} seconds")
            print(f"  Total errors: {report['total_errors']}")
            print(f"  Error rate: {report['error_rate']:.1%}")
            
            # Show first few errors with feedback
            error_events = report['error_events'][:3]  # First 3 errors
            if error_events:
                print(f"\n🔍 FIRST {len(error_events)} ERRORS WITH FEEDBACK:")
                
                feedback_gen = FeedbackGenerator()
                for i, event in enumerate(error_events, 1):
                    feedback = feedback_gen.generate_feedback(event, raga_name)
                    print(f"\n  Error {i} @ {event.timestamp_ms/1000:.1f}s:")
                    print(f"    {feedback['title']}: {feedback['message']}")
                    print(f"    Suggestion: {feedback['suggestion']}")
        else:
            print(f"❌ Analysis failed: {report.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all demonstrations"""
    print("🎵 CARNATIC RAGA GRAMMAR VALIDATION SYSTEM")
    print("🔬 COMPREHENSIVE TESTING AND DEMONSTRATION")
    print("=" * 70)
    
    # Show available components
    demonstrate_raga_database()
    test_swara_quantization()
    test_grammar_validation()
    test_feedback_generation()
    
    # Test with real audio files if available
    audio_dir = "../drive-download-20260211T121423Z-3-001"
    
    # Look for common audio file extensions
    audio_files = []
    if os.path.exists(audio_dir):
        for ext in ['.mp3', '.wav', '.flac', '.ogg']:
            import glob
            audio_files.extend(glob.glob(f"{audio_dir}/*{ext}"))
    
    if audio_files:
        print(f"\n🎤 FOUND {len(audio_files)} AUDIO FILES FOR TESTING")
        print("=" * 50)
        
        # Test with first audio file
        test_file = audio_files[0]
        test_raga = "Śankarābharaṇaṁ"  # Common raga for testing
        
        print(f"Testing with: {os.path.basename(test_file)}")
        print(f"Assumed raga: {test_raga}")
        
        analyze_audio_file(test_file, test_raga)
    else:
        print(f"\n⚠️  NO AUDIO FILES FOUND")
        print(f"Place audio files in: {audio_dir}")
        print("Supported formats: MP3, WAV, FLAC, OGG")
    
    print(f"\n✅ DEMONSTRATION COMPLETE!")
    print("The raga grammar validation system is ready for use.")

if __name__ == "__main__":
    main()