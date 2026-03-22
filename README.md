# Music Practice Assistant

A Python-based tool for real-time raga music practice with instant feedback on pitch accuracy and swara detection. This application combines audio processing, Indian classical music theory, and a live web-based dashboard for interactive practice sessions.

## Features

- **Real-time Audio Processing**: Live pitch detection and swara quantization
- **Raga Grammar Validation**: Validates sung notes against raga-specific grammar rules
- **Interactive Feedback**: Instant visual and textual feedback during practice
- **Web Dashboard**: Modern UI for session management and real-time monitoring
- **Multiple Ragas**: Support for various Indian classical ragas with their specific characteristics

## Project Structure

```
music-practice-assist/
├── raga_grammar/           # Core processing modules
│   ├── pitch_pipeline.py   # Audio to pitch conversion
│   ├── swara_quantizer.py  # Raga-based note quantization
│   ├── grammar_validator.py # Raga rule validation
│   ├── feedback_generator.py# Feedback generation
│   ├── raga_database.py    # Raga definitions and properties
│   └── live_audio_processor.py # Real-time audio handling
├── web/                    # FastAPI web application
│   ├── app.py             # Main API endpoints
│   ├── session_manager.py # Session management
│   ├── schemas.py         # Data models
│   └── static/            # Frontend assets
├── tests/                 # Test suite
├── ui/                    # UI-related components
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd music-practice-assist
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the UI

### Web Dashboard

Start the live practice dashboard:

```bash
python run_live_dashboard.py
```

This will start the FastAPI server on `http://localhost:8000`

**Access the dashboard:**
- Open your browser and navigate to `http://localhost:8000`
- Select a raga from the dropdown
- Grant microphone permissions when prompted
- Start singing and receive real-time feedback

### Available Features in Dashboard

- **Raga Selection**: Choose from available ragas
- **Real-time Visualization**: See pitch detection and swara quantization live
- **Feedback Display**: Get instant validation feedback
- **Session Control**: Start/stop practice sessions

## Command Line Tools

### Test Raga Grammar
```bash
python test_raga_grammar.py
```

### Tonic Detection
```bash
python tonic_sa_detection.py
```

## Dependencies

Key dependencies include:

- **fastapi**: Web framework for the dashboard API
- **uvicorn**: ASGI server for running the web application
- **librosa**: Audio processing and pitch detection
- **numpy/scipy**: Scientific computing
- **pydantic**: Data validation
- **websockets**: Real-time communication

See `requirements.txt` for the complete list.

## Configuration

Configuration can be adjusted in the respective module files:
- Audio processing parameters in `raga_grammar/pitch_pipeline.py`
- Raga definitions in `raga_grammar/raga_database.py`
- Web server settings in `run_live_dashboard.py`

## Architecture

The system works in three main stages:

1. **Audio Capture & Pitch Detection**: Real-time audio from microphone → pitch sequence
2. **Swara Quantization**: Raw pitch → raga-aligned swaras
3. **Feedback Generation**: Validation against raga rules → user feedback

## Testing

Run the test suite:
```bash
pytest tests/
```

## Documentation

See additional documentation:
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `ALGORITHM_EXPLANATION.md` - Algorithm descriptions
- `TONIC_RATIO_PITCH_DETECTION.md` - Pitch detection methodology

## License

[Add license information if applicable]

## Contributing

[Add contribution guidelines if applicable]

## Support

For issues or questions, please refer to the project documentation or create an issue in the repository.
