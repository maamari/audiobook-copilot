> **Note**: This entire project, including all code, documentation, the following README, and this disclaimer was generated with the assistance of AI (Claude 3.5 Sonnet). Karime Maamari does not take responsibility for any errors, inaccuracies, or potential issues that may arise from using this AI-generated code. Use at your own discretion.

# Audiobook Copilot

An interactive audiobook player with real-time AI-powered question answering capabilities. Integrates voice input, audio transcription, and contextual understanding to provide immediate answers about audiobook content.

## Core Features

- Real-time question answering about audiobook content
- Voice-based interaction with AI assistant
- Variable speed playback (1x, 1.25x, 1.5x)
- Automatic content transcription
- Position tracking and bookmarking
- 30-second navigation controls

## Requirements

- Python 3.8+
- FFmpeg
- OpenAI API key
- Microphone
- MP3 audiobook file

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
export OPENAI_API_KEY='your-key'
```

2. Place audiobook in project root:
```bash
cp your-audiobook.mp3 output_small.mp3
```

3. Start server:
```bash
python audioplayer.py
```

4. Access interface:
- Local: http://localhost:5000
- Remote: Configure ngrok for external access

## Technical Architecture

- **Frontend**: HTML5/JS with Socket.IO
- **Backend**: Flask/Python
- **AI Models**: OpenAI GPT-4 & Whisper
- **Audio Processing**: FFmpeg & Pygame

## File Structure
```
.
├── audioplayer.py          # Server & core logic
├── templates/index.html    # Web interface
├── transcript.json         # Generated transcript
└── bookmarks.json         # User bookmarks
```

## Known Limitations

- Questions limited to recent context (10 minutes)
- Initial transcript generation time scales with file size
- Browser microphone permissions required
- Active internet connection needed

## Troubleshooting

- Verify audiobook file exists as `output_small.mp3`
- Check OpenAI API key configuration
- Ensure microphone permissions are granted
- Monitor browser console for errors

## Performance Recommendations

- Use high-quality MP3 files under 100MB
- Maintain stable internet connection
- Use Chrome/Edge for best compatibility
- Allow initial transcription to complete

## License

MIT License

## Acknowledgments

- OpenAI APIs
- FFmpeg
- Flask & Socket.IO
- PyGame
- Font Awesome 