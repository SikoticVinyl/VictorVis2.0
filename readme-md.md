# Halo Stats Analyzer

This project analyzes Halo 5 player statistics using the official Halo API and machine learning to predict match outcomes and analyze player performance.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/halo-stats-analyzer.git
cd halo-stats-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your Halo API key:
   - Get your API key from [Halo Developer Portal](https://developer.haloapi.com/)
   - Set it as an environment variable:
     ```bash
     export HALO_API_KEY='your-api-key-here'
     ```
   - Or pass it directly when initializing the analyzer

## Usage

```python
from app import HaloStatsAnalyzer

# Initialize analyzer
analyzer = HaloStatsAnalyzer(api_key='your-api-key-here')  # or leave empty to use env variable

# Analyze a player
history, X_test, y_test = analyzer.analyze_player('PlayerGamertag')

# Plot training results
analyzer.plot_training_history(history)
```

## Features

- Fetches player match history from Halo 5 API
- Processes match data into meaningful features
- Trains a deep learning model to predict match outcomes
- Visualizes training metrics and player performance
- Supports custom analysis parameters

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Requests
- Scikit-learn

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

- Halo API documentation and team
- TensorFlow and scikit-learn communities
