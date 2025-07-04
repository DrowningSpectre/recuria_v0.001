# Recuria: Exploring Stability Between Order and Chaos with TensorFlow


### Overview

Recuria is a modular Python project designed to investigate emergent behavior at the boundary between order and chaos. It blends recursive system modeling with modern TensorFlow-based neural network training to observe and analyze stability patterns in various input types — including prime numbers, random sequences, and periodic patterns.

## Features

🧠 Recursive systems that dynamically adjust weights based on input stability

🔢 Analysis of input types: prime sequences, random binaries, repeating patterns

⚙️ TensorFlow integration with modular architecture

📊 Visualization of stability evolution and internal weight changes

🔧 Configurable via JSON for easy experimentation

💾 Saves models and results for further analysis

## Project Structure

/
├── data/           # Dataset storage
├── models/         # Saved TensorFlow models
├── recuria/        # Core recursive system logic
│   ├── engine.py         # Recursive dynamics and weight adaptation
│   ├── primes.py         # Prime input generator and utilities
│   ├── visualize.py      # Plotting and output generation
│   └── runner.py         # Execution interface for Recuria
├── model/          # TensorFlow model definition
│   └── network.py        # Model architecture builder
├── training/       # Model training and evaluation helpers
│   └── trainer.py
├── main.py         # Main execution entry point
├── config.json     # Settings for model and training
└── README.md       # Project documentation

## Getting Started

1. Clone the Repository

git clone 

2. Set Up Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Run the Program

python main.py

Trains the model, evaluates it, saves results to /models, and performs Recuria analysis with plotted output.

Configuration Example (config.json)

{
  "model": {
    "input_shape": [28, 28],
    "hidden_units": 128,
    "dropout": 0.2,
    "output_classes": 10
  },
  "training": {
    "epochs": 5,
    "validation_split": 0.1,
    "batch_size": 32
  }
}

## Philosophy

Recuria explores what lies between pure determinism and randomness — the space where structure and unpredictability meet. By leveraging well-known inputs such as primes or pseudo-random sequences, it studies how systems stabilize or collapse, possibly hinting at broader patterns underlying emergent phenomena in nature and cognition.

## Contributions

Contributions are welcome — fork, suggest, discuss, or create pull requests to collaborate.

## License

This project is licensed under the Apache License 2.0.