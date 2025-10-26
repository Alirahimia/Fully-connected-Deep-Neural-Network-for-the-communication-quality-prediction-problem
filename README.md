
Fully Connected Deep Neural Network for Communication Quality Prediction

This repository presents a reproducible implementation of a fully connected deep neural network (FNN) designed to predict communication quality in construction projects. The model leverages structured project data to forecast communication effectiveness, supporting proactive decision-making in complex engineering environments. This work is based on peer-reviewed research published in Automation in Construction, and integrates advanced machine learning techniques including XGBoost for benchmarking.

🧠 Overview

- Goal: Predict communication quality scores based on project features using deep learning.
- Model: Fully connected neural network with multiple hidden layers, trained on labeled datasets.
- Benchmark: Compared against XGBoost and other classical ML models.
- Application: Enhances project management by identifying communication bottlenecks early.

📁 Repository Structure

`
├── data/
│   └── communicationqualitydataset.csv
├── models/
│   └── fnn_model.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── utils/
│   └── preprocessing.py
├── main.py
├── requirements.txt
└── README.md
`

🚀 Getting Started

Prerequisites

Ensure you have Python 3.8+ installed. Recommended environment: Conda or virtualenv.

Installation

`bash
git clone https://github.com/Alirahimia/Fully-connected-Deep-Neural-Network-for-the-communication-quality-prediction-problem.git
cd Fully-connected-Deep-Neural-Network-for-the-communication-quality-prediction-problem
pip install -r requirements.txt
`

Running the Model

`bash
python main.py
`

This will load the dataset, preprocess the features, train the FNN model, and output performance metrics.

📊 Dataset

The dataset contains anonymized project-level features such as:

- Team size
- Project duration
- Communication channels
- Stakeholder diversity
- Historical communication scores

Target variable: Communication Quality Score (continuous value between 0 and 1)

🧪 Evaluation Metrics

- Mean Squared Error (MSE)
- R² Score
- Comparative performance with XGBoost

📌 Key Features

- Modular codebase for easy experimentation
- Clean preprocessing pipeline
- Hyperparameter tuning options
- Jupyter notebook for exploratory data analysis
- Ready for integration with larger BIM or project management systems

📚 Related Publications

This repository supports the findings of the following publication:

> Rahimia, A., et al. (2023). Communication Quality Prediction in Construction Projects Using FNN and XGBoost. Automation in Construction.

🤝 Collaborations

Developed in collaboration with researchers from the University of Melbourne. Contributions welcome via pull requests or issues.

🙋‍♂️ Contact

For questions, collaborations, or demo requests, please reach out via LinkedIn or open an issue in this repository.

--
