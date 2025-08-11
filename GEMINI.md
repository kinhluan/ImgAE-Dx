# Project Overview

This project is a research endeavor focused on unsupervised anomaly detection in medical imaging, specifically using autoencoder architectures. The core of the project is a comparative study between a standard U-Net architecture and a novel Reversed Autoencoder (RA) architecture. The goal is to evaluate their effectiveness in identifying anomalies in medical X-ray images.

**Current Development Strategy:** The project is transitioning from an initial research notebook to a robust, modular Python codebase designed for cloud-native streaming ML. The final research notebook for Google Colab will be generated from this production-ready code to ensure reproducibility and maintainability.

## Key Files

*   `DEVELOPMENT_TODO.md`: The comprehensive roadmap outlining the transition to a cloud-native streaming ML system, detailing all phases and tasks.
*   `PROGRESS_SUMMARY.md`: Tracks the current progress against the `DEVELOPMENT_TODO.md`, highlighting completed tasks and next priorities.
*   `STREAMING_ARCHITECTURE_UPDATED.md`: Provides updated architectural details, especially concerning the streaming and cloud-native aspects.
*   `configs/project_config.yaml`: The main configuration file for project settings, model parameters, and training configurations.
*   `src/`: The directory containing the modular Python codebase, organized into sub-modules for streaming, data, models, training, and utilities.
*   `notebooks/Anomaly_Detection_Research_Colab.ipynb`: The *final generated* Jupyter Notebook encapsulating the entire research process for execution in Google Colab.
*   `docs/IMG_AE_DX_ARCHITECTURE.md`: Detailed description of the model architectures and the anomaly detection algorithm.
*   `PROJECT_JOURNEY.md`, `REQUIREMENT.md`, `Structure.md`, `IDEA.md`, `QUESTION.md`, `TODO.md`: Supporting documentation outlining research methodology, requirements, and brainstorming.

## Building and Running

This project is designed for development as a modular Python application, with a final output optimized for the Google Colab environment.

**Development Workflow:**

1.  **Setup Environment:** Create a virtual environment and install dependencies (as specified in `pyproject.toml` or `requirements.txt` once created).
2.  **Implement Modules:** Develop the core functionalities within the `src/` directory (e.g., data streaming, model architectures, training loops).
3.  **Configure:** Adjust settings in `configs/project_config.yaml`.
4.  **Run Experiments:** Execute Python scripts from the `src/` directory for training and evaluation.

**To run the final research notebook in Google Colab:**

1.  Open the `Anomaly_Detection_Research_Colab.ipynb` file in Google Colab (this file will be generated from the production code).
2.  Follow the instructions within the notebook to mount your Google Drive, set up the environment, and download/stream the necessary data.
3.  Execute the cells sequentially to run the entire experiment.

**Dependencies:**

The project's dependencies will be explicitly listed in a `pyproject.toml` or `requirements.txt` file. Based on the project's scope, the following Python libraries are likely required:

*   TensorFlow or PyTorch (PyTorch is currently favored based on `config_manager.py`)
*   NumPy, Pandas, Matplotlib, Scikit-learn
*   Kaggle API (for downloading/streaming the dataset)
*   Weights & Biases (for experiment tracking)
*   PyYAML (for configuration management)

## Development Conventions

*   **Unsupervised Learning:** The models are trained exclusively on "normal" data (images with no findings) to learn a representation of normal anatomy.
*   **Comparative Analysis:** The project follows a rigorous comparative experimental design, training and evaluating both the U-Net and RA models on the same data and with the same procedures.
*   **Evaluation:** The primary evaluation metric is the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), supplemented by a qualitative analysis of reconstruction error maps.
*   **Checkpointing:** The training process includes a checkpointing mechanism to save the model's state after each epoch, allowing for the resumption of long training sessions.
*   **Reproducibility:** The entire workflow is built as a modular Python codebase, from which a self-contained Jupyter Notebook will be generated to ensure full reproducibility of the research findings.
*   **Software Engineering Practices:** Emphasis on clean code, modularity, and proper configuration management (as exemplified by `config_manager.py`).