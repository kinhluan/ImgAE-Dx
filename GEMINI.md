# Project Overview

This project is a research endeavor focused on unsupervised anomaly detection in medical imaging, specifically using autoencoder architectures. The core of the project is a comparative study between a standard U-Net architecture and a novel Reversed Autoencoder (RA) architecture. The goal is to evaluate their effectiveness in identifying anomalies in medical X-ray images. The project is implemented as a single Jupyter Notebook designed to be run on Google Colab, utilizing the NIH Chest X-ray dataset.

## Key Files

* `Anomaly_Detection_Research_Colab.ipynb`: The main Jupyter Notebook containing the entire research process, from data loading and preprocessing to model training, evaluation, and analysis.
* `PROJECT_JOURNEY.md`: A detailed document outlining the research methodology, including the problem statement, research questions, hypotheses, and implementation strategy.
* `REQUIREMENT.md`: Specifies the project requirements, including the use of an autoencoder architecture, the choice of problem, and the expected structure of the final notebook.
* `Structure.md`: Outlines the structure of the project using the Question-Evidence-Conclusion (QEC) framework.
* `IDEA.md`, `QUESTION.md`, `TODO.md`: These files likely contain brainstorming ideas, specific questions to be addressed, and a to-do list for the project.

## Building and Running

This project is designed to be run in a Google Colab environment.

**To run the project:**

1. Open the `Anomaly_Detection_Research_Colab.ipynb` file in Google Colab.
2. Follow the instructions within the notebook to mount your Google Drive, set up the environment, and download the necessary data.
3. Execute the cells in the notebook sequentially to run the entire experiment.

**Dependencies:**

The project's dependencies are not explicitly listed in a `requirements.txt` file, but based on the context, the following Python libraries are likely required:

* TensorFlow or PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Kaggle API (for downloading the dataset)

## Development Conventions

* **Unsupervised Learning:** The models are trained exclusively on "normal" data (images with no findings) to learn a representation of normal anatomy.
* **Comparative Analysis:** The project follows a rigorous comparative experimental design, training and evaluating both the U-Net and RA models on the same data and with the same procedures.
* **Evaluation:** The primary evaluation metric is the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), supplemented by a qualitative analysis of reconstruction error maps.
* **Checkpointing:** The training process includes a checkpointing mechanism to save the model's state after each epoch, allowing for the resumption of long training sessions.
* **Reproducibility:** The entire workflow is encapsulated in a single Jupyter Notebook to ensure reproducibility.
