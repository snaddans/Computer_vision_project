# Computer_vision_project
# 🐄 Bovine Vision: Cattle & Buffalo Breed Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

Welcome to **Bovine Vision**! This project is a Deep Learning-based Computer Vision application designed to automatically identify and classify six distinct breeds of cattle and buffaloes from digital images. 

This project was developed as a Bring Your Own Project (BYOP) for a university Computer Vision coursework evaluation.

---

## 📖 Table of Contents
1. [What Does This Project Do?](#-what-does-this-project-do)
2. [Supported Breeds](#-supported-breeds)
3. [Project Structure](#-project-structure)
4. [Step-by-Step Setup Guide](#-step-by-step-setup-guide)
5. [How to Run the Project](#-how-to-run-the-project)
6. [Understanding the Outputs](#-understanding-the-outputs)
7. [Results](#-results)

---

## 🤔 What Does This Project Do?
Proper identification of cattle and buffalo breeds is essential for dairy farm management, optimizing nutritional plans, and providing specialized veterinary care. 

This project utilizes a state-of-the-art **ConvNeXt-Tiny** convolutional neural network. It takes an image of a bovine animal as input and outputs the predicted breed. The pipeline automatically handles data augmentation, model training, and generates detailed visual analytics (like loss curves and confusion matrices) to track the model's performance.

## 🐃 Supported Breeds
The model is currently trained to recognize the following six classes:
* **Gir** (Cattle)
* **Holstein Friesian** (Cattle)
* **Jaffrabadi** (Buffalo)
* **Jersey** (Cattle)
* **Murrah** (Buffalo)
* **Sahiwal** (Cattle)

---

## 📂 Project Structure
Before running the code, it's helpful to understand how the project is organized:

```text
bovine-vision/
│
├── convn0011.py                # Main Python script containing the training pipeline
├── convon_tiny002.ipynb        # Jupyter Notebook version of the pipeline for interactive running
├── README.md                   # This file!
│
├── Cattle-Buffalo-breeds.v1i.folder/  # THE DATASET (You must place this here)
│   ├── train/                  # Training images divided by breed folders
│   └── test/                   # Testing/Validation images divided by breed folders
│
└── gg/                         # OUTPUT FOLDER (Automatically generated)
    ├── convnext_tiny_best.pth  # The saved trained model weights
    ├── train_distribution.png  # Dataset analytics
    └── confusion_matrix_epoch_X.png # Epoch-by-epoch evaluation graphs
```
⚙️ Step-by-Step Setup Guide
Follow these instructions exactly to get the project running on your local machine.

1. Prerequisites
You need Python 3.8 or higher installed on your computer.

To check if you have Python, open your terminal (Command Prompt/PowerShell on Windows, Terminal on Mac/Linux) and type: python --version

2. Clone the Repository
Download this code to your computer by cloning the GitHub repository:
Bash
git clone https://github.com/snaddans/Computer_vision_project.git
cd Computer_vision_project
3. Create a Virtual Environment (Recommended)
It is best practice to create an isolated environment for this project so dependencies don't clash with other projects on your computer.

Bash
# Create the environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
4. Install Required Libraries
With your virtual environment activated, install the necessary Python packages:

Bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
5. Setup the Dataset
The code expects a specific folder structure for the images. Ensure your dataset folder is named Cattle-Buffalo-breeds.v1i.folder and is placed in the root directory of the project. It must contain a train and test folder, each containing sub-folders for the 6 breeds.

(Note: If your dataset is located elsewhere, you will need to open convn0011.py or the Jupyter Notebook and update the dataset_path variable to point to your actual folder).

🚀 How to Run the Project
You have two options to run the project: using the standard Python script or using the interactive Jupyter Notebook.

Option A: Run via Terminal (Python Script)
If you just want to run the training pipeline from start to finish, run the python file:

Bash
python convn0011.py
Watch the terminal! It will print out the loss, accuracy, and a detailed classification report for every single epoch.

Option B: Run via Jupyter Notebook
If you want to run the code cell-by-cell and inspect the data interactively:

Install Jupyter: pip install jupyter

Launch the notebook: jupyter notebook

Open convon_tiny002.ipynb in your browser.

Click "Run All" (or run the cells one by one).

📊 Understanding the Outputs
As the project runs, it automatically creates a new folder called gg/ in your project directory. You don't need to do anything manually!

Inside the gg/ folder, the code will save:

Dataset Graphs: Visualizations showing how many images exist for each breed in the train and test sets.

Training Histories: Line graphs showing the model's accuracy and loss over time.

Confusion Matrices: Heatmaps generated at every epoch showing exactly which breeds the model is confusing with one another.

The Model: convnext_tiny_best.pth - This is your final, trained AI brain. You can load this file later to make predictions on brand new images without having to retrain!

🏆 Results
When trained on a standard CPU for 30 epochs, the ConvNeXt-Tiny model achieved the following robust results:

Final Training Accuracy: ~91.60%

Final Validation Accuracy: ~87.41%

Macro Avg F1-Score: ~0.84

Created by Shashank Dubey (VIT Bhopal University) for Computer Vision BYOP Evaluation.
