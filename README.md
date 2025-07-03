# Task-1-2-3-
Audio-based Anomaly Classifier , Image-Based Object State Recognition , Sensor Fusion with Anomaly Detection




# Task1
## Topic-Audio-based Anomaly Classifier

### Dataset - https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

## ***objective***
#### The main goal of this project is to detect anomalies in a given audio 

## ***Steps for preprocessing of the audios***
 

| Step             | What It Does                           | Tool Used                                 |
| ---------------- | -------------------------------------- | ----------------------------------------- |
| Load audio files | Picks audio from RAVDESS               | `glob`, `librosa.load`, `IPython.display` |
| Trim silence     | Removes quiet parts                    | `librosa.effects.trim`                    |
| Plot waveform    | Shows amplitude over time              | `pandas`, `matplotlib`, `seaborn`         |
| Spectrogram      | Shows how frequencies change over time | `librosa.stft`, `librosa.display`         |
| Mel-spectrogram  | Human-like frequency perception        | `librosa.feature.melspectrogram`          |

## ***Tools/Libraries used***
| Tool/Library                 | Purpose                              |
| ---------------------------- | ------------------------------------ |
| **`pandas` / `numpy`**       | Data handling and arrays             |
| **`matplotlib` / `seaborn`** | Plotting graphs                      |
| **`librosa`**                | Audio analysis and processing        |
| **`IPython.display`**        | For playing audio inside notebook    |
| **`glob`**                   | For finding file paths from folders  |
| **`itertools.cycle`**        | For automatic color cycling in plots |

## ***Detecting Anamolies in the present audios***
| Stage              | Description                             |
| ------------------ | --------------------------------------- |
| Data Collection    | Used `glob` to fetch `.wav` files       |
| Labeling           | Used RAVDESS emotion codes to tag files |
| Feature Extraction | Used `librosa` MFCCs                    |
| Model Training     | Random Forest classifier                |
| Evaluation         | Accuracy + Confusion Matrix             |
| Inference          | Predict + Visualize + Play audio        |

## ***Tools/Libraries used***
| Library                 | Purpose                                             |
| ----------------------- | --------------------------------------------------- |
| `librosa`               | Load and process audio files, extract MFCC features |
| `pandas`, `numpy`       | Data manipulation                                   |
| `matplotlib`, `seaborn` | Visualizing waveforms and spectrograms              |
| `scikit-learn`          | Machine learning model (Random Forest)              |
| `glob`, `os`            | File handling (fetch all `.wav` files from folders) |
| `IPython.display`       | Play audio in notebook (Kaggle/Colab)               |

# Task2
# State Recognition of a given object

## Objective : To predict whether a given object is in on or off
### Dataset : https://www.kaggle.com/datasets/vickykumaryadav/simulated-iot-environmental-sensor-dataset

##   Process involved to successful completion of the project

### 1. ***Image Preprocessing:***
  #### Applies resizing, normalization, and augmentations (like flipping, rotation) to make the model more robust.
### 2. ***Train/Validation Split:***
  #### Randomly splits data into 80% for training and 20% for validation.
### 2. ***Pretrained Model:***
 #### Loads MobileNetV3 (lightweight CNN model pre-trained on ImageNet), and adjusts the final layer for binary classification.
### 2. ***Training Loop:***  
 #### rains the model over multiple epochs, evaluates on validation set each time, and saves the best-performing model.
### 2. ***Prediction:***
 #### predicts the new given image as on or off.

## Librabries Used:

#### 
| Library                                           | Purpose                                                          |
| ------------------------------------------------- | ---------------------------------------------------------------- |
| `torch`, `torch.nn`, `torch.optim`                | Core PyTorch libraries for building and training neural networks |
| `torchvision.datasets`                            | To load image datasets with automatic label generation           |
| `torchvision.transforms`                          | For image preprocessing and data augmentation                    |
| `torchvision.models`                              | To use pre-trained models (like MobileNetV3)                     |
| `torch.utils.data`                                | To split dataset and create dataloaders                          

# Task 3
# SensorFusionwithAnomalyDetection(Simulated)

## Objective : classify Normalbehavior Environmental abnormality

##  Process involved to successful completion of the project
###  1. Dataset Loading
###  2. Feature Selection + Scaling
###  3. Isolation Forest Anomaly Detection
###  4. K-Means Clustering
###  5. Visualization
###  6.Prediction Function

## Librabries Used:
####

| Library                                | Purpose                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------- |
| `pandas`                               | Loading, managing, and manipulating structured data (`DataFrame`)          |
| `numpy`                                | Numeric operations (not directly used here, but standard for ML workflows) |
| `matplotlib.pyplot`                    | For plotting line/scatter plots                                            |
| `seaborn`                              | Advanced visualization (used to color anomalies on the scatterplot)        |
| `sklearn.ensemble.IsolationForest`     | To detect anomalies without labels                                         |
| `sklearn.cluster.KMeans`               | To cluster sensor data and identify abnormal patterns                      |
| `sklearn.preprocessing.StandardScaler` | To normalize sensor values (so all features are on the same scale)         |
| `os`                                   | To access and list file paths from your Kaggle environment                 |
