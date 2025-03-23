# Sign Language Detection

This project aims to develop a gesture based sign language detection system using pre-trained ML and deep learning techniques. The system takes video input, processes them to extract relevant features, and translates these gestures into corresponding sentence.

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Ketulmj/Sign-Language-Detection.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd Sign-Language-Detection
   ```

# Dataset

https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset

3. **Install Dependencies**:

   It's recommended to use a virtual environment. If you don't have `virtualenv` installed, you can install it using:

   ```bash
   pip install virtualenv
   ```

   Create and activate a virtual environment:

   ```bash
   virtualenv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the sign language detection:

1. **Detection from test data**:

   ```bash
   python predict_realtime.py
   ```
2. . **Detection of user-input video with UI**:

   ```bash
   streamlit run predict_UI.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
