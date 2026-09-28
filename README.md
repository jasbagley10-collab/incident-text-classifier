# incident-text-classifier
A supervised machine-learning pipeline for classifying operational incident reports using TF-IDF and Logistic Regression.

Incident Text Classifier

A machine-learning text classification project that categorizes operational incident reports into four categories:
Navigation, Mechanical, Weather, Human Factors

The project demonstrates an end-to-end natural language processing (NLP) workflow using Python and scikit-learn, including data preparation, feature extraction, model training, evaluation, and prediction.

Overview
Operational incident reports are often written as short, unstructured descriptions.
This project explores how machine learning can automatically categorize those reports.

Ex:
“The aircraft lost GPS position during the mission.”
The model analyzes the text and predicts:
navigation
The overall workflow is:
Incident Report > TF-IDF > Numerical Features > Logistic Regression > Predicted Category

Dataset
The project uses a small synthetic dataset containing 40 operational incident reports.
Each report belongs to one of four categories.

Category        
Examples of Issues
 navigation      
 GPS, positioning, routes, waypoints
 
 mechanical      
 Motors, propellers, batteries, hardware
 
 weather         
 Wind, rain, visibility, temperature
 
 human_factors   
 Operator error, procedures, communication, decision-making
 
The dataset is intentionally small because the primary goal is demonstrating the machine-learning workflow.
It should not be considered representative of real-world aviation or UAS incident data.

Train/Test Split
The dataset is divided into training and testing data.
train_test_split(
 texts,
 labels,
 test_size=0.25,
 random_state=42,
 stratify=labels
 )
The model trains on 75% of the data and is evaluated on the remaining 25%.

random_state=42 makes the split reproducible.
stratify=labels helps maintain the same category distribution between the training and testing sets.
TF-IDF Feature Extraction
Machine-learning algorithms cannot directly understand text.
TF-IDF converts the text into numerical features.
TF-IDF stands for:
Term Frequency–Inverse Document Frequency
In simple terms, it gives more importance to words that are useful for distinguishing documents from one another.
The project also uses unigrams and bigrams:
TfidfVectorizer(
 ngram_range=(1, 2)
 )
This allows the model to consider both individual words and two-word combinations.
For example:
GPS
 signal
 lost
 GPS signal
 signal lost
This gives the classifier more context than looking at individual words alone.
Logistic Regression
The project uses Logistic Regression as the classification algorithm.
Despite its name, Logistic Regression is commonly used for classification problems.
The model learns relationships between the TF-IDF features and the incident categories.

For example, a report might produce:
human_factors: 0.04
 mechanical:    0.08
 navigation:    0.12
 weather:       0.76
The model would therefore classify the incident as:
weather
Model Evaluation
The initial model achieved:
Accuracy: 0.800

A confusion matrix is also generated in:
results/confusion_matrix.png
Interpreting the Results
The model correctly classified most of the test examples.
However, the results also demonstrate why accuracy alone is not enough.
The navigation category had a recall of 0.33 in this particular test split.
That means the model failed to identify some incidents that were actually labeled as navigation incidents.
This provides a useful starting point for error analysis and future model improvements.
Important Limitation

The dataset contains only 40 synthetic examples, with 10 examples used for testing.
Because the dataset is very small, the 80% accuracy should not be interpreted as production-level model performance.
The evaluation demonstrates the mechanics of building and measuring a classifier rather than establishing real-world reliability.

Running the Project
Clone the Repository
git clone https://github.com/YOUR-USERNAME/incident-text-classifier.git
 cd incident-text-classifier
Create a Virtual Environment
python -m venv .venv

Windows:
.venv\Scripts\activate

macOS/Linux:
source .venv/bin/activate

Install Dependencies
pip install -r requirements.txt

Train the Model
Run:
python src/train.py
The training script:
Loads the dataset.
Separates text and labels.
Creates training and testing sets.
Converts text into TF-IDF features.
Trains the Logistic Regression classifier.
Evaluates the predictions.
Saves the trained model.
The trained model is saved as:
incident_classifier.joblib
The model file is excluded from Git through .gitignore.

Make a Prediction
After training, run:
python src/predict.py
The program will prompt for an incident report:
Enter an incident report:
For example:
The aircraft experienced strong crosswinds during landing.
The model returns the predicted category and probability for each class.
You can also provide the incident directly:
python src/predict.py “The aircraft lost GPS position during flight.”

Generate Evaluation Results
Run:
python src/evaluate.py
This generates:
results/
 ├── classification_report.txt
 └── confusion_matrix.png
Project Structure
incident-text-classifier/
 ├── data/
 │   └── incidents.csv
 ├── results/
 │   ├── classification_report.txt
 │   └── confusion_matrix.png
 ├── src/
 │   ├── train.py
 │   ├── predict.py
 │   └── evaluate.py
 ├── .gitignore
 ├── LICENSE
 ├── README.md
 └── requirements.txt

File Descriptions
data/incidents.csv
Contains the labeled incident-report dataset used to train and evaluate the model.

src/train.py
Builds the machine-learning pipeline, trains the model, evaluates it, and saves the trained classifier.

src/predict.py
Loads the trained model and uses it to classify new incident reports.

src/evaluate.py
Generates evaluation metrics and a confusion matrix.

requirements.txt
Lists the Python packages required to run the project.

.gitignore
Prevents generated files, virtual environments, and the trained model from being committed unnecessarily.

LICENSE
Defines how other people may use, modify, and distribute the project.

Limitations
This project is intentionally a small baseline.

Small Dataset
The dataset contains only 40 examples.
A real-world application would require substantially more training data.

Synthetic Data
The examples were created specifically for this project.
They are not collected from actual operational incident databases.
Limited Categories
Only four categories are currently supported.
A larger system could include categories such as:
Communications, Battery, Payload, Software, Airspace, Security, Unknown

Simple Baseline Model
TF-IDF combined with Logistic Regression is intentionally straightforward.
More advanced NLP approaches could potentially capture more complex language patterns.

Future Improvements
Potential next steps include:
Expand the dataset.
Add more realistic and ambiguous examples.
Add an unknown category.
Compare multiple classification algorithms.
Perform cross-validation.
Add automated tests.
Expand model evaluation.
Perform structured error analysis.
Experiment with additional NLP techniques.
Build a simple web interface.
Containerize the application.
Experiment with transformer-based language models.

Key Takeaway
This project demonstrates the complete lifecycle of a small machine-learning application:
Data > Data Preparation > Feature Engineering > Model Training > Evaluation > Prediction

Rather than treating the model as a black box, each stage is implemented separately so that the system can be inspected, tested, and improved.
The current implementation serves as a baseline that can be expanded into a larger NLP application as additional data and functionality are introduced.

License
This project is licensed under the MIT License.
