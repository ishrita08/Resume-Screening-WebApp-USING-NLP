
# **Resume Screening WebApp using NLP**

## **Overview**

The **Resume Scanner Web App** is a machine learning-based solution designed to classify resumes into specific job categories. By analyzing the text within resumes, the app predicts the most likely profession of the applicant, from **Data Science** to **Civil Engineering**. It leverages **Natural Language Processing (NLP)** techniques combined with supervised learning models to automate and streamline resume screening.

## **Features**

- 📝 **Automated Resume Classification**:  
  Automatically categorizes resumes into predefined job roles based on their content.
  
- ⚡ **Real-time Analysis**:  
  Upload and analyze resumes in real-time using the pre-trained model.
  
- 🧹 **Data Cleaning**:  
  Cleans and preprocesses raw resume text to enhance classification accuracy by removing noise like URLs, special characters, etc.
  
- 🤖 **Machine Learning Model**:  
  Utilizes **K-Nearest Neighbors (KNN)** with **One-vs-Rest** classification for job prediction.

- ♻️ **Easily Transferable**:  
  The model can be adapted to new data and job categories with minimal retraining, making it versatile and scalable.

## **Dataset**

The app is trained on the `UpdatedResumeDataSet.csv` dataset, which contains resume texts labeled with corresponding job categories. This dataset provides a variety of job categories, allowing the app to distinguish between different professions effectively.

- **Columns**:  
  - **Resume**: The textual content of the resume.  
  - **Category**: The job category that the resume belongs to (e.g., Data Science, Web Development, HR).

---

## **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/resume-scanner.git
   cd resume-scanner
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:  
   Make sure `UpdatedResumeDataSet.csv` is placed in the root directory of the project.

---

## **Usage**

### **Step 1: Data Preprocessing**

1. **Load and Explore the Dataset**:
   ```python
   df = pd.read_csv('UpdatedResumeDataSet.csv')
   print(df.shape)
   df['Category'].value_counts()
   ```

2. **Visualize Data Distribution**:
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   plt.figure(figsize=(15,5))
   sns.countplot(df['Category'])
   plt.show()
   ```

3. **Clean the Resume Text**:
   ```python
   import re

   def cleanResume(txt):
       # Cleaning steps for removing special characters, URLs, etc.
       cleanText = re.sub('http\S+\s', ' ', txt)
       cleanText = re.sub('RT|cc', ' ', cleanText)
       cleanText = re.sub('#\S+\s', ' ', cleanText)
       cleanText = re.sub('@\S+', ' ', cleanText)
       cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
       cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
       cleanText = re.sub('\s+', ' ', cleanText)
       return cleanText

   df['Resume'] = df['Resume'].apply(cleanResume)
   ```

### **Step 2: Feature Extraction & Model Training**

1. **Encode Categories**:
   ```python
   from sklearn.preprocessing import LabelEncoder
   le = LabelEncoder()
   df['Category'] = le.fit_transform(df['Category'])
   ```

2. **TF-IDF Vectorization**:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer

   tfidf = TfidfVectorizer(stop_words='english')
   requiredText = tfidf.fit_transform(df['Resume'])
   ```

3. **Split the Data**:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(requiredText, df['Category'], test_size=0.2, random_state=42)
   ```

4. **Train the K-Nearest Neighbors Model**:
   ```python
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.multiclass import OneVsRestClassifier

   clf = OneVsRestClassifier(KNeighborsClassifier())
   clf.fit(X_train, y_train)
   ```

5. **Evaluate the Model**:
   ```python
   from sklearn.metrics import accuracy_score

   y_pred = clf.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Model Accuracy: {accuracy}")
   ```

### **Step 3: Save the Model**

1. **Save the Trained Model**:
   ```python
   import pickle

   pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
   pickle.dump(clf, open('clf.pkl', 'wb'))
   ```

---

## **Running the Web App**

1. **Load the Pretrained Models**:
   ```python
   import pickle

   clf = pickle.load(open('clf.pkl', 'rb'))
   tfidf = pickle.load(open('tfidf.pkl', 'rb'))
   ```

2. **Predict a Job Category for a Given Resume**:
   Provide a resume text, and the app will predict the job category:
   ```python
   myresume = "I am a data scientist with expertise in machine learning and deep learning..."
   cleaned_resume = cleanResume(myresume)
   input_features = tfidf.transform([cleaned_resume])
   prediction_id = clf.predict(input_features)[0]
   ```

3. **Category Mapping**:
   Map the predicted category ID to a human-readable job category:
   ```python
   category_mapping = {
       0: "Advocate", 1: "Arts", 2: "Automation Testing", 3: "Blockchain", 
       4: "Business Analyst", 5: "Civil Engineer", 6: "Data Science", ...
   }

   category_name = category_mapping.get(prediction_id, "Unknown")
   print("Predicted Category:", category_name)
   ```

---

## **Example**

**Input Resume**:

```plaintext
I am a data scientist specializing in machine learning, deep learning, and computer vision...
```

**Predicted Output**:

```
Predicted Category: Data Science
```

---

## **Contributing**

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the functionality of this project.

