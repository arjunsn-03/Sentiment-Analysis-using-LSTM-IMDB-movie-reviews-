
# IMDB Sentiment Analysis using LSTM 🎬🧠

This project demonstrates a deep learning approach to sentiment analysis on the IMDB dataset of 50,000 movie reviews. A Long Short-Term Memory (LSTM) neural network is built using TensorFlow and Keras to classify reviews as either **positive** or **negative**.

## ✨ Features

  - **End-to-End Workflow**: Covers every step from data collection via the Kaggle API to building a predictive system.
  - **Deep Learning Model**: Implements a powerful LSTM network, ideal for understanding the context and sequence of textual data.
  - **High Performance**: Achieves a test accuracy of approximately **88.21%**.
  - **Data Preprocessing**: Includes robust text tokenization, padding, and data splitting techniques.
  - **Exploratory Data Analysis (EDA)**: Visualizes sentiment distribution and review length to understand the dataset's characteristics.
  - **Predictive System**: A ready-to-use function is included to classify the sentiment of any new movie review.

-----

## 📋 Table of Contents

  - [Workflow](https://www.google.com/search?q=%23-workflow)
  - [Dataset](https://www.google.com/search?q=%23-dataset)
  - [Model Architecture](https://www.google.com/search?q=%23-model-architecture)
  - [Performance](https://www.google.com/search?q=%23-performance)
  - [Technologies & Libraries Used](https://www.google.com/search?q=%23-technologies--libraries-used)
  - [Setup & Usage](https://www.google.com/search?q=%23-setup--usage)
  - [Predictive System in Action](https://www.google.com/search?q=%23-predictive-system-in-action)

-----

## 🚀 Workflow

The project follows a systematic machine learning pipeline:

1.  **Data Collection**: The "IMDB Dataset of 50k Movie Reviews" is downloaded directly from Kaggle using the Kaggle API.
2.  **Exploratory Data Analysis (EDA)**: The dataset is analyzed to check for class balance. It is found to be perfectly balanced with 25,000 positive and 25,000 negative reviews.
3.  **Data Preprocessing**:
      * Sentiment labels ('positive'/'negative') are converted to numerical format (1/0).
      * The text data is tokenized using `tf.keras.preprocessing.text.Tokenizer`, keeping the top 5,500 most frequent words.
      * All review sequences are padded to a uniform length of 200 tokens using `pad_sequences`.
      * The dataset is split into training (80%) and testing (20%) sets.
4.  **Modeling**:
      * A `Sequential` model is constructed in Keras.
      * The model consists of an **Embedding** layer, an **LSTM** layer with dropout regularization, and a final **Dense** output layer with a sigmoid activation function.
5.  **Training**: The model is compiled with the `adam` optimizer and `binary_crossentropy` loss function. It is trained for 5 epochs with a batch size of 64.
6.  **Evaluation**: The trained model's performance is evaluated on the unseen test data.
7.  **Prediction**: A function is created to take a raw text review, preprocess it, and predict its sentiment.

-----

## 📊 Dataset

The project utilizes the **IMDB Dataset of 50K Movie Reviews**, a classic binary sentiment classification dataset.

  - **Content**: 50,000 movie reviews.
  - **Labels**: 25,000 labeled as 'positive' and 25,000 as 'negative'.
  - **Source**: [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

-----

## 🏗️ Model Architecture

The neural network is a `Sequential` model with the following layers:

| Layer (type) | Output Shape | Param \# | Purpose |
| :--- | :--- | :--- |:--- |
| **Embedding** | (None, 200, 128) | 704,000 | Converts word indices into dense 128-dimensional vectors. |
| **LSTM** | (None, 128) | 131,584 | Processes the sequence of word vectors to capture context and long-term dependencies. Includes 20% dropout and recurrent dropout for regularization. |
| **Dense** | (None, 1) | 129 | The output layer with a sigmoid activation function to produce a probability score between 0 and 1 for binary classification. |

**Total params: 835,713**
**Trainable params: 835,713**
**Non-trainable params: 0**

-----

## 📈 Performance

The model was evaluated on a test set of 10,000 reviews, yielding the following results:

  - **Test Loss**: $0.3015$
  - **Test Accuracy**: $88.21%$

This indicates that the model generalizes well to new, unseen data.

-----

## 💻 Technologies & Libraries Used

  - **Python 3**
  - **TensorFlow & Keras**: For building and training the LSTM model.
  - **Pandas**: For data manipulation and loading.
  - **Scikit-learn**: For splitting the data into training and test sets.
  - **Seaborn & Matplotlib**: For data visualization and EDA.
  - **Kaggle API**: For programmatic data collection.
  - **Jupyter Notebook**: For interactive development.

-----

## ⚙️ Setup & Usage

To run this project locally, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: Create a `requirements.txt` file listing all libraries (`tensorflow`, `pandas`, `scikit-learn`, `kaggle`, etc.).*

3.  **Set up Kaggle API**:

      * Download your `kaggle.json` API token from your Kaggle account settings.
      * Place the `kaggle.json` file in the root directory of the project. The notebook is configured to read credentials from this file.

4.  **Run the Jupyter Notebook**:

    ```bash
    jupyter notebook IMDB_Sentiment_Analysis_using_LSTM_DL.ipynb
    ```

-----

## 🔮 Predictive System in Action

You can easily test the model with your own reviews using the `predict_sentiment` function defined in the notebook.

**Code:**

```python
def predict_sentiment(review):
  # Tokenize and pad the review
  sequence = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequence, maxlen=200)

  # Make a prediction
  prediction = model.predict(padded_sequence)
  sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
  return sentiment

# Example 1: Positive Review
new_review_1 = 'This movie was marvellous and beautiful'
sentiment_1 = predict_sentiment(new_review_1)
print(f"The sentiment of the review is: {sentiment_1}")
# Expected Output: The sentiment of the review is: positive

# Example 2: Negative Review
new_review_2 = 'This movie was not that good'
sentiment_2 = predict_sentiment(new_review_2)
print(f"The sentiment of the review is: {sentiment_2}")
# Expected Output: The sentiment of the review is: negative
```
