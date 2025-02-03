# NLP Assignment 3 (AIT - DSAI)

- [Student Information](#student-information)
- [Files Structure](#files-structure)
- [Task 1 - Get Language Pair](#task-1---get-language-pair)
   - [Dataset](#dataset)
   - [Preprocessing](#preprocessing)
   - [Tokenizing](#tokenizing)
- [Task 2 - Experiment with Attention Mechanisms](#task-2---experiment-with-attention-mechanisms)
- [Task 3 - Evaluation and Model Selection](#task-3---evaluation-and-model-selection)
- [Task 4 - Web Application](#task-4---web-application)

---

## Student Information
- **Name**: Soe Htet Naing  
- **ID**: st125166  

---

## Files Structure
- **app**: Contains application files such as `app.py`, models, and templates.  
- **code**: Includes Jupyter notebooks for different attention mechanisms.  
- **images**: Stores images for performance plots and visualizations.  
- **trained_models**: Holds pre-trained models and metadata files.  

---

## Task 1 - Get Language Pair

### Dataset
- The dataset was downloaded from [ALT Project](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/).  
- English (`ALT.en-my.en`) and Myanmar (`ALT.en-my.my`) text files were downloaded separately, with each containing 18,088 rows.  
- The English and Myanmar sentences were extracted and combined into a single dataset.

### Data Preprocessing
- After extracting sentences from both files, the dataset was split into:
  - Training set: 14,651 rows
  - Validation set: 1,628 rows
  - Test set: 1,809 rows  
- The final dataset was uploaded to Hugging Face as a public repository.

### Tokenizing
- The [pyidaungsu](https://pypi.org/project/pyidaungsu/) library was used for tokenizing Myanmar text.
- Tokenization is performed at the word level, ensuring each token accurately represents Myanmar words.  
- Example:
  - Input: "ကျွန်တော် က တစ် ဦး တည်း သော ဒီ မှာ မွေး ကတည်းက နေ လာ တဲ့ ကိုယ်စားလှယ်လောင်း ဖြစ် ပါ တယ် ။"  
  - Output: `['ကျွန်တော်', 'က', 'တစ်', 'ဦးတည်း', 'သော', 'ဒီ', 'မှာ', 'မွေး', 'က', 'တည်း', 'က', 'နေ', 'လာ', 'တဲ့', 'ကိုယ်စားလှယ်လောင်း', 'ဖြစ်', 'ပါ', 'တယ်', '။']`  

---

## Task 2 - Experiment with Attention Mechanisms

### Performance Table
| Attention Type         | Train Loss | Train PPL | Validation Loss | Validation PPL | Time Taken |
|-------------------------|------------|-----------|-----------------|----------------|------------|
| General Attention       | 1.402      | 4.062     | 4.661           | 105.730        | 1m 8s      |
| Additive Attention      | 2.359      | 10.583    | 4.124           | 61.791         | 1m 39s     |
| Multiplicative Attention| 2.371      | 10.710    | 3.994           | 54.298         | 1m 6s      |

### Performance Plots
- **[General Attention Loss](images/general.png)**
- **[Additive Attention Loss](images/additive.png)**
- **[Multiplicative Attention Loss](images/multiplicative.png)**

### Attention Maps
- **[General Attention Map](images/general_att.png)**
- **[Additive Attention Map](images/additive_attention.png)**
- **[Multiplicative Attention Map](images/multiplicative_att.png)**

---

## Task 3 - Evaluation and Model Selection

### Evaluation in Terms of Scores
- **Validation Loss and PPL**: Multiplicative attention achieves the lowest validation loss (3.994) and perplexity (54.298), indicating better generalization compared to the other mechanisms.
- **Training Scores**: General attention shows the lowest training loss (1.402), but its validation performance is significantly poorer.
- **Time Efficiency**: Multiplicative attention is the fastest to train (1m 6s), making it more computationally efficient.

### Effectiveness
- Multiplicative attention balances good training performance and exceptional validation results. It also demonstrates the lowest validation perplexity, which is crucial for practical applications.

### Choosing Multiplicative Attention
- Multiplicative attention is chosen as the best model because it achieves the best validation scores while being computationally efficient. Its consistent performance indicates robustness, making it a reliable choice for real-world translation tasks.

---

## Task 4 - Web Application

### How to run
1. Ensure all dependencies are installed.
2. Navigate to the `app` folder.
3. Run the command `streamlit run app.py`.
4. Open [http://localhost:8501/](http://localhost:8501/) in your browser.
5. **[Screenshots of testing the web app](images/sample.png)**


### Usage
- **Input**: Once the web app is running, a textbox will appear for you to type any English text.
- **Output**: Click the "Translate" button to see the translated Myanmar text displayed below.

### Documentation

#### Model Architecture
The application uses a Seq2Seq Transformer model with Multiplicative Attention. The model consists of:
- **Encoder**: Processes the input English text sequence.
- **Decoder**: Generates the translated Myanmar text sequence.
- **Attention Mechanism**: Ensures relevance between source and target sequences.

#### Model Input and Output
- **Input**: English sentence entered by the user, tokenized and converted into tensor form.
- **Output**: Translated Myanmar sentence generated by the model.

#### Model Parameters
- Embedding dimensions, number of layers, heads, positional encoding, and feedforward dimensions are optimized for efficiency.
- Special tokens (`<sos>`, `<eos>`, `<pad>`, `<unk>`) are incorporated to handle sequence boundaries and unknown words.

#### Model Processing
- **Tokenization**: Myanmar text is tokenized using the `pyidaungsu` library to ensure accurate word-level processing.
- **Vocabulary Transformation**: Words are mapped to indices using pre-built vocabularies for both English and Myanmar.
- **Inference**: The Seq2Seq Transformer processes the input to produce the translated output in an end-to-end manner.

#### Streamlit Integration
- **User Input**: Taken through Streamlit text input widgets.
- **Output Display**: Generated text is dynamically displayed based on the model's predictions.
