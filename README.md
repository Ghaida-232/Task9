# Summarization using LLMs (Extractive vs. Abstractive)

## 1. Objective
The goal of this task was to explore **automatic text summarization** using two different approaches:  
- **Extractive summarization**: selecting key sentences directly from the source text.  
- **Abstractive summarization**: generating new sentences using **pretrained Large Language Models (LLMs)**.  

Specifically, the objectives included:  
- Applying both extractive and abstractive methods.  
- Utilizing LLMs such as **BART**, **T5**, or **LLaMA-2** for abstractive summarization.  
- Evaluating model performance using **ROUGE metrics**.  
- Comparing machine-generated summaries with human references.  
- Documenting the workflow.

---

## 2. Dataset Selection
The dataset used for this task is **XSum (Extreme Summarization)** from the EdinburghNLP collection on Hugging Face, which consists of BBC news articles paired with professionally written single-sentence summaries and unique IDs.  
- **Domain**: The dataset contains BBC news articles across a wide range of domains (politics, sports, health, business, science, etc.). This makes it highly relevant for summarization tasks because news is one of the most common real-world applications of automatic summarization.  
- **Size**: XSum is large-scale, with **~204k training samples, ~11k validation, and ~11k test examples**. The scale ensures robust evaluation of models and provides enough data for training and fine-tuning if needed. For this task we consider a pre-trained models  
- **Suitability**: Unlike extractive-friendly datasets (e.g., CNN/DailyMail), XSum is explicitly designed for **abstractive summarization**. Each article is paired with a concise, professionally written **single-sentence summary** that often paraphrases and introduces new words (~36% novel unigrams). This makes it an excellent benchmark for testing how well LLMs can generate fluent, human-like summaries rather than copying sentences from the source.  
---

## 3. Data Preprocessing
- Removed HTML tags, escape characters, and special symbols.  
- Normalized whitespace and truncated overly long articles to improve efficiency.  
- Applied **tokenization** using the target model’s tokenizer (e.g., `BartTokenizer`, `T5Tokenizer`).  
- Stored cleaned samples into a structured CSV for reproducibility.

---

## 4. Methodology

### 4.1 Extractive Summarization
- Implemented **TextRank** and embedding-based ranking methods.  
- Extracted top-ranked sentences as summaries.  
- Pros: fast, unsupervised, no model training needed.  
- Limitation: lacks paraphrasing and abstraction.

### 4.2 Abstractive Summarization
- Used pretrained **LLMs** (e.g., `facebook/bart-large-cnn`, `t5-base`, `meta-llama/Llama-2-7b-chat-hf`).  
- Applied **prompt engineering** to guide summarization style and length.  
- Generated concise summaries beyond direct sentence extraction.

### 4.3 Optional Enhancements
- Explored **LangChain pipelines** for structured summarization prompts.  
- Considered **RAG (Retrieval-Augmented Generation)** for domain-specific summarization.

---

## 5. Evaluation

### Metrics
- **ROUGE-1**: unigram overlap.  
- **ROUGE-2**: bigram overlap.  
- **ROUGE-L**: longest common subsequence.  

### Process
- Compared generated summaries against human references.  
- Visualized results in **bar charts** and **comparison tables**.  
- Stored outputs in `outputs/generated_summaries.json`.  

### Example Output (Illustrative)
**Input text (excerpt):**  
> "Former Premier League footballer Sam Sodje has appeared in court alongside three brothers accused of charity fraud."

**Reference summary:**  
> "Former footballer Sam Sodje has appeared in court over fraud charges."

- **Extractive summary:** selected first sentence directly.  
- **Abstractive summary:** "Ex-Premier League player Sam Sodje faces fraud trial with brothers."

---

## 6. Results & Visualizations

- **Figure 1: ROUGE Scores Comparison**  
<img width="750" height="420" alt="rouge_scores" src="figures/rouge_scores.png" />  
<br/>  
*The bar chart above compares ROUGE-1, ROUGE-2, and ROUGE-L scores across extractive and abstractive methods.*  

**Insights:**  
- Abstractive models (BART, T5) consistently outperformed extractive baselines in ROUGE-1 and ROUGE-2.  
- ROUGE-L scores indicated that abstractive models better preserved sequence-level relevance.  
- Extractive methods scored lower but provided stable baselines for comparison.  

---

- **Figure 2: Example Summaries**  
<img width="750" height="420" alt="summary_examples" src="figures/summary_examples.png" />  
<br/>  
*The figure shows qualitative comparisons of input, human-written reference, extractive summary, and abstractive summary.*  

**Insights:**  
- Extractive summaries often repeated long sentences from the source.  
- Abstractive models generated **concise, paraphrased outputs** closer to human style.  
- The qualitative difference highlighted the advantage of LLMs in producing natural and readable summaries.

---

## 7. Key Outcomes & Insights
- **Extractive methods** produced grammatically correct but sometimes lengthy summaries.  
- **Abstractive LLMs** generated concise and human-like paraphrases.  
- **BART** and **T5** achieved higher ROUGE scores compared to simple extractive baselines.  
- Visual comparisons highlighted that abstractive models captured meaning better, while extractive models remained closer to the original text.  
- ROUGE-L scores confirmed that abstractive models maintained sequence relevance with fewer words.  

---

## 8. Project Structure
```
project-root/
│
├── data/
│ └── preprocessed_xsum_samples.csv
│
├── notebooks/
│ └── summarization_analysis.ipynb
│
├── outputs/
│ └── generated_summaries.json
│
├── figures/
│ ├── rouge_scores.png
│ └── summary_examples.png
│
├── requirements.txt
│
└── README.md
```
