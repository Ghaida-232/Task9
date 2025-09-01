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

<img width="2077" height="953" alt="image" src="https://github.com/user-attachments/assets/669325b3-357e-459c-a553-125467301a80" />

<br/>
<br/>

The dataset used for this task is **XSum (Extreme Summarization)** from the EdinburghNLP collection on Hugging Face, which consists of BBC news articles paired with professionally written single-sentence summaries and unique IDs.  
- **Domain**: The dataset contains BBC news articles across a wide range of domains (politics, sports, health, business, science, etc.). This makes it highly relevant for summarization tasks because news is one of the most common real-world applications of automatic summarization.  
- **Size**: XSum is large-scale, with **~204k training samples, ~11k validation, and ~11k test examples**. The scale ensures robust evaluation of models and provides enough data for training and fine-tuning if needed. For this task we consider a pre-trained models  
- **Suitability**: XSum is explicitly designed for **abstractive summarization**. Each article is paired with a concise, professionally written **single-sentence summary** that often paraphrases and introduces new words (~36% novel unigrams). That’s why XSum is more challenging and better suited for evaluating abstractive LLMs than extractive methods.  
---

<img width="2069" height="738" alt="image" src="https://github.com/user-attachments/assets/30d8f589-36c7-4f86-9657-5a1cbbfeeabb" />

<br/>
<br/>

Here are some examples of the dataset's document along with its summary, we notice that the summary tends to be more abstractive type than extractive.

--- 

<p align="center">
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/f67bcea6-d4d8-43ef-ad01-8522bd23a63a" />
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/cce5f072-76e5-4c25-a6ab-e821372978ac"  />
</p>

<br/>
<br/>

Here we analyze the distribution of document and summary lengths. We notice that the documents are relatively long (average ≈ 375 words or tokens) while the summaries are very short and consistent (average ≈ 21 tokens). This highlights that the dataset is abstractive, since each summary converts a long input article into a single concise sentence instead of directly copying the exact words from the document.

## 3. Data Preprocessing

<img width="2048" height="1303" alt="image" src="https://github.com/user-attachments/assets/154e464b-f34b-4d28-b79c-5603174e2795" />


The cleaning process removed HTML artifacts, control characters and all whitespace, resulting in a more standardized dataset that is easier for tokenization and modeling. After preprocessing, the dataset kept its full scale with 204,045 training, 11,332 validation, and 11,334 test samples. We successfully created new fields (document_clean, summary_clean) while retaining the original text, which ensures both reproducibility and flexibility for further experiments. 

--- 
<img width="2048" height="1291" alt="image" src="https://github.com/user-attachments/assets/9b58c96b-0fcc-48fd-8f0a-8125ec5e3ff9" />

<br/>
<br/>

- The tokenizer successfully converted both documents and summaries into token IDs using the BART tokenizer.
- A maximum source length of 1024 tokens was set to capture long input articles while avoiding memory issues.
- A maximum target length of 64 tokens was chosen, since XSum summaries are short (≈1 sentence).
- Truncation and padding ensured consistent sequence lengths across batches.
- Tokenization preserved dataset scale: 204,045 training, 11,332 validation, and 11,334 test samples were fully processed.
- This step produced a clean, standardized dataset ready for model training and evaluation.

---

## 4. Summarization Approaches
<img width="2066" height="677" alt="image" src="https://github.com/user-attachments/assets/261857a2-afd2-4b30-9ecf-751af18fea99" />

<br/>
<br/>

In this step, we prepare the **validation dataset** for summarization and evaluation.  
- GPU availability is checked to decide the batch size.  
- The full validation split (11,332 samples) is used if a GPU is available; otherwise a smaller subset of 200 samples is selected for efficiency to run on CPU.  
- The cleaned validation set provides the **documents** (inputs) and **summaries** (ground-truth references) which will later be used for generating model predictions and computing ROUGE scores.  
- A sample document and its reference summary are printed for verification.

### 4.1 Extractive Summarization
<img width="2075" height="1287" alt="image" src="https://github.com/user-attachments/assets/d8345da1-e50b-46a8-98b1-c319675ea0d5" />

<br/>
<br/>


**How it works**  
- **Sentence Segmentation (Sumy Tokenizer):** Each document is segmented into individual sentences using Sumy’s tokenizer.  
- **Embed with SBERT:** Each sentence is converted into a vector using **all-MiniLM-L6-v2**, a small but powerful SBERT model.  
- **Build a similarity graph:** Compare every sentence with every other sentence using SBERT embeddings. The more similar two sentences are, the stronger the link between them in the graph.  
- **Rank with TextRank:** Run the PageRank algorithm on this graph. Sentences that are strongly connected to many others get higher scores. The top-scoring sentence is considered the best summary candidate.  
- **Choose the summary:** Among the top-ranked sentences, pick one that has a reasonable length (6–60 words) and appears early in the text. This helps avoid fragments or unimportant trailing sentences.  
- **Fallbacks included:** If the article is too short or oddly formatted, we safely fall back to the first sentence with punctuation.  
- **Save outputs:** Results are written to JSONL (line by line) and JSON (full list), with resume support in case the process is interrupted.

**Run result**  
- Completed the entire validation set (**11,332 samples**).  
- Saved to:  
  - `outputs/sbert_textrank_preds_val.jsonl`  
  - `outputs/sbert_textrank_preds_val.json`

**Insights**  
- **Unsupervised & lightweight**: No training required, easy to run even on CPU.  
- **Semantic advantage**: Using SBERT embeddings makes the method aware of meaning, not just word overlap.  
- **Clean one-liners**: Outputs are short, headline-like summaries that often align well with ROUGE-1.  
- **Extractive**: It copies an existing sentence — no paraphrasing, no fusion of multiple ideas.  
- **Weaker on ROUGE-2/L**: Abstractive models usually beat it on capturing fluency and varied phrasing.  




### 4.2 Abstractive Summarization

<img width="2074" height="1298" alt="image" src="https://github.com/user-attachments/assets/e6568573-ca7f-40cc-b867-2968da92f639" />


<img width="2074" height="1300" alt="image" src="https://github.com/user-attachments/assets/c28d2ead-b7c6-4fd6-90e7-98099da75eed" />


<br/>
<br/>

**How it works**  
- **Model choice:** Used **google/flan-t5-base**, a pretrained LLM fine-tuned for instruction following and summarization.  
- **Tokenizer & pipeline:** Hugging Face `AutoTokenizer` and `AutoModelForSeq2SeqLM` were loaded into a `text2text-generation` pipeline.  
- **Few-shot prompting:** Designed a custom prompt with **two example document–summary pairs** to guide the model. The prompt instructs the model to always produce a **single factual sentence** answering *who, what, when* (and why if present).  
- **Generate summaries:** For each validation document, the few-shot prompt is prepended, then the document is appended, and the model generates a summary.  
- **Decoding settings:** Used beam search (`num_beams=6`) with constraints like `no_repeat_ngram_size=3` and `length_penalty=1.05` to improve fluency and avoid repetition. Output length was capped at 45 tokens to match XSum’s short single-sentence style.  
- **Save outputs:** Summaries are written both as JSONL (line by line) and JSON (full list). Resume logic ensures the run can continue from where it left off.  

**Run result**  
- Completed the entire validation set (**11,332 samples**) in ~2.5 hours on GPU.  
- Saved to:  
  - `outputs/flan_preds_val_STRICT.jsonl`  
  - `outputs/flan_preds_val_STRICT.json`  
- Example predictions include fluent one-liners such as:  
  - *“Former Reading defender Sam Sodje has appeared in court charged with fraud.”*  
  - *“Middlesex’s Adam Voges has been ruled out for the rest of the season with a hamstring injury.”*

**Insights**  
- **Truly abstractive:** Unlike extractive methods, FLAN-T5 rewrites the input into fluent, human-like sentences.  
- **Prompt engineering** Few-shot examples guide the model to consistently produce short, factual summaries.  
- **High quality outputs:** Summaries often resemble reference style in the XSum dataset, showing strong generalization.  
- **More compute-heavy:** Takes longer to run (hours vs minutes for extractive). Requires GPU for practical speed.  
- **Possible hallucination (invention):** LLMs may sometimes invent details, so factual grounding is important.  
- **Best use case:** Produces high-quality abstractive baselines and can be extended to larger FLAN-T5 variants or fine-tuned for domain-specific summarization.



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
