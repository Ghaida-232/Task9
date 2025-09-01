# Summarization using LLMs (Extractive vs. Abstractive)

## 1. Objective
The goal of this task was to explore **automatic text summarization** using two different approaches:  
- **Extractive summarization**: selecting key sentences directly from the source text.  
- **Abstractive summarization**: generating new sentences using **pretrained Large Language Models (LLMs)**.  

Specifically, the objectives included:  
- Apply both extractive and abstractive summarization techniques.
- Utilize pretrained large language models (LLMs) for abstractive summarization.
- Evaluate model performance using standard metrics.
- Compare model-generated summaries to human-written references.
- Save the model and document the process in a notebook and README.md.

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
<img width="2071" height="1057" alt="image" src="https://github.com/user-attachments/assets/f24bae1f-158a-4629-870a-bfdf4b2307ea" />

<br/>
<br/>

First, we load the generated summaries for both approaches:  
- **Extractive outputs** from `sbert_textrank_preds_val.json/jsonl`.  
- **Abstractive outputs** from `flan_preds_val_STRICT.json/jsonl`.  

These files contain the predicted summaries for the validation set.  
We then align them with the reference summaries from the dataset to ensure all three lists (documents, references, and predictions) have the same size (**N = 11,332**).  This prepares the data for evaluation using ROUGE and qualitative comparison.

### Metrics
<img width="2071" height="832" alt="image" src="https://github.com/user-attachments/assets/31768199-7513-46a8-bb9f-82e6941fe241" />

<br/>
<br/>

To measure the quality of generated summaries, we use the **ROUGE metric** (Recall-Oriented Understudy for Gisting Evaluation).  
Specifically:  
- **ROUGE-1:** Overlap of unigrams (single words).  
- **ROUGE-2:** Overlap of bigrams (two-word sequences).  
- **ROUGE-L:** Longest common subsequence (measures fluency and structure).

**Workflow**  
1. Define a helper function `avg_rouge()` to calculate average ROUGE scores across all validation samples.  
2. Compare predictions against reference summaries for both extractive and abstractive methods.  
3. Store results in a dataframe for easy comparison.

**Results (Validation Set, N=11,332)**  

| Method                     | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-----------------------------|---------|---------|---------|
| Extractive (SBERT TextRank) | 0.1834  | 0.0260  | 0.1312  |
| Abstractive (FLAN-T5)       | 0.3536  | 0.1338  | 0.2803  |


**Insights**  
- **FLAN-T5 (abstractive)** clearly outperforms SBERT TextRank across all ROUGE metrics.  
- Higher **ROUGE-2** indicates that abstractive summaries capture phrase-level meaning more accurately.  
- Higher **ROUGE-L** suggests that abstractive summaries better preserve the sequence and overall structure of the reference summaries.  
- **SBERT TextRank** remains a useful **baseline**, but its lower scores reflect the limitation of copying sentences directly, especially since the dataset used (XSum) is designed for abstractive summaries.
  

### Visualization of ROUGE Scores

The following figures compare **Extractive (SBERT TextRank)** and **Abstractive (FLAN-T5)** summarization methods using ROUGE metrics:

- **Figure 1: ROUGE-1 (F1)**  
  <img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/7c1b35d4-311e-4e68-8582-97272945cb26" />
  
  *Abstractive FLAN-T5 achieves a score of 0.354 compared to 0.184 for extractive, showing stronger unigram overlap with reference summaries.*

<br/>

- **Figure 2: ROUGE-2 (F1)**  
  <img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/beba21d0-88c3-401e-b365-3ec550a5e54d" />
  
  *FLAN-T5 again outperforms extractive with 0.134 vs 0.025, demonstrating better capture of phrase-level meaning.*

  <br/>

- **Figure 3: ROUGE-L (F1)**  
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/ca5a5fc2-4ee4-448e-9b61-170c72527491" />

  *FLAN-T5 achieves 0.280 compared to 0.133 for extractive, indicating better preservation of overall sequence and structure.*

  <br/>

**Insights**  
- The bar charts confirm that **abstractive summarization consistently outperforms extractive summarization** across all ROUGE metrics.  
- The gap is especially large for **ROUGE-2**, highlighting the abstractive model’s ability to generate coherent phrases rather than just selecting words.  
- Extractive summarization still provides a **baseline**, but the XSum dataset favors abstractive models since its reference summaries are highly abstractive.


### Example Output 
To better understand the difference between methods, we compare **Input documents**, **Reference (human) summaries**, **Extractive (SBERT-TextRank)** outputs, and **Abstractive (FLAN-T5)** outputs on random validation samples.

<img width="2048" height="1286" alt="image" src="https://github.com/user-attachments/assets/8a644579-7b21-42fa-b2d9-6e0547244edc" />
<img width="2048" height="1260" alt="image" src="https://github.com/user-attachments/assets/1e9d6e46-66be-45b4-8b37-c3ade53970de" />

<br/>
<br>
Here are some exapmles from the figures for better vision:

| id | Input (truncated) | Reference | Extractive | Abstractive |
|----|-------------------|-----------|------------|-------------|
| 0  | The ex-Reading defender denied fraudulent trading charges relating to the Sodje Sports Foundation ... | Former Premier League footballer Sam Sodje has appeared in court alongside three brothers accused of charity fraud. | The ex-Reading defender denied fraudulent trading charges relating to the Sodje Sports Foundation – a charity to raise money for Nigerian sport. | Former Reading defender Sam Sodje has appeared in court charged with fraud. |
| 1  | Middlesex hope to have the Australian back for their T20 Blast game ... Adam Voges ... injured calf muscle ... | Middlesex batsman Adam Voges will be out until August after suffering a torn calf muscle in his right leg. | Voges retired from international cricket in February with a Test batting average of 61.87 from 31 innings, second only to Australian great Sir Donald Bradman's career average of 99.94 from 52 Tests. | Middlesex’s Adam Voges has been ruled out for the rest of the season with a hamstring injury. |
| 2  | Duchess of Cambridge featured in Vogue magazine ... 100 years anniversary ... | The Duchess of Cambridge will feature on the cover of British Vogue to mark the magazine’s centenary. | He said the images also encapsulated what Vogue had done over the past 100 years - "to pair the best photographers with the great personalities of the day, in order to reflect broader shifts in culture and society" | The Duchess of Cambridge is to feature in British Vogue’s 100th issue for the first time. |
| 3  | Google hires creator of 4chan website ... | Google has hired the creator of one of the web’s most notorious forums – 4chan. | He added: "I can't wait to contribute my own experience from a dozen years of building online communities, and to begin the next chapter of my career at such an incredible company." | Google has appointed a former administrator of the controversial 4chan website as its new chief executive. |
| 4  | Two teenagers charged in connection with Belfast car crash involving police officers ... | Two teenagers have been charged in connection with an incident in west Belfast in which a car collided with two police vehicles. | A man, aged 19, and a boy, aged 16, have been charged with six counts of aggravated vehicle taking. | Two teenagers have been charged with aggravated vehicle taking after two police officers were injured in a car crash in Belfast. |

**Insights**  
- **Reference summaries** are short, highly abstractive and paraphrase the source.  
- **Extractive outputs** tend to copy long factual sentences directly, sometimes keeping unnecessary details.  
- **Abstractive outputs** are much closer to the reference style: concise, rephrased and often better aligned with the XSum dataset’s abstractive nature.  
- This confirms why abstractive models like FLAN-T5 outperform extractive methods.



## 6. Project Structure
```
TASK9/
│
├── data/
│ ├── preprocessed_dataset.csv # Cleaned dataset ready for training/evaluation
│ └── xsum_tokenized_bart/ # Tokenized dataset prepared 
│
├── figures/
│ ├── rouge_scores.png # Visualization of ROUGE-1, ROUGE-2, ROUGE-L scores
│ └── summary_examples.png # Table of example summaries (input, reference, extractive, abstractive)
│
├── outputs/
│ ├── flan_preds_val_STRICT.json # Abstractive summaries (full JSON list)
│ ├── flan_preds_val_STRICT.jsonl # Abstractive summaries (JSONL, line by line with resume support)
│ ├── sbert_textrank_preds_val.json # Extractive summaries (full JSON list)
│ └── sbert_textrank_preds_val.jsonl # Extractive summaries (JSONL, line by line with resume support)
│
├── summarization_analysis.ipynb # Main notebook with preprocessing, modeling, evaluation, and
│
└── README.md
```

## 7. How to Run

### 1. **Install requirements**
```bash
pip install transformers datasets sentence-transformers sumy rouge-score
pip install pandas numpy matplotlib tqdm
```
(use GPU-enabled PyTorch if available for faster abstractive runs)

### 2. **Open the notebook**

Run summarization_analysis.ipynb from start to finish.

Steps are organized: preprocessing → tokenization → extractive → abstractive → evaluation → visualization.

### 3. **Outputs**

- Data:

data/preprocessed_dataset.csv

data/xsum_tokenized_bart/

- Predictions:

Extractive → outputs/sbert_textrank_preds_val.json/jsonl

Abstractive → outputs/flan_preds_val_STRICT.json/jsonl

- Figures:

ROUGE scores → figures/rouge_scores.png

Example summaries → figures/summary_examples.png

### 4. **Notes**

If GPU is available, the full validation set (11,332 samples) is used; otherwise, a small subset.

Both extractive and abstractive runs support resume — you can stop and rerun without losing progress.

Evaluation is done with ROUGE-1, ROUGE-2, and ROUGE-L.
