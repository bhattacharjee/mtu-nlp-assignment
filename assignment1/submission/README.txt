DESCRIPTION OF FILES:

1. baseline_comparison.ipynb
   -------------------------
    Data was cleaned, vectorized, and host of different classifiers
    were tried to determine which ones were most promising

2. best_model.ipynb
   ----------------
    The most promising models from baseline_comparison.ipynb were
    tuned for the best hyperparameters. Also, additional data from
    GermEval2021 was used.

3. bert.ipynb
   ----------
    A Transformers model was trained with two different configurations
    - Dense transformer after BERT
    - CNN after BERT

4. parse_germeval2018.py
   ---------------------
    Python script to read the data from GermEval2018 and convert it to 
    the format of GermEval2021

5. GermEval 2018 data
   ------------------
    Relabled for toxicity
    https://raw.githubusercontent.com/bhattacharjee/mtu-nlp-assignment/main/assignment1/germeval2018_a.txt
	

6. NLP-Assignment1-LitReview.docx
   ------------------------------
    Report document
