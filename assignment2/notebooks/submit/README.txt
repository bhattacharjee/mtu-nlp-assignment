Data from three strategies is shared:
1. baseline
2. baseline with augmented data (approximately doubles the size of the data)
3. augmented data with gradual unfreezing

All runs were done in Google Colab using high-ram instances and nVidia P100
GPUs.

The Jupyter Notebooks have been saved with their outputs from the runs, and
can be opened locally for viewing.

The folder structure is as follows:
1. baseline - files from the baseline
2. augmentation - files from the run with augmented data
3. augmentation-with-gradual-unfreezing - files with augmentation and gradual
                                          unfreezing
4. ChunkData - contains the code that was written for a disk-backed array
               that doesn't put pressure on RAM


Report
------
The report is present in the following file:
Rajbir-Bhattacharjee-R00195734-report-nlp-a2.docx





Files in folder: baseline
-------------------------
Log file:
baseline/log_train.txt

CSV after evaluation on test set:
baseline/submission.csv

Jupyter notebook that includes the output from the run
baseline/MTU_NLP_Assignment_2_Question_Answering_wip.ipynb







Files in folder: augmentation
-----------------------------
Log file:
augmentation/log_train.txt

CSV after evaluation on test set:
augmentation/submission.csv

Jupyter notebook that includes the output from the run (after augmentation and tokenization)
augmentation/MTU_NLP_Assignment_2_Question_Answering_augmented_wip.ipynb

Jupyter notebook used to perform back-translation and random deletion and swap for augmentation
augmentation/Augmentation_wip.ipynb

Jupyter notebook used to tokenize the augmented data (it had to be done separately to avoid Colab timeouts)
augmentation/Augmentation_preprocess_temp.ipynb






Files in folder: augmentation-with-gradual-unfreezing
-----------------------------------------------------
Log file:
augmentation-with-gradual-unfreezing/log_train.txt

CSV after evaluation on test set:
augmentation-with-gradual-unfreezing/MTU_NLP_Assignment_2_QA_augmented_gradual_unfreezing_wip.ipynb

Jupyter notebook that includes the output from the run
augmentation-with-gradual-unfreezing/submission.csv

Jupyter notebook used to perform back-translation and random deletion and swap for augmentation (identical to the previous one)
augmentation-with-gradual-unfreezing/Augmentation_wip.ipynb

Jupyter notebook used to tokenize the augmented data (it had to be done separately to avoid Colab timeouts) (identical to the previous one)
augmentation-with-gradual-unfreezing/Augmentation_preprocess_temp.ipynb






ChunkData
---------
ChunkData/__pycache__
ChunkData/__pycache__/ChunkData.cpython-39.pyc
ChunkData/TestChunkData.py
ChunkData/ChunkData.py






