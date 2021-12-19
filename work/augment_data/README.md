## Starter code for the QA robustqa project
- Download datasets within the Canvas folder 'robustqa'
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test (or Dev) set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`

