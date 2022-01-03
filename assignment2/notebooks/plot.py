import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re

def get_dataframe(filename:str):
    f1_scores = []
    em_scores = []
    iterations = []
    epochs = []
    epoch_nums = []
    highest_iter = 0
    with open(filename, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        cpat = re.compile(".*F1:\s(\d+.\d+),\sEM:\s(\d+.\d+)")
        cpat2 = re.compile("100\%\s(\d+)/(\d+)")
        
        # Find out the number of examples
        num = 0
        num_iter = 0
        for line in lines:
            m = re.match(cpat2, line)
            if m is not None:
                num += 1
                n_examples = m.group(2)
                examples_per_epoch = m.group(2)
                epoch_nums.append(num - 1)
                epochs.append(int(n_examples) // 32 * num)
                highest_iter = int(n_examples) // 32 * num

            m = re.match(cpat, line)
            if m is not None:
                theiter = min(highest_iter, num_iter * 2500)
                iterations.append(theiter)
                f1_scores.append(float(m.group(1)))
                em_scores.append(float(m.group(2)))
                num_iter += 1
                
        f1_scores = pd.Series(f1_scores)
        em_scores = pd.Series(em_scores)
        epochs = pd.Series(epochs)
        return pd.DataFrame({
                    "ITER": iterations,
                    "F1": f1_scores,
                    "EM": em_scores
                }),\
                pd.DataFrame({"EPOCH": epoch_nums, "ITER": epochs})
                
plt.rcParams['figure.figsize'] = (20, 10)
inlabel = "Unfrozen"

def plot_f1_scores():
    for filename in glob.glob("./*.txt"):
        df1, df2 = get_dataframe(filename)
        sns.lineplot(data=df1, x="ITER", y="F1", label=f"F1 - {filename}", ci=None)
    for i in df2['ITER']:
        plt.vlines(i, 0, 100, color='black')
        plt.text(i + 55 , 60,f"epoch - {i}", rotation=270)
    plt.show()

def plot_em_scores():
    for filename in glob.glob("./*.txt"):
        df1, df2 = get_dataframe(filename)
        sns.lineplot(data=df1, x="ITER", y="EM", label=f"EM - {filename}", ci=None)
    for i in df2['ITER']:
        plt.vlines(i, 0, 100, color='black')
        plt.text(i + 55 , 40,f"epoch - {i}", rotation=270)
    plt.show()

    
plot_f1_scores()
plot_em_scores()
