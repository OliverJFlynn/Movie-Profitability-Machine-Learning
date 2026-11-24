# Predicting Movie Profitability With Machine Learning

Can a film's profitability be predicted from things we know before it comes out? This project uses three machine learning models to find out, using budget, director track record, genre, runtime and critical reception (IMDb score and award nominations).

A film counts as **profitable** if its box office is greater than its budget.

<img width="2652" height="2042" alt="Predicting Movie Profitability from Budget" src="https://github.com/user-attachments/assets/fbfbbd54-6a94-44f9-b3ac-3e563244fb9f" />

## Dataset
[Movies (IMDb, Earnings and more)](https://www.kaggle.com/datasets/delfinaoliva/movies) by Delfina Oliva on Kaggle: 3,905 films from 1929 to 2016 with 12 variables after cleaning. The data isn't included here, so download it from the link above and save it as `movies_data_new.csv` next to the notebook.

I removed the actor columns because they don't reliably show a film's real lead cast (Fight Club's "Actor 2" is Meat Loaf, not Brad Pitt), and research suggests actors have little link to financial success.

## What I did
1. **Data exploration:** looked at genre, directors, awards, release year, running time, IMDb score, budget and box office. 73% of films in the dataset were profitable.
2. **Pre-processing:** grouped directors by how many films they made, square-root transformed skewed columns, scaled features to 0–1, split 75/25 into train and test sets, and used SMOTE on the training set to balance profitable and unprofitable films.
3. **Experiments:** trained a Decision Tree, Logistic Regression and Random Forest.

## Results

| Model | Accuracy |
|---|---|
| Decision Tree | 74.00% |
| Logistic Regression | 75.03% |
| Random Forest | 75.54% |

All three models landed within about 1.5% of each other. Random Forest did best, but the simpler Logistic Regression was almost identical, which suggests the link between these features and profitability is mostly linear. All three were better at spotting profitable films than unprofitable ones.

I also tested the models on five films of my own choosing.

## Conclusion
A director's previous box office record was the strongest predictor of profitability, ahead of budget, genre and critical reception. Award nominations and IMDb score also mattered. Budget played a smaller role than I expected, and genre a smaller one still.

So a production company should weight a director's track record more heavily than budget when judging a film's commercial potential.

**Limitation:** "Director Box Office %" is built from box office figures, so it may leak some information about the outcome. Removing it and re-testing would be a good next step.

## Tools
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, imbalanced-learn

## Run it
```
pip install -r requirements.txt
jupyter notebook Movie_Profitability.ipynb
```
