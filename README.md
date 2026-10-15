# Titanic survival prediction

My first machine learning project: the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic). The task is to predict which passengers survived the sinking from data like age, sex, ticket class and family aboard.

## What I did

- Explored the data to see which groups survived more (sex and class matter a lot, and class matters differently for men and women)
- Filled missing values: age using the median for each title, fare using the median for each class, port of embarkation using the most common one
- Built a few features:
  - **Title** extracted from the name (Mr, Mrs, Miss, Master, Rare)
  - **HasCabin** whether a cabin number was recorded
  - **FamilySize** and **IsAlone** from the sibling/spouse and parent/child counts
  - **TicketFreq** how many passengers shared the same ticket
  - **FarePerPerson** since the fare is the total for the whole ticket
- One-hot encoded the categorical columns
- Trained a random forest and evaluated it with 5-fold stratified cross-validation

## Next steps i want to add in the future

- Try a shallower forest or fewer features
- Add a feature based on whether a passenger's family or ticket group survived
- Put the preprocessing in a scikit-learn Pipeline so every step is refit inside each CV fold

## Running it

Download `train.csv` and `test.csv` from the competition page into a `data/` folder, then run the notebook. Requires pandas, scikit-learn and matplotlib.
