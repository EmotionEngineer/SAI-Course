# 7. Recommender Systems — Homework

**Use the notebook:** “VK Recommendation Model” with RecBole and GRU4Rec.

## Assignment
1. Run the notebook in Kaggle with the VK dataset available. Check that the input files are accessible and the required libraries are installed.
2. In the data-loading section, display a few rows before and after transforming `status`. Note which interaction types were assigned numerical ratings.
3. Trace which columns the notebook saves to the interaction files and which to the item file. Create the files without changing their format.
4. Run GRU4Rec training and save the result. If training is impossible because of unavailable data or resources, include the error output and explain at which step it occurred.
5. Open the created `result.csv`: state the number of rows and column names, and show a few recommendations.

## Answer
- What represents a user and what represents a recommended item in this task?
- Why does the model need a user's past interactions?

You do not need to implement another model or calculate additional metrics.