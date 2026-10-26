# 12. Additional Topics — Homework

Complete assignments for both notebooks in this section. If compute resources are limited, ask your instructor whether you may complete only one.

## A. Custom Loss Functions
**Notebook:** Custom Loss Functions for CatBoost and TensorFlow.

1. Run the synthetic-data creation and CatBoost training with the custom MAE Log10 loss function.
2. Run the TensorFlow section using the same loss function.
3. Plot the training and validation losses as in the example.
4. Record the values shown in the plot and describe how the losses change during training.

**Answer:** What does a loss function define during training? Why does the notebook compare training and validation?

## B. Optimized Large Data Loading
**Notebook:** Processing Large CSV Files in Python.

1. Run the `process_large_csv_in_batches` example on an available CSV/TSV file. If the specified file is unavailable, use another suitable file and state its name.
2. Note which columns are dropped, the specified batch size, and where the result is saved.
3. Run the `sample_csv` example for sequential or random sampling (one mode is enough).
4. Check the created file: number of rows, column names, and a few first rows.

**Answer:** Why is the file processed in chunks? How does the sampling mode you chose differ from the other mode shown in the notebook?

No code optimization or speed/memory benchmarking is required.