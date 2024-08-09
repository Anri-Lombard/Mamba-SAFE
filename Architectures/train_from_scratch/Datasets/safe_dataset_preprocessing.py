from datasets import load_dataset

dataset = load_dataset("datamol-io/safe-gpt")

# Select 100000000 random samples from the dataset for train set
dataset_train = dataset["train"].shuffle(seed=42).select(range(100000000))

# Select 100000 random samples from the dataset for test set
dataset_test = dataset["test"].shuffle(seed=42).select(range(1000000))

# Save the dataset to csv files
dataset_train.to_csv("train_safe_dataset.csv")
dataset_test.to_csv("test_safe_dataset.csv")