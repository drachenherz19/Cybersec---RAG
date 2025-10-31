from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TFAutoModelForCausalLM
from datasets import load_dataset
import tensorflow as tf


# Step 1: Load the dataset
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# Step 2: Initialize the tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(preprocess_function, batched=True)


# Convert dataset to TensorFlow format
def convert_to_tf_dataset(dataset, batch_size=8):
    return dataset.to_tf_dataset(
            columns=['input_ids', 'attention_mask'],
            label_cols='labels',
            shuffle=True,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer)
            )


train_dataset = convert_to_tf_dataset(tokenized_dataset["train"])
test_dataset = convert_to_tf_dataset(tokenized_dataset["test"])

# Step 3: Load the model with TensorFlow
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Step 4: Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Step 5: Train the model
model.fit(train_dataset, validation_data=test_dataset, epochs=3)

# Step 6: Save the model
model.save_pretrained("fine_tuned_model_tf")
tokenizer.save_pretrained("fine_tuned_model_tf")


# Load the fine-tuned model
model_path = "fine_tuned_model_tf"
model = TFAutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def extract_keywords(input_text):
    inputs = tokenizer(input_text, return_tensors="tf")
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the keyword extraction
user_input = input("Enter a cybersecurity threat description:\n")
print("Extracted Keywords:", extract_keywords(user_input))