import datasets
import numpy as np
import tensorflow as tf
import transformers
from transformers import TFAutoModelForSequenceClassification

RANDOM_SEED = 331
np.random.seed(RANDOM_SEED)




checkpoint = "distilbert-base-uncased"
tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

train_ds = datasets.Dataset.from_pandas(df_train[['comment_text','target']])
valid_ds = datasets.Dataset.from_pandas(df_valid[['comment_text','target']])


def tokenize_function(example):
    return tokenizer(example["comment_text"], truncation=True)

tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
tokenized_valid_ds = valid_ds.map(tokenize_function, batched=True)

batch_size = 16

data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

tf_train_ds = tokenized_train_ds.to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols=["target"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)

tf_valid_ds = tokenized_valid_ds.to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols=["target"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=batch_size,
)



model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)


num_epochs = 2
num_train_steps = len(tf_train_ds) * num_epochs

lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

fit_history = model.fit(tf_train_ds,
                        epochs=1,
                        validation_data=tf_valid_ds,
                        verbose=1)