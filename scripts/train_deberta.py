import os # Facilitates working with folders and filepaths
import numpy as np 
import pandas as pd
import torch # Main PyTorch library for deep learning
import json
import torch.nn as nn # Used for neural network tools like loss functions

from datasets import load_dataset # Loads datasets from Hugging Face
from transformers import ( # Imports tools from Hugging Face Transformers
    AutoTokenizer, # Loads the appropriate tokenizer for the model
    AutoModelForSequenceClassification, # Loads a transformer model used for classification
    TrainingArguments, # Used to store training settings like epochs and batch size
    Trainer, # Used to handle training and evaluation loop
    DataCollatorWithPadding, # Used to pad batches so inputs have matching lengths
    set_seed # Uses the same data split each run
)

from sklearn.metrics import ( # Imports evaluation metrics from scikit-learn
    accuracy_score, # Used to calculate overall accuracy of the prediction
    precision_recall_fscore_support, # Used to calculate precision, recall, and F1-score
    classification_report, # Used to create a detailed performance report by class
    confusion_matrix # Used to identify where model is making incorrect and correct predictions
)

from sklearn.utils.class_weight import compute_class_weight # Used to create class weights to handle label imbalances

def main(): # Prevents the full training process from starting unintentionally
    
    seed = 52 # Ensures reproducible results
    set_seed(seed) # Sets random seeds for training

    device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available, otherwise use CPU

    print("This training script  will use this device:", device) # Prints whether GPU or CPU is being used
    print("CUDA available for use:", torch.cuda.is_available()) # Checks for CUDA/GPU support

    if torch.cuda.is_available(): # Runs if CUDA-compatible GPU is found
        print("GPU being used is:", torch.cuda.get_device_name(0)) # Prints the name of the GPU identified

    dataset = load_dataset("cnamuangtoun/resume-job-description-fit") # Loads the resume-job description dataset from Hugging Face

    print(dataset) # Prints the dataset structure, including number of rows and train/test splits

    print(dataset["train"].column_names) # Prints the column names in the training dataset




    train_valid_split = dataset["train"].train_test_split(test_size = 0.1, seed = seed) # 90% Training and 10% validation, repeatable splits

    train_dataset = train_valid_split["train"] # Stores the training dataset
    valid_dataset = train_valid_split["test"] # Stores the validation dataset
    test_dataset = dataset["test"] # Uses the datasets original test split for the final evaluation

    print("The Training dataset size is:", len(train_dataset)) # Prints the number of training examples
    print("The Validation dataset size is:", len(valid_dataset)) # Prints the number of validation examples
    print("The Test dataset size is:", len(test_dataset)) # Print the number of test examples

    # Gets unique labels from the dataset
    unique_labels = sorted(list(set(train_dataset["label"]))) # Sorts unique labels from dataset alphabetically

    print("The original labels are:", unique_labels) # Prints the unique labels

    preferred_order = ["No Fit", "Potential Fit", "Good Fit"] # Setting the label order to be used

    if all(label in unique_labels for label in preferred_order): # Checks if preferred labels match those in the dataset
        label_names = preferred_order # Assigned the preferred labels
    else:
        label_names = unique_labels # Use dataset's labels if preferred labels are missing

    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}

    print("The label to id mapping is:", label2id) # Prints the label to number mapping
    print("The id to label mapping is:", id2label) # Checks model prediction labels match input label mappings

    def encode_labels(example): # Define the function used to convert text labels to numeric labels
        example["labels"] = label2id[example["label"]] # Looks up the number for the current text label
        return example # Returns the updated example with new numeric label column

    train_dataset = train_dataset.map(encode_labels) # Applies the label conversion to the training dataset
    valid_dataset = valid_dataset.map(encode_labels) # Applies the label conversion to the validation dataset
    test_dataset = test_dataset.map(encode_labels)   # Applies the label conversion to the test dataset

    train_labels = train_dataset["labels"] # Stores the numeric training labels, used for calculating the class weights

    class_weights = compute_class_weight( # Calculates the weights for all classes, to handle label distribution imbalances
        class_weight="balanced", # Assigns higher weight to classes with smaller distributions
        classes=np.array([0, 1, 2]), # Lists the numeric class labels
        y=np.array(train_labels) # Provides actual training labels for weight calculation
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float) # Converts class weights into a PyTorch tensor

    print("The class weights are:", class_weights) # Prints the class weights for inspection

    model_checkpoint = "microsoft/deberta-v3-small" # Using the pre-trained DeBERTa v3 small model

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) # Loads the correct tokenizer for DeBERTa

    def encode_and_tokenize_function(example): # Defines the function to tokenize one resume-job pair
        tokenized = tokenizer( # Uses the DeBERTa tokenizer to convert text into inputs for the model
            example["resume_text"], # Resume text input
            example["job_description_text"], # Job description text input
            truncation=True, # Limits the text to maximum text length by cutting excess words
            max_length=512 # Defines maximum length for each resume-job pair
        )
        
        tokenized["labels"] = label2id[example["label"]] # Provides the numerical label for training
        
        return tokenized # Returns the tokenized input and labels


    train_tokenized = train_dataset.map( # Tokenization applied on the training dataset
        encode_and_tokenize_function, # Uses previously defined tokenization function
        remove_columns=train_dataset.column_names # Removes the text columns
    )

    valid_tokenized = valid_dataset.map( # Tokenization applied on the validation dataset
        encode_and_tokenize_function, # Uses previously defined tokenization function
        remove_columns=valid_dataset.column_names # Removes the text columns
    )

    test_tokenized = test_dataset.map( # Tokenization applied on the test dataset
        encode_and_tokenize_function, # Uses previously defined tokenization function
        remove_columns=test_dataset.column_names # Removes the text columns
    )

    print(train_tokenized.column_names) # Prints the column names after tokenization
    

    num_labels = len(label_names) # Counts the number of labels model would need to predict

    model = AutoModelForSequenceClassification.from_pretrained( # Loads the pre-trained transformer model for classification
        model_checkpoint, # Uses the model name stored earlier
        num_labels=num_labels, # Gives the model number of output classes
        id2label=id2label, # Model can now convert numeric predictions into label names 
        label2id=label2id, # Model can now also convert label names into numeric predictions
        dtype=torch.float32,  # Loads FP32 weights for mixed-precision training
    )

    print("Model parameter dtype:", next(model.parameters()).dtype)

    model.to(device) # Model uses GPU if available, otherwise uses CPU

    def compute_metrics(eval_pred): # Defines the function to calculate the model's evaluation scores
        logits, labels = eval_pred # Separating the model's raw outputs and the true labels
        predictions = np.argmax(logits, axis=1) # Selects the class with the highest score as the prediction

        accuracy = accuracy_score(labels, predictions) # Percentage of correct predictions calculated

        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support( # Calculates the weighted precision, recall, and F1 
            labels, # True Labels
            predictions, # Model's Predictions
            average="weighted", # Classes weighted by number of examples
            zero_division=0 # Prevents errors if a class has zero predictions
        )

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support( # This calculates the macro-precision, recall, and F1
            labels, # True Labels
            predictions, # Model's Predictions
            average="macro", # All classes treated equally regardless of class size
            zero_division=0 # Prevents errors if a class has zero predictions
        )

        return { # Returns a dictionary with all the evaluation scores
            "accuracy": accuracy, # Accuracy which is the overall correctness score
            "precision_weighted": precision_weighted, # Precision score across classes, weighted by class size
            "recall_weighted": recall_weighted, # Recall score across classes, weighted by class size
            "f1_weighted": f1_weighted, # F1-score across classes, weighted by class size
            "precision_macro": precision_macro, # Macro precision score
            "recall_macro": recall_macro, # Macro recall score
            "f1_macro": f1_macro # Macro F1-score
        }

    training_args = TrainingArguments( 
        output_dir = "./outputs/deberta/checkpoints", # Folder to save the model checkpoints and training outputs
        eval_strategy = "epoch", # Evaluation is run at the end of every epoch
        save_strategy = "epoch", # Model checkpoint is saved at the end of every epoch
        learning_rate = 1e-5, # Controls how quickly model updates its weights during training
        per_device_train_batch_size = 4, # Number of training examples that are processed at once on a device
        per_device_eval_batch_size = 4, # Number of evaluation examples that are processed at once on a device
        gradient_accumulation_steps = 2,  # Simulates a larger training batch
        num_train_epochs = 8, # Number of full passes through the training dataset
        weight_decay = 0.01, # Regularization is added to reduce overfitting
        warmup_ratio = 0.1, # Learning rate gradually increased during the first 10% of training
        logging_steps = 50, # Training logs printed every 50 steps
        load_best_model_at_end = True, # The best checkpoint reloaded after training
        metric_for_best_model = "eval_loss", # Best model chosen based on validation loss
        greater_is_better = False, # Specifies lower validation loss is better
        report_to = "none", # Ensures weights and biases are not sent to external-tracking tools
        fp16 = False,
        bf16 = torch.cuda.is_available(), # Uses 16-bit numbers if CUDA GPU is available during training, for faster training
        seed = seed,  # Controls training randomness
        data_seed = seed,  # Controls data sampling randomness
        )

    # Weighted Trainer class

    class WeightedLossTrainer(Trainer): # Creates a custom version of Hugging Face's trainer
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs): # **kwargs allows any extra keyword arguments to be accepted
            labels = inputs.get("labels") # Retrieves correct labels for current batch
            outputs = model(**inputs) # Sends the input batch through the model, unpacking the dictionary automatically
            logits = outputs.get("logits") # Gets the model's raw prediction scores

            loss_function = nn.CrossEntropyLoss( # A classification loss function is created
                weight=class_weights.to(logits.device) # Class weights are applied and moved to the same device as the model
            )

            loss = loss_function( # Weighted classification loss is calculated
                logits.view(-1, model.config.num_labels), # Model outputs reshaped into expected format
                labels.view(-1) # Labels reshaped into expected format
            )

            return (loss, outputs) if return_outputs else loss # Returns loss and outputs or loss
        
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Creates the helper which pads each batch to the same length

    trainer = WeightedLossTrainer( # Creates the custom trainer that uses weighted loss 
        model=model, # The DeBERTa classification model is being trained
        args=training_args, # The training settings are applied
        train_dataset=train_tokenized, # Teaches the model using tokenized training data
        eval_dataset=valid_tokenized, # Performance is checked on the validation data using tokenized validation data
        processing_class=tokenizer, # Tokenizer passed to trainer
        data_collator=data_collator, # Batches padded correctly before sending them to the model
        compute_metrics=compute_metrics # Calculates accuracy, precision, recall and F1 during evaluation
    )

    trainer.train() # Starts training the model using the trainer setup, training data and settings.

    test_results = trainer.evaluate(test_tokenized) # Trained model is evaluated on the test dataset

    print(test_results) # Prints the final loss, accuracy, precision, recall and F1-score metrics

    predictions_output = trainer.predict(test_tokenized) # Uses the trained model for predictions on the test dataset

    logits = predictions_output.predictions # Obtains raw prediction scores for each class
    true_labels = predictions_output.label_ids # Gets the actual numeric labels from the test dataset
    predicted_ids = np.argmax(logits, axis=1) # Class with highest score for each example is selected

    predicted_labels = [id2label[i] for i in predicted_ids] # Predicted numeric labels changed into text
    true_label_names = [id2label[i] for i in true_labels] # Converts actual numeric labels into text

    # Prints precision, recall, F1-score, and support for each class

    report = classification_report(true_label_names,
        predicted_labels,
        labels=label_names,
        output_dict=True,
        zero_division=0,
    )

    print(
        classification_report(
            true_label_names,
            predicted_labels,
            labels=label_names,
            zero_division=0,
        ))

    cm = confusion_matrix(true_label_names, predicted_labels, labels=label_names) # Confusion matrix to compare true labels vs predicted labels

    cm_df = pd.DataFrame( # Confusion matrix converted into readable pandas table
        cm, # Confusion matrix values
        index=[f"Actual {label}" for label in label_names], # Row names with actual classes
        columns=[f"Predicted {label}" for label in label_names] # Column names with predicted classes
    )

    results_dir = "outputs/deberta"
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "test_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(test_results, file, indent=2)

    with open(os.path.join(
            results_dir,
            "classification_report.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    cm_df.to_csv(
        os.path.join(
            results_dir,
            "confusion_matrix.csv",
        )
    )

    model_output_dir = "models/trained_deberta_model"

    os.makedirs(model_output_dir,
        exist_ok=True,)

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    print(f"Saved model to {model_output_dir}")

if __name__ == "__main__":
    main()
