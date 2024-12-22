# ESG NER Pipeline

In this project, we implement a Named Entity Recognition (NER) pipeline using SpaCy to extract Environmental, Social, and Governance (ESG) indicators from documents. The pipeline includes data preparation, model training, and entity extraction from PDFs.

---

## Features

- **Custom NER Model**: Trains a SpaCy model to recognize ESG-specific entities.
- **Data Preparation**: Converts training data into SpaCy-compatible format.
- **Entity Extraction**: Extracts ESG entities from individual or multiple PDFs.
- **Export Results**: Outputs extracted entities into a structured CSV file.

---

## Requirements

Ensure you have the following installed:

- Python 3.8+
- SpaCy
- pdfplumber
- json
- csv

Install dependencies using pip:
```bash
pip install spacy pdfplumber
```

Download the French SpaCy language model:
```bash
python -m spacy download fr_core_news_lg
```

---

## Directory Structure

```
project-directory/
|
|-- esg_ner_pipeline.py       # Main Python script
|-- spacy_formatted_esg_dataset.json  # Training dataset
|-- output/                   # Trained model output
|-- ESG/                      # Directory containing PDF files
|-- resultats_ner_esg.csv     # Extracted entities (output file)
```

---

## Usage

### 1. Prepare the Training Dataset

Ensure your training data is in SpaCy-compatible JSON format. Use the `spacy_formatted_esg_dataset.json` as an example.

### 2. Train the Model

Run the script to train the NER model:
```bash
python esg_ner_pipeline.py
```

This script:
- Loads the training data from `spacy_formatted_esg_dataset.json`.
- Trains the NER model.
- Saves the trained model to the `output_esg_ner_model` directory.

### 3. Extract Entities from PDFs

Place your PDF files in the `ESG/` directory. The script will extract entities and save results in `resultats_ner_esg.csv`.

---

## Outputs

### CSV File Format

The results are exported as a CSV file with the following columns:

- `entité`: The extracted entity text.
- `label`: The entity label/category.
- `start`: Start index of the entity in the text.
- `end`: End index of the entity in the text.
- `fichier`: The source PDF file.

Example:
```csv
entité,label,start,end,fichier
12,000 tonnes,ÉMISSIONS_CO2_SCOPE_1,32,42,example.pdf
800,000 m³,CONSOMMATION_D'EAU,11,21,example2.pdf
```

---

## Key Functions

- **`load_training_data(json_path)`**:
  - Loads and preprocesses training data from a JSON file.

- **`prepare_training_data(training_data)`**:
  - Converts training data to SpaCy `DocBin` format.

- **`train_model(train_db, output_dir, n_iter, drop)`**:
  - Trains the NER model.

- **`extract_from_pdf(pdf_path, model_dir)`**:
  - Extracts entities from a single PDF.

- **`extract_from_multiple_pdfs(pdf_directory, model_dir, output_csv)`**:
  - Extracts entities from multiple PDFs and saves results in a CSV file.

---

## Notes

- Ensure your training dataset is balanced and diverse to avoid overfitting.
- Use clear, unit-specific values in the dataset (e.g., `12,000 tonnes`, `800,000 m³`) for better detection.
- Validate the model's accuracy on unseen test data.

---

## Future Improvements

- Implement additional post-processing steps to filter false positives.
- Extend the dataset with more ESG-specific entities and contexts.

