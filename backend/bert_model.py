from transformers import pipeline

# Use a public model fine-tuned for fake news detection
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

# Test prediction
text = "The COVID-19 vaccine causes infertility."
result = classifier(text)

print(result)
