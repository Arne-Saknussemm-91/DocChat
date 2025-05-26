def clean_text(text):
    import re
    # Remove headers, footers, and special characters
    text = re.sub(r'\n+', '\n', text)  # Remove extra newlines
    text = re.sub(r'\s+', ' ', text)   # Remove extra whitespace
    text = re.sub(r'[^\w\s,.!?]', '', text)  # Remove special characters
    return text.strip()

def validate_retrieval(retrieved_chunks, expected_keywords):
    # Check if retrieved chunks contain expected keywords
    return any(keyword in chunk for chunk in retrieved_chunks for keyword in expected_keywords)

def log_output(output, log_file='output.log'):
    # Log output to a specified log file
    with open(log_file, 'a') as f:
        f.write(output + '\n')