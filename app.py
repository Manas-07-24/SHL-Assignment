import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import BytesIO

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load the assessment data
@st.cache_data
def load_data():
    # In a real application, you would load from your CSV file
    # For the demo, we'll use the data provided in the document
    data = pd.read_csv(r"SHL_dataset.csv", encoding="latin1")

    print(data)
    return data

# Process the assessment data to create feature vectors
@st.cache_data
def process_assessment_data(df):
    # Extract test types and create a more detailed description
    df['description'] = df.apply(
        lambda row: f"{row['Assessment Name']} is a {row['Duration']} minute assessment that tests {expand_test_types(row['Test Type'])}. "
                    f"It {'supports' if row['Remote Testing'] == 'Yes' else 'does not support'} remote testing and "
                    f"{'uses' if row['Adaptive/IRT'] == 'Yes' else 'does not use'} adaptive testing technology.",
        axis=1
    )
    
    # Create a vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Create a feature matrix
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    
    return df, vectorizer, tfidf_matrix

# Function to expand test type codes
def expand_test_types(test_type_string):
    # Define mapping of codes to descriptions
    test_type_map = {
        'A': 'Ability',
        'B': 'Behavior',
        'C': 'Communication',
        'K': 'Knowledge',
        'P': 'Personality',
        'S': 'Skills'
    }
    
    # Extract individual codes
    codes = re.findall(r'[A-Z]', test_type_string)
    
    # Map codes to descriptions
    expanded = [test_type_map.get(code, code) for code in codes]
    
    # Join descriptions with commas and 'and' for the last one
    if len(expanded) == 1:
        return expanded[0]
    elif len(expanded) == 2:
        return f"{expanded[0]} and {expanded[1]}"
    else:
        return ", ".join(expanded[:-1]) + f", and {expanded[-1]}"

# Process query and find similar assessments
def process_query(query, vectorizer, tfidf_matrix, df, max_results=10):
    # Try to extract a URL from the query
    url_match = re.search(r'https?://\S+', query)
    
    if url_match:
        # If URL is found, try to extract text from it
        url = url_match.group(0)
        try:
            query = extract_text_from_url(url)
            st.info(f"Extracted text from URL: {url[:50]}...")
        except Exception as e:
            st.warning(f"Could not extract text from URL: {str(e)}")
    
    # Get the query vector
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top indices
    top_indices = similarity_scores.argsort()[-max_results:][::-1]
    
    # Get top assessments
    top_assessments = df.iloc[top_indices].copy()
    
    # Add similarity scores
    top_assessments['Relevance Score'] = similarity_scores[top_indices]
    
    return top_assessments

# Extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
            
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")

# Function to make URLs clickable
def make_clickable(url):
    return f'<a href="{url}" target="_blank">Link</a>'

# Page configuration
st.set_page_config(
    page_title="SHL Assessment Recommendation System",
    page_icon="🧪",
    layout="wide"
)

# App title and description
st.title("SHL Assessment Recommendation System")
st.markdown("""
This application helps hiring managers find the right SHL assessments for their job openings.
Enter a job description, requirements, or skills needed, and the system will recommend relevant assessments.
""")

# Load and process data
data = load_data()
data, vectorizer, tfidf_matrix = process_assessment_data(data)

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Text Input", "URL Input", "File Upload"])

with tab1:
    # Text input
    query = st.text_area(
        "Enter job description or requirements:",
        height=150,
        placeholder="Example: I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        max_results = st.number_input("Max results:", min_value=1, max_value=10, value=5)
    
    submit_button = st.button("Find Assessments", key="text_submit")
    
    if submit_button and query:
        with st.spinner("Finding relevant assessments..."):
            results = process_query(query, vectorizer, tfidf_matrix, data, max_results)
            
            if not results.empty:
                # Display results
                st.subheader("Recommended Assessments")
                
                # Convert to display format
                display_df = results[['Assessment Name', 'Remote Testing', 'Adaptive/IRT', 'Duration', 'Test Type', 'URL', 'Relevance Score']].copy()
                display_df['Relevance Score'] = display_df['Relevance Score'].apply(lambda x: f"{x:.2f}")
                
                # Make URLs clickable
                display_df['URL'] = display_df['URL'].apply(make_clickable)
                
                # Display table
                st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("No matching assessments found. Try refining your query.")

with tab2:
    # URL input
    url_input = st.text_input(
        "Enter job posting URL:",
        placeholder="https://example.com/job-posting"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        url_max_results = st.number_input("Max results:", min_value=1, max_value=10, value=5, key="url_max")
    
    url_submit = st.button("Find Assessments", key="url_submit")
    
    if url_submit and url_input:
        with st.spinner("Extracting text from URL and finding relevant assessments..."):
            try:
                extracted_text = extract_text_from_url(url_input)
                st.success("Successfully extracted text from URL")
                
                # Show extracted text
                with st.expander("View Extracted Text"):
                    st.text_area("Extracted content:", extracted_text, height=200)
                
                # Process the extracted text
                results = process_query(extracted_text, vectorizer, tfidf_matrix, data, url_max_results)
                
                if not results.empty:
                    # Display results
                    st.subheader("Recommended Assessments")
                    
                    # Convert to display format
                    display_df = results[['Assessment Name', 'Remote Testing', 'Adaptive/IRT', 'Duration', 'Test Type', 'URL', 'Relevance Score']].copy()
                    display_df['Relevance Score'] = display_df['Relevance Score'].apply(lambda x: f"{x:.2f}")
                    
                    # Make URLs clickable
                    display_df['URL'] = display_df['URL'].apply(make_clickable)
                    
                    # Display table
                    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.info("No matching assessments found. Try a different URL.")
            except Exception as e:
                st.error(f"Error processing URL: {str(e)}")

with tab3:
    # File upload
    uploaded_file = st.file_uploader("Upload a job description file", type=["txt", "pdf", "docx"])
    
    col1, col2 = st.columns([1, 5])
    with col1:
        file_max_results = st.number_input("Max results:", min_value=1, max_value=10, value=5, key="file_max")
    
    file_submit = st.button("Find Assessments", key="file_submit")
    
    if uploaded_file is not None:
        # For simplicity, we'll just read text files
        # In a real application, you'd want to handle PDF and DOCX files
        try:
            file_text = uploaded_file.getvalue().decode("utf-8")
            
            with st.expander("View File Content"):
                st.text_area("File content:", file_text, height=200)
            
            if file_submit:
                with st.spinner("Finding relevant assessments..."):
                    results = process_query(file_text, vectorizer, tfidf_matrix, data, file_max_results)
                    
                    if not results.empty:
                        # Display results
                        st.subheader("Recommended Assessments")
                        
                        # Convert to display format
                        display_df = results[['Assessment Name', 'Remote Testing', 'Adaptive/IRT', 'Duration', 'Test Type', 'URL', 'Relevance Score']].copy()
                        display_df['Relevance Score'] = display_df['Relevance Score'].apply(lambda x: f"{x:.2f}")
                        
                        # Make URLs clickable
                        display_df['URL'] = display_df['URL'].apply(make_clickable)
                        
                        # Display table
                        st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    else:
                        st.info("No matching assessments found. Try uploading a different file.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Sample queries section
st.sidebar.title("Sample Queries")
st.sidebar.markdown("""
Try these sample queries to test the system:

1. I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.

2. Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.

3. I need to assess candidates for a customer service role with a focus on communication skills. The assessment should take less than 30 minutes.

4. We're hiring for sales managers who can lead teams effectively and meet targets. Need assessments under 50 minutes.
""")

# Add explanations and API instructions
st.sidebar.title("About")
st.sidebar.markdown("""
This application uses TF-IDF vectorization and cosine similarity to match job descriptions with appropriate SHL assessments.

The system analyzes:
- Test types (A: Ability, B: Behavior, C: Communication, K: Knowledge, P: Personality, S: Skills)
- Duration constraints
- Remote testing requirements
- Adaptive testing capabilities

For API access, send a GET request to `/api/recommend` with a `query` parameter.
""")

# Add a footer
st.markdown("""
---
Built for SHL AI Intern RE Generative AI assignment
""")