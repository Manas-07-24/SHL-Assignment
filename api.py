from fastapi import FastAPI, Query, HTTPException
import pandas as pd
from app import load_data, process_assessment_data, process_query
import traceback
import uvicorn
from typing import List, Dict, Any, Optional

# Create FastAPI app
app = FastAPI(
    title="Assessment Recommendation API",
    description="API for recommending assessments based on text queries",
    version="1.0.0"
)

# Global variables for data storage
data = None
vectorizer = None
tfidf_matrix = None

@app.on_event("startup")
async def startup_event():
    """Load and process data when the API starts"""
    global data, vectorizer, tfidf_matrix
    try:
        data = load_data()
        data, vectorizer, tfidf_matrix = process_assessment_data(data)
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to load application data")

@app.get("/api/recommend", response_model=Dict[str, Any])
async def recommend(
    query: str = Query(..., description="The query text to match against assessments"),
    max_results: Optional[int] = Query(10, description="Maximum number of results to return")
):
    """
    Return assessment recommendations based on the query text.
    
    Args:
        query: Text to match against assessments
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary containing status, query, and assessment results
    """
    if not query:
        raise HTTPException(status_code=400, detail="Missing query parameter")
    
    try:
        # Process query
        results = process_query(query, vectorizer, tfidf_matrix, data, max_results)
        
        # Format results
        formatted_results = []
        for _, row in results.iterrows():
            formatted_results.append({
                'assessment_name': row['Assessment Name'],
                'remote_testing': row['Remote Testing'],
                'adaptive_irt': row['Adaptive/IRT'],
                'duration': row['Duration'],
                'test_type': row['Test Type'],
                'url': row['URL'],
                'relevance_score': float(row['Relevance Score'])
            })
        
        return {
            'status': 'success',
            'query': query,
            'results': formatted_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Only execute if running directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)