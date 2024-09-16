import requests
import os
from csas.models import DynamicIssueQuestion
import numpy as np
import json
from django.apps import apps
import logging

logger = logging.getLogger(__name__)

def generate_embedding(text):
    headers = {
        'Authorization': f'Bearer ' + os.environ['OPENAI_API_KEY'],
        'Content-Type': 'application/json'
    }
    data = {'input': text, 'model': 'text-embedding-ada-002'}
    response = requests.post('https://api.openai.com/v1/embeddings',
                             headers=headers,
                             json=data)
    embedding = response.json().get('data')[0].get('embedding')
    logger.debug(f"Generated embedding (OpenAI)")
    return embedding

def generate_embedding_alt(text):
    model = apps.get_app_config('csas').model

    # Generate embedding
    embedding = model.encode(text)

    # Ensure the embedding is a list, not a numpy array
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    
    # If it's a single embedding, wrap it in a list
    if not isinstance(embedding, list):
        embedding = [embedding]

    logger.debug(f"Generated embedding (Alt)")
    return embedding

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2.T)
    norm_vec1 = np.linalg.norm(vec1, axis=1, keepdims=True)
    norm_vec2 = np.linalg.norm(vec2, axis=1, keepdims=True)
    similarity = dot_product / np.dot(norm_vec1, norm_vec2.T)
    logger.debug(f"Cosine similarity: {similarity}")
    return similarity

def cosine_similarity_knn(embedding, all_embeddings, top_n=3):
    if not all_embeddings:
        return None
    embedding_array_2d = np.array(embedding).reshape(1, -1)
    all_embeddings_array = np.array(all_embeddings)
    
    # Verify dimensions
    if embedding_array_2d.shape[1] != all_embeddings_array.shape[1]:
        logger.error(f"Embedding dimensions do not match: {embedding_array_2d.shape[1]} != {all_embeddings_array.shape[1]}")
        raise ValueError(f"Embedding dimensions do not match: {embedding_array_2d.shape[1]} != {all_embeddings_array.shape[1]}")
    
    similarities = cosine_similarity(embedding_array_2d, all_embeddings_array).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    nearest_neighbors = [(index, similarities[index]) for index in top_indices]
    logger.debug(f"Nearest neighbors: {nearest_neighbors}")
    return nearest_neighbors

def find_similar_issues(text, embedding_func=generate_embedding):
    embedding = embedding_func(text)
    logger.debug(f"Embedding for text '{text}'")
    all_issue_questions = DynamicIssueQuestion.objects.all().values_list(
        'embedding', flat=True)
    if not all_issue_questions:
        return None
    # Ensure embeddings are in the correct format (list of lists)
    all_embeddings = [
        e if isinstance(e, list) else [] for e in all_issue_questions
    ]
    # Check dimensions
    if all_embeddings and len(all_embeddings[0]) != len(embedding):
        logger.error(f"Embedding dimensions do not match: {len(all_embeddings[0])} != {len(embedding)}")
        raise ValueError(f"Embedding dimensions do not match: {len(all_embeddings[0])} != {len(embedding)}")
    top_n_similar = cosine_similarity_knn(embedding, all_embeddings, top_n=3)
    similar_issues = []
    for idx, _ in top_n_similar:
        try:
            question = DynamicIssueQuestion.objects.get(pk=idx).question
            similar_issues.append(question)
        except DynamicIssueQuestion.DoesNotExist:
            continue  # Skip if the corresponding question does not exist
    logger.debug(f"Similar issues: {similar_issues}")
    return similar_issues

def has_high_similarity(text, embedding_func=generate_embedding):
    embedding = embedding_func(text)
    logger.debug(f"Embedding for text '{text}'")
    all_issue_questions = DynamicIssueQuestion.objects.all().values_list(
        'embedding', flat=True)
    if not all_issue_questions:
        logger.debug("No issue questions found in the database.")
        return False
    all_embeddings = [
        e if isinstance(e, list) else [] for e in all_issue_questions
    ]
    if not all_embeddings:
        logger.debug("No embeddings found in the database.")
        return False
    top_n_similar = cosine_similarity_knn(embedding, all_embeddings, top_n=3)
    for _, similarity in top_n_similar:
        if similarity > 0.95:
            logger.debug(f"High similarity found: {similarity}")
            return True
    logger.debug("No high similarity found")
    return False

def is_text_toxic(text):
    headers = {
        'Authorization': f'Bearer ' + os.environ['OPENAI_API_KEY'],
        'Content-Type': 'application/json'
    }
    data = {'input': text}
    response = requests.post('https://api.openai.com/v1/moderations',
                             headers=headers,
                             json=data)
    moderation_result = response.json()
    return moderation_result.get('results')[0].get('flagged', False)

def is_text_toxic_alt(text):
    prompt = f"""You are an AI assistant tasked with detecting toxic content. Analyze the following text and determine if it contains any toxic, harmful, or inappropriate content. Respond with a JSON object containing a 'flagged' field set to true if the text is toxic, or false if it is not toxic.

Text to analyze: "{text}"

Response:"""

    data = {
        'model': 'meta-llama/llama-3.1-405b-instruct',
        'messages': [
            {'role': 'system', 'content': 'You are an AI assistant that analyzes text for toxicity.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0,
        'max_tokens': 100
    }

    headers = {
        'Authorization': f'Bearer {os.environ["OPENROUTER_API_KEY"]}',
        'Content-Type': 'application/json'
    }

    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers=headers,
        json=data
    )

    try:
        result = json.loads(response.json()['choices'][0]['message']['content'])
        return result.get('flagged', False)
    except (json.JSONDecodeError, KeyError, IndexError):
        # If there's an error parsing the response, err on the side of caution
        return True

def is_text_toxic_alt(text):
    prompt = f"""You are an AI assistant tasked with detecting toxic content. Analyze the following text and determine if it contains any toxic, harmful, or inappropriate content. Respond with a JSON object containing a 'flagged' field set to true if the text is toxic, or false if it is not toxic.

Text to analyze: "{text}"

Response:"""

    data = {
        'model': 'meta-llama/llama-3.1-405b-instruct',
        'messages': [
            {'role': 'system', 'content': 'You are an AI assistant that analyzes text for toxicity.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0,
        'max_tokens': 100
    }

    headers = {
        'Authorization': f'Bearer {os.environ["OPENROUTER_API_KEY"]}',
        'Content-Type': 'application/json'
    }

    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers=headers,
        json=data
    )

    try:
        result = json.loads(response.json()['choices'][0]['message']['content'])
        return result.get('flagged', False)
    except (json.JSONDecodeError, KeyError, IndexError):
        # If there's an error parsing the response, err on the side of caution
        return False
