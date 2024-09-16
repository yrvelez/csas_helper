import json
import requests
import numpy as np
import os
from django.http import JsonResponse, HttpResponseRedirect
from django.views.decorators.http import require_http_methods
from .models import DynamicIssueQuestion, UserDatabase, GlobalSetting
from django.views.decorators.csrf import csrf_exempt
from django.core import serializers
from django.urls import reverse
from django.shortcuts import render, redirect, reverse
from django.contrib import admin
from .models import GlobalSetting
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .embeds import generate_embedding, cosine_similarity_knn, find_similar_issues, has_high_similarity, is_text_toxic, is_text_toxic_alt, generate_embedding_alt
import logging

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["POST"])
def set_ai_choice(request):
    try:
        data = json.loads(request.body)
        ai_choice = data.get('ai_choice')
        
        if ai_choice not in ['openai', 'llama']:
            return JsonResponse({'status': 'error', 'message': 'Invalid AI choice.'}, status=400)
        
        request.session['ai_choice'] = ai_choice
        request.session.save()  # Ensure the session is saved
        
        return JsonResponse({'status': 'success', 'message': 'AI choice set successfully.'})
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON format.'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
def get_global_setting(key):
    try:
        return GlobalSetting.objects.get(key=key).value
    except GlobalSetting.DoesNotExist:
        return None

def view_db_content(request):
    questions = DynamicIssueQuestion.objects.all()
    return render(request, 'view_db_content.html', {'questions': questions})


def view_user_content(request):
    user_data = UserDatabase.objects.all()
    return render(request, 'view_user_content.html', {'user_data': user_data})


def thank_you(request):
    return render(request, 'thank_you.html')

@csrf_exempt
@require_http_methods(["POST"])
def save_session_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Iterate over each key-value pair in the received data
            for key, value in data.items():
                # Update or create the global setting
                GlobalSetting.objects.update_or_create(
                    key=key, defaults={'value': value})

            return JsonResponse({
                'status': 'success',
                'message': 'Global settings updated successfully',
                'received_data': data
            })
        except json.JSONDecodeError:
            return JsonResponse(
                {
                    'status': 'error',
                    'message': 'Invalid JSON format'
                },
                status=400)
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            },
                                status=500)

    else:
        return JsonResponse(
            {
                'status': 'error',
                'message': 'Invalid request method'
            },
            status=400)
    


# Function to get the embedding dimension
def get_embedding_dimension(embedding_func):
    # This function should return the dimension of the embeddings produced by the given function
    if embedding_func == generate_embedding:
        return 1536  # Example dimension for generate_embedding
    elif embedding_func == generate_embedding_alt:
        return 384  # Example dimension for generate_embedding_alt
    else:
        raise ValueError("Unknown embedding function")

@csrf_exempt
@require_http_methods(["POST"])
def upload_completion(request):
    
    # load global settings and extract ai_choice
    settings = GlobalSetting.objects.all()
    global_settings = {setting.key: setting.value for setting in settings}
    ai_choice = global_settings['ai_choice'] 

    logger.info("AI choice: " + ai_choice)
    
    if ai_choice == 'llama':
        generate_embedding_func = generate_embedding_alt    
    else:
        generate_embedding_func = generate_embedding

    completion = request.POST.get('completion')
    issue = request.POST.get('issue')

    if completion:
        # Save the completion to the database
        if ai_choice == 'llama':
            is_text_toxic_func = is_text_toxic_alt
            generate_embedding_func = generate_embedding_alt 
        else:
            is_text_toxic_func = is_text_toxic
            generate_embedding_func = generate_embedding

        # Check for an existing question in the database
        existing_question = DynamicIssueQuestion.objects.filter(
            question=completion).first()

        # Get midpoint of the scale from session
        min_scale = float(get_global_setting('min_scale') or 1)
        max_scale = float(get_global_setting('max_scale') or 5)
        midpoint = (min_scale + max_scale) / 2

        if existing_question or has_high_similarity(
                completion, generate_embedding_func) or is_text_toxic_func(completion):
            new_rating = request.POST.get('rating', midpoint)
            new_rating = float(new_rating)

            # Update the existing question rating
            existing_question.ratings += 1
            existing_question.avg_rating = (
                (existing_question.avg_rating *
                 (existing_question.ratings - 1)) +
                new_rating) / existing_question.ratings
            existing_question.var_rating = (1 / existing_question.ratings)
            existing_question.save()
        else:
            if not has_high_similarity(completion, generate_embedding_func) and not is_text_toxic_func(completion):
                embed_completion = generate_embedding_func(completion)

                # Ensure embedding dimensions match
                expected_dim = get_embedding_dimension(generate_embedding_func)

                if len(embed_completion) != expected_dim:
                    raise ValueError(f"Embedding dimensions do not match: {len(embed_completion)} != {expected_dim}")

                new_question = DynamicIssueQuestion(question=completion,
                                                    avg_rating=midpoint,
                                                    ratings=1,
                                                    var_rating=1,
                                                    embedding=embed_completion)
                new_question.save()

        return redirect('view_db_content')

    return redirect('fetch_llama_completion', issue=issue)

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_entry(request, id):
    try:
        question = DynamicIssueQuestion.objects.get(pk=id)
        question.delete()
        return JsonResponse({'status': 'success', 'message': 'Entry deleted successfully'})
    except DynamicIssueQuestion.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Entry not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["PUT"])
def edit_entry(request, id):
    try:
        data = json.loads(request.body)
        question_text = data.get('question')
        question = DynamicIssueQuestion.objects.get(pk=id)
        question.question = question_text
        question.save()
        return JsonResponse({'status': 'success', 'message': 'Entry updated successfully'})
    except DynamicIssueQuestion.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Entry not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def upload_completions(request):
    logger.debug("Starting upload_completions")

    ai_choice = get_global_setting('ai_choice')
    logger.debug(f"Global setting 'ai_choice': {ai_choice}")

    if ai_choice is None:
        logger.info("AI choice is not set in the global settings.")
        return JsonResponse({'status': 'error', 'message': 'AI choice is not set in the global settings.'}, status=400)

    completions = request.POST.getlist('approved_completions')
    logger.debug(f"Received completions: {completions}")

    if ai_choice == 'llama':
        generate_embedding_func = generate_embedding_alt
        similarity_check = has_high_similarity
        toxicity_check = is_text_toxic_alt
    else:
        generate_embedding_func = generate_embedding
        similarity_check = has_high_similarity
        toxicity_check = is_text_toxic

    # Get midpoint
    min_scale = float(get_global_setting('min_scale') or 1)
    max_scale = float(get_global_setting('max_scale') or 5)
    midpoint = (min_scale + max_scale) / 2
    logger.debug(f"Midpoint: {midpoint}")

    new_rating = float(request.POST.get('rating', midpoint))  # Get the rating outside the loop
    logger.debug(f"New rating: {new_rating}")

    if completions:
        for completion in completions:
            try:
                logger.debug(f"Processing completion: {completion}")

                existing_question = DynamicIssueQuestion.objects.filter(
                    question=completion).first()
                logger.debug(f"Existing question: {existing_question}")

                if existing_question:
                    logger.debug(f"Existing question found: {existing_question.question}")
                    continue

                is_similar = similarity_check(completion, generate_embedding_func)
                is_toxic = toxicity_check(completion)
                logger.debug(f"Is similar: {is_similar}, Is toxic: {is_toxic}")

                if not is_similar and not is_toxic:
                    embed_completion = generate_embedding_func(completion)
                    logger.debug(f"Generated embedding for completion: {embed_completion}")

                    # Create a new question
                    new_question = DynamicIssueQuestion(
                        question=completion,
                        avg_rating=new_rating,
                        ratings=1,
                        var_rating=1,
                        embedding=embed_completion)
                    new_question.save()
                    logger.info(f"New question saved: {new_question.question}")
                else:
                    logger.debug(f"Completion '{completion}' is similar or toxic. Skipping.")
            except Exception as e:
                logger.error(f"Error processing completion '{completion}': {str(e)}")

        return redirect('view_db_content')

    logger.debug("No completions received")
    return redirect('view_db_content')

def update_existing_question_rating(question, new_rating):
    question.ratings += 1
    question.avg_rating = (
        (question.avg_rating *
         (question.ratings - 1)) + new_rating) / question.ratings
    question.var_rating = (1 / question.ratings)
    question.save()

@csrf_exempt
@require_http_methods(["GET", "POST"])
def main_page(request):
    context = {}

    # Fetch global settings
    settings = GlobalSetting.objects.all()
    global_settings = {setting.key: setting.value for setting in settings}

    # Use global settings or default values
    session_data = {
        'prompt': global_settings.get('prompt', 'Default Prompt'),
        'ai_choice': global_settings.get('ai_choice', 'openai'),
        'num_items': global_settings.get('num_items', '3'),
        'min_scale': global_settings.get('min_scale', '1'),
        'max_scale': global_settings.get('max_scale', '5'),
        'survey_text': global_settings.get('survey_text', 'Rate on a 1-5 scale.')
    }

    if request.method == 'POST':
        issue = request.POST.get('issue')
        prompt = request.POST.get('prompt')
        ai_choice = global_settings['ai_choice'] 

        if ai_choice == 'llama':
            # Preload the model
            from django.apps import apps
            model = apps.get_app_config('csas').model
            if not model:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                apps.get_app_config('csas').model = model
                logger.info("Open-source model preloaded successfully.")

        if not prompt:
            prompt = f"You are a classification expert who takes an input and returns a political issue as a summary. Please extract the political issue or concern mentioned by the respondent using one to three words. Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. Never ask about two issues.\nIf a related issue or theme has already been mentioned, return the same issue or theme as the output. You must only return one issue. Do not duplicate broad issue areas or themes.\n Examples:\nPreviously Mentioned Issues () I care about the environment.->Environment######Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation######Previously Mentioned Issues () Abortion should be legal under all circumstances.->Abortion######Previously Mentioned Issues (Immigration) Close the borders.->Immigration######Previously Mentioned Issues (Inflation) I am concerned about rising prices.->Inflation######"

        if ai_choice == 'openai':
            return redirect('fetch_openai_completion', issue=issue, prompt=prompt)
        elif ai_choice == 'llama':
            return redirect('fetch_llama_completion', issue=issue, prompt=prompt)

    elif request.method == 'GET':
        # Update the URLs to include the prompt
        dynamic_issue_url_oai = request.build_absolute_uri('/dynamic-issue-oai/').replace('http:', 'https:')
        dynamic_issue_url_llama = request.build_absolute_uri('/dynamic-issue-llama/').replace('http:', 'https:')
        select_questions_simulation_url = request.build_absolute_uri(reverse('select_questions_simulation')).replace('http:', 'https:')
        update_rating_url = request.build_absolute_uri(reverse('update_rating')).replace('http:', 'https:')
        ez_url = request.build_absolute_uri(reverse('survey')).replace('http:', 'https:').replace('survey/', 'survey') + '?id=${e://Field/ResponseID}'

        # Add embed iframe for ez_url
        ez_url = f'<iframe src="{ez_url}" width="500" height="500" frameborder="0" style="border:0" allowfullscreen></iframe>'

        context = {
            'dynamic_issue_url_oai': dynamic_issue_url_oai,
            'dynamic_issue_url_llama': dynamic_issue_url_llama,
            'select_questions_simulation_url': select_questions_simulation_url,
            'update_rating_url': update_rating_url,
            'ez_url': ez_url,
            'ai_choice': global_settings.get('ai_choice', 'openai')  # Pass the saved ai_choice to the template
        }

    return render(request, 'main.html', context)

@csrf_exempt
@require_http_methods(["GET"])
def fetch_llama_completion(request, issue, prompt=None):
    logger = logging.getLogger(__name__)
    logger.info("Received request for fetch_llama_completion.")

    issues_list = issue.split(';')  # Split the issue string into a list
    completions = []  # List to hold completions for each issue

    for single_issue in issues_list:
        try:
            previous_list = ', '.join(find_similar_issues(single_issue,
                                                              embedding_func=generate_embedding_alt))
            logger.debug(f"Similar issues: {previous_list}")
        except Exception as e:
            previous_list = ''
            logger.error(f"Error finding similar issues: {e}")

        if prompt is None:
            prompt = """You are a classification expert who takes an input and returns a political issue as a summary. Please extract the political issue or concern mentioned by the respondent using one to three words. Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. Never ask about two issues.
If a related issue or theme has already been mentioned, return the same issue or theme as the output. You must only return one issue. Do not duplicate broad issue areas or themes. If an issue is not political or irrelevant, return 'Room Temperature Semiconductors'.

Examples:
Previously Mentioned Issues () I care about the environment.->Environment
Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation
Previously Mentioned Issues () Abortion should be legal under all circumstances.->Abortion
Previously Mentioned Issues (Immigration) Close the borders.->Immigration
Previously Mentioned Issues (Inflation) I am concerned about rising prices.->Inflation"""

        prompt_template = f"{prompt}\n\nPreviously Mentioned Issues ({previous_list}) {single_issue}->"

        data = {
            'model': 'meta-llama/llama-3.1-70b-instruct',
            'messages': [{
                'role': 'user',
                'content': prompt_template
            }],
            'temperature': 0,
            'max_tokens': 10
        }

        logger.debug(f"Sending request to OpenRouter API with data: {data}")

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                },
                json=data
            )
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Received response from OpenRouter API: {response_data}")

            completion = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            logger.debug(f"Raw completion: {completion}")

            # Clean the completion text
            completion = completion.split('#')[0].split('\n')[0].split('Previously')[0].split('/')[0].split(';')[0].strip()
            logger.debug(f"Cleaned completion: {completion}")

            if not completion:
                completion = "Room Temperature Semiconductors"

            completions.append(completion)

        except requests.RequestException as e:
            logger.error(f"Error making request to OpenRouter API: {e}")
            completions.append("Error processing request")
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            completions.append("An unexpected error occurred")

    return render(request, 'show_completion.html', {'completions': completions, 'issue': issue})


@csrf_exempt
@require_http_methods(["GET"])
def fetch_openai_completion(
    request,
    issue,
    prompt="Please extract the political issue or concern mentioned by the respondent using one to three words. Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. Never ask about two issues.\nIf a related issue or theme has already been mentioned, return the same issue or theme as the output. You must only return one issue. Do not duplicate broad issue areas or themes.\n Examples:\nPreviously Mentioned Issues () I care about the environment.->Environment######Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation######Previously Mentioned Issues () Abortion should be legal under all circumstances.->Abortion######Previously Mentioned Issues (Immigration) Close the borders.->Immigration######Previously Mentioned Issues (Inflation) I am concerned about rising prices.->Inflation######Previously Mentioned Issues () "
):
    issues_list = issue.split(';')  # Split the issue string into a list
    completions = []  # List to hold completions for each issue

    for single_issue in issues_list:
        try:
            previous_list = ', '.join(find_similar_issues(single_issue))
        except:
            previous_list = ''

        data = {
            'model':
            'gpt-4o-mini',
            'messages': [{
                'role':
                'system',
                'content':
                f"Previously Mentioned Issues ({previous_list}) {prompt}{single_issue}->"
            }],
            'temperature':
            0,
            'max_tokens':
            15
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}'
        }

        response = requests.post('https://api.openai.com/v1/chat/completions',
                                 headers=headers,
                                 data=json.dumps(data))

        completion = response.json().get('choices',
                                         [{}])[0].get('message',
                                                      {}).get('content', '')
        completions.append(completion)  # Append the completion to the list

    return render(
        request,
        'show_completion.html',
        {
            'completions': completions,  # Return the list of completions
            'issue': issue
        })

@csrf_exempt
@require_http_methods(["POST"])
def dynamic_issue_openai(request):
    logger = logging.getLogger(__name__)
    logger.info("Received request for dynamic_issue_openai.")

    try:
        data = json.loads(request.body)
        issue = data.get('input', '').strip()
        logger.debug(f"Input issue: {issue}")

        if not issue:
            logger.warning("Empty issue received.")
            return JsonResponse({'completion': 'Room Temperature Semiconductors'}, status=400)

        prompt_template = (
            "You are a classification expert who takes an input and returns a political issue as a summary. "
            "Please extract the political issue or concern mentioned by the respondent using one to three words. "
            "Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. "
            "Never ask about two issues. If a related issue or theme has already been mentioned, return the same issue or theme as the output. "
            "You must only return one issue. Do not duplicate broad issue areas or themes. If an issue is not political or irrelevant, return 'Room Temperature Semiconductors.'\n\n"
            "Examples:\n"
            "Previously Mentioned Issues () I care about the environment.->Environment\n"
            "Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation\n"
            "Previously Mentioned Issues () Abortion should be legal under all circumstances.->Abortion\n"
            "Previously Mentioned Issues (Immigration) Close the borders.->Immigration\n"
            "Previously Mentioned Issues (Inflation) I am concerned about rising prices.->Inflation"
        )

        try:
            previous_list = ', '.join(find_similar_issues(issue))
            logger.debug(f"Similar issues: {previous_list}")
        except Exception as e:
            previous_list = ''
            logger.error(f"Error finding similar issues: {e}")

        messages = [
            {
                'role': 'system',
                'content': prompt_template
            },
            {
                'role': 'user',
                'content': f"Previously Mentioned Issues ({previous_list}) {issue}->"
            }
        ]

        api_data = {
            'model': 'gpt-4o-mini',
            'messages': messages,
            'temperature': 0,
            'max_tokens': 15
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}'
        }

        logger.debug(f"Sending request to OpenAI API with data: {api_data}")

        response = requests.post('https://api.openai.com/v1/chat/completions',
                                 headers=headers,
                                 data=json.dumps(api_data))
        response.raise_for_status()  # Raise an error for bad status codes
        response_data = response.json()
        logger.debug(f"Received response from OpenAI API: {response_data}")

        completion = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        logger.debug(f"Raw completion: {completion}")

        # Clean the completion text
        completion = completion.split('#')[0].split('\n')[0].split('Previously')[0].split('/')[0].split(';')[0].strip()
        logger.debug(f"Cleaned completion: {completion}")

        embed_completion = generate_embedding(completion)

        # Get midpoint
        min_scale = float(get_global_setting('min_scale') or 1)
        max_scale = float(get_global_setting('max_scale') or 5)
        midpoint = (min_scale + max_scale) / 2
        logger.debug(f"Midpoint: {midpoint}")

        # Check if the completion is toxic or similar
        if is_text_toxic(completion) or has_high_similarity(completion):
            logger.info(f"Completion '{completion}' is toxic or similar. Selecting a random question.")
            # Return a random question from the database as the completion
            random_question = DynamicIssueQuestion.objects.order_by('?').first()
            if random_question:
                completion = random_question.question
                logger.debug(f"Randomly selected question: {completion}")
        else:
            # Set default ratings for the new question
            dynamic_issue_question = DynamicIssueQuestion(
                question=completion,
                avg_rating=midpoint,
                ratings=1,
                var_rating=1,
                embedding=embed_completion
            )
            dynamic_issue_question.save()  # Save the instance to the database
            logger.info(f"Saved new question: {completion}")

        return JsonResponse({'completion': completion})

    except requests.exceptions.RequestException as e:
        logger.error(f"Request to OpenAI API failed: {e}")
        return JsonResponse({'completion': 'Room Temperature Semiconductors'}, status=500)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return JsonResponse({'completion': 'Room Temperature Semiconductors'}, status=400)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return JsonResponse({'completion': 'Room Temperature Semiconductors'}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def dynamic_issue_llama(request):
    logger = logging.getLogger(__name__)
    logger.info("Received request for dynamic_issue_llama.")

    try:
        data = json.loads(request.body)
        issue = data.get('input', '').strip()
        logger.debug(f"Input issue: {issue}")

        if not issue:
            logger.warning("Empty issue received.")
            return JsonResponse({'completion': 'Room Temperature Semiconductors'}, status=400)

        prompt_template = (
            "You are a classification expert who takes an input and returns a political issue as a summary. "
            "Please extract the political issue or concern mentioned by the respondent using one to three words. "
            "Be descriptive and stay true to what the user has written. Select only one issue, concern, or topic. "
            "Never ask about two issues. If a related issue or theme has already been mentioned, return the same issue or theme as the output. "
            "You must only return one issue. Do not duplicate broad issue areas or themes. If an issue is not political or irrelevant, return 'Room Temperature Semiconductors.'\n\n"
            "Examples:\n"
            "Previously Mentioned Issues () I care about the environment.->Environment\n"
            "Previously Mentioned Issues (Taxation) My taxes are too high.->Taxation\n"
            "Previously Mentioned Issues () Abortion should be legal under all circumstances.->Abortion\n"
            "Previously Mentioned Issues (Immigration) Close the borders.->Immigration\n"
            "Previously Mentioned Issues (Inflation) I am concerned about rising prices.->Inflation"
        )

        try:
            previous_list = ', '.join(find_similar_issues(issue,
                                                          embedding_func=generate_embedding_alt))
            logger.debug(f"Similar issues: {previous_list}")
        except Exception as e:
            previous_list = ''
            logger.error(f"Error finding similar issues: {e}")

        messages = [
            {
                'role': 'system',
                'content': prompt_template
            },
            {
                'role': 'user',
                'content': f"Previously Mentioned Issues ({previous_list}) {issue}->"
            }
        ]

        data = {
            'model': 'meta-llama/llama-3.1-70b-instruct',
            'messages': messages,
            'temperature': 0,
            'max_tokens': 15
        }

        logger.debug(f"Sending request to OpenRouter API with data: {data}")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"
            },
            json=data
        )
        response.raise_for_status()
        response_data = response.json()
        logger.debug(f"Received response from OpenRouter API: {response_data}")

        completion = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        logger.debug(f"Raw completion: {completion}")

        # Clean the completion text
        completion = completion.split('#')[0].split('\n')[0].split('Previously')[0].split('/')[0].split(';')[0].strip()
        logger.debug(f"Cleaned completion: {completion}")

        embed_completion = generate_embedding_alt(completion)

        # Get midpoint
        min_scale = float(get_global_setting('min_scale') or 1)
        max_scale = float(get_global_setting('max_scale') or 5)
        midpoint = (min_scale + max_scale) / 2
        logger.debug(f"Midpoint: {midpoint}")

        # Check if the completion is toxic or similar
        if is_text_toxic_alt(completion) or has_high_similarity(completion,
                                                                embedding_func=generate_embedding_alt):
            logger.info(f"Completion '{completion}' is toxic or similar. Selecting a random question.")
            # Return a random question from the database as the completion
            random_question = DynamicIssueQuestion.objects.order_by('?').first()
            if random_question:
                completion = random_question.question
                logger.debug(f"Randomly selected question: {completion}")
        else:
            # Set default ratings for the new question
            dynamic_issue_question = DynamicIssueQuestion(
                question=completion,
                avg_rating=midpoint,
                ratings=1,
                var_rating=1,
                embedding=embed_completion
            )
            dynamic_issue_question.save()  # Save the instance to the database
            logger.info(f"Saved new question: {completion}")

        return JsonResponse({'completion': completion})

    except requests.exceptions.RequestException as e:
        logger.error(f"Request to OpenRouter API failed: {e}")
        return JsonResponse({'completion': 'Room Temperature Semiconductors'}, status=500)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return JsonResponse({'completion': 'Room Temperature Semiconductors'}, status=400)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return JsonResponse({'completion': 'Room Temperature Semiconductors'}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def update_rating(request):
    try:
        # Check if there are any query parameters
        if not request.GET:
            return JsonResponse(
                {
                    'status': 'error',
                    'message': 'No data provided.'
                },
                status=400)

        for question, new_rating_str in request.GET.items():
            new_rating = float(new_rating_str) if new_rating_str else 0
            if new_rating <= 0:
                continue  # Skip invalid ratings

            # Update rating for each question
            try:
                issue = DynamicIssueQuestion.objects.get(question=question)
                issue.ratings += 1
                issue.avg_rating = (
                    (issue.avg_rating *
                     (issue.ratings - 1)) + new_rating) / issue.ratings
                issue.var_rating = 1 / issue.ratings
                issue.save()
            except DynamicIssueQuestion.DoesNotExist:
                continue  # Skip if question not found

        return JsonResponse({
            'status': 'success',
            'message': 'Ratings updated successfully.'
        })

    except ValueError:
        return JsonResponse(
            {
                'status': 'error',
                'message': 'Invalid rating format.'
            },
            status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def update_ratings(request):
    try:
        data = json.loads(request.body)
        ratings = data.get('ratings', [])

        if not ratings:
            return JsonResponse({'status': 'error', 'message': 'No ratings provided'}, status=400)

        # Process the ratings here
        # For example, you might update your database with the new ratings

        # Return a success response
        return JsonResponse({'status': 'success', 'message': 'Ratings updated successfully'})

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON format'}, status=400)
    except Exception as e:
        # Log the error for debugging
        print(f"Error in update_ratings: {str(e)}")
        return JsonResponse({'status': 'error', 'message': 'An unexpected error occurred'}, status=500)


@require_http_methods(["GET"])
def view_issues(request):
    questions = DynamicIssueQuestion.objects.all()
    questions_json = serializers.serialize('json', questions)
    return JsonResponse(questions_json, safe=False)

@csrf_exempt
@require_http_methods(["GET", "POST"])
def select_questions_simulation(request):
    try:
        # Fetch all questions from the database
        questions = list(DynamicIssueQuestion.objects.all().values('question', 'avg_rating', 'ratings', 'var_rating'))
        
        if not questions:
            logger.warning("No questions found in the database.")
            return JsonResponse({'error': 'No questions available.'}, status=400)
        
        # Convert database rows to means and covariances in numpy arrays
        means = []
        covariances = []
        for question in questions:
            avg_rating = question.get('avg_rating')
            var_rating = question.get('var_rating')
            
            # Validate avg_rating and var_rating
            if avg_rating is None or var_rating is None:
                logger.warning(f"Missing ratings for question: {question.get('question')}")
                continue  # Skip questions with missing ratings
            
            means.append([avg_rating])
            covariances.append(np.diag([var_rating]))
        
        if not means:
            logger.error("No valid questions with ratings found.")
            return JsonResponse({'error': 'No valid questions with ratings available.'}, status=400)
    
        # Run simulations
        num_sims = 250  # Adjust as needed
        results = []
        for _ in range(num_sims):
            samples = [
                np.random.multivariate_normal(mean, cov)
                for mean, cov in zip(means, covariances)
            ]
            max_value_index = np.argmax(samples, axis=0)[0]
            results.append(max_value_index)
    
        # Count the number of times each index had the highest value
        counts = np.bincount(results, minlength=len(questions))
        proportions = counts / num_sims
    
        # Apply probability floor and renormalize
        floor = 0.01
        proportions = np.maximum(proportions, floor)
        proportions /= proportions.sum()
    
        # Fetch global settings with defaults
        min_scale_str = get_global_setting('min_scale')
        max_scale_str = get_global_setting('max_scale')
    
        try:
            min_scale = float(min_scale_str) if min_scale_str else 1.0  # Default to 1.0
        except ValueError:
            logger.error(f"Invalid min_scale value: {min_scale_str}. Defaulting to 1.0.")
            min_scale = 1.0
    
        try:
            max_scale = float(max_scale_str) if max_scale_str else 5.0  # Default to 5.0
        except ValueError:
            logger.error(f"Invalid max_scale value: {max_scale_str}. Defaulting to 5.0.")
            max_scale = 5.0
    
        midpoint = (min_scale + max_scale) / 2.0
        logger.debug(f"Midpoint calculated: {midpoint}")
    
        # Fetch num_items with default
        num_items_str = get_global_setting('num_items')
        try:
            num_items = int(num_items_str) if num_items_str else int(midpoint)
        except ValueError:
            logger.error(f"Invalid num_items value: {num_items_str}. Defaulting to midpoint: {midpoint}")
            num_items = int(midpoint)
    
        # Ensure num_items does not exceed the number of available questions
        num_items = min(num_items, len(questions))
    
        # Randomly select K questions based on the proportions
        selected_indices = np.random.choice(len(questions),
                                            size=num_items,
                                            replace=False,
                                            p=proportions)
    
        # Convert numpy.int64 indices to Python int
        selected_indices = [int(i) for i in selected_indices]
    
        selected_questions = {}
    
        # Populate the dictionary with question and probability pairs
        for idx, i in enumerate(selected_indices, start=1):
            selected_questions[f'q_{idx}'] = questions[i]['question']
            selected_questions[f'pr_{idx}'] = round(proportions[i], 4)  # Rounded for readability
    
        logger.info(f"Selected questions: {selected_questions}")
    
        return JsonResponse(selected_questions, safe=False)
    
    except Exception as e:
        logger.exception(f"Unexpected error in select_questions_simulation: {e}")
        return JsonResponse({'error': 'An unexpected error occurred.'}, status=500)
    # Getting all questions from the database
    questions = DynamicIssueQuestion.objects.all().values(
        'question', 'avg_rating', 'ratings', 'var_rating')
    # Convert database rows to means and covariances in numpy arrays
    means = []
    covariances = []
    for question in questions:
        means.append([question['avg_rating']])
        covariances.append(np.diag([question['var_rating']]))

    # Run 5000 simulations
    num_sims = 250
    results = []
    for _ in range(num_sims):
        samples = [
            np.random.multivariate_normal(mean, cov)
            for mean, cov in zip(means, covariances)
        ]
        max_value_index = np.argmax(samples, axis=0)[0]
        results.append(max_value_index)

    # Count the number of times each index had the highest value
    counts = np.bincount(results)
    proportions = counts / num_sims

    # Applying probability floor and renormalizing
    floor = 0.01
    proportions = np.maximum(proportions, floor)
    proportions /= proportions.sum()

    # Get midpoint
    midpoint = (float(get_global_setting('min_scale')) +
                float(get_global_setting('max_scale'))) / 2
    # If midpoint is empty, set it to 3
    if not midpoint:
        midpoint = 3

    num_items_str = get_global_setting('num_items')
    print(num_items_str)
    # Set a default value if num_items is not found or if it's None
    num_items = int(num_items_str) if num_items_str is not None else midpoint
    # Randomly select K questions based on the proportions
    selected_indices = np.random.choice(len(questions),
                                        size=num_items,
                                        replace=False,
                                        p=proportions)

    # Convert numpy.int64 indices to Python int
    selected_indices = [int(i) for i in selected_indices]

    selected_questions = {}

    # Populate the dictionary with question and probability pairs
    for idx, i in enumerate(selected_indices, start=1):
        selected_questions[f'q_{idx}'] = questions[i]['question']
        selected_questions[f'pr_{idx}'] = proportions[i]

    return JsonResponse(selected_questions, safe=False)

def survey_view(request):
    min_scale = get_global_setting('min_scale')  # Default to 1 if not set
    max_scale = get_global_setting('max_scale')  # Default to 5 if not set
    survey_text = get_global_setting('survey_text')

    if survey_text is None:
        survey_text = 'Rate the questions in importance on a 1-5 scale.'

    # Retrieve ai_choice from global settings
    ai_choice = get_global_setting('ai_choice')
    logger.debug(f"Global setting 'ai_choice': {ai_choice}")

    if ai_choice is None:
        ai_choice = 'openai'  # Default to 'openai' if not set

    # Add them to the context
    context = {
        'min_scale': min_scale,
        'max_scale': max_scale,
        'survey_text': survey_text,
        'ai_choice': ai_choice
    }

    return render(request, 'survey.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def submit_survey(request):
    # Process the survey data here
    # For example, save the data to the database

    # Redirect to a thank you page or another view
    return HttpResponseRedirect(reverse('thank_you'))
