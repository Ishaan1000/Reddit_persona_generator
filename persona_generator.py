import praw
from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv("D:/beyond_chat_assignment_persona_generator/api.env")

def setup_reddit_client():
    """Initialize and return the Reddit API client."""
    return praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent="UserPersonaScraper/1.0"
    )

def fetch_user_content(reddit, username, limit=25):
    """Fetch user's posts and comments from Reddit."""
    try:
        user = reddit.redditor(username)
        content = []
        
        # Get posts
        for submission in user.submissions.new(limit=limit):
            content.append({
                'type': 'post',
                'title': submission.title,
                'text': submission.selftext,
                'subreddit': submission.subreddit.display_name,
                'date': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d'),
                'url': f"https://reddit.com{submission.permalink}"
            })
        
        # Get comments
        for comment in user.comments.new(limit=limit):
            content.append({
                'type': 'comment',
                'text': comment.body,
                'subreddit': comment.subreddit.display_name,
                'date': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d'),
                'url': f"https://reddit.com{comment.permalink}"
            })
        
        return content
    except Exception as e:
        print(f"❌ Error fetching data for user {username}: {str(e)}")
        return []

def initialize_gpt2_large():
    """Initialize GPT2-large model with optimized settings."""
    print("🔄 Loading GPT2-large model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    
    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Set model to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"✅ Model loaded on {device}")
    return pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

def generate_persona(generator, content):
    """Generate persona using GPT2-large with proper data display"""
    if not content:
        return None

    # Prepare clean, visible samples for analysis
    visible_samples = []
    for item in content[:5]:  # Use first 5 items
        if item['text'].strip():  # Only include non-empty content
            visible_samples.append(
                f"{item['type'].upper()} in r/{item['subreddit']} ({item['date']}):\n"
                f"{item['text'][:200]}{'...' if len(item['text']) > 200 else ''}\n"
                f"URL: {item['url']}\n"
            )

    if not visible_samples:
        return "⚠️ No valid content found to analyze"

    samples_text = "\n".join(visible_samples)
    
    prompt = f"""
    Analyze these Reddit activities and create a detailed persona:

    # Reddit User Persona

    **AGE**  
    [Estimate based on: {samples_text[:100]}...]

    **OCCUPATION**  
    [From keywords: {samples_text[:200]}...]

    **INTERESTS**  
    - {content[0]['subreddit'] if content else 'Unknown'}
    - [Add more interests]

    ---

    **PERSONALITY**

    | Analytical | Passionate |
    |---|---|
    | Patient | Detail-oriented |

    ---

    **BEHAVIOUR & HABITS**

    - Collects Hot Wheels: "{content[2]['text'][:100]}..." (Source: {content[2]['url']})
    - [Add more habits]

    ---

    **GOALS & NEEDS**

    - To complete collections
    - To find rare models
    - [Add more goals]

    ---

    **FRUSTRATIONS**

    - Scalpers: "{content[2]['text'][:100]}..." (Source: {content[2]['url']})
    - [Add more frustrations]

    Raw Data Analyzed:
    {samples_text}
    """
    
    try:
        print("🧠 Generating persona from this data:")
        print(samples_text)  # Debug: Show what's being analyzed
        
        response = generator(
            prompt,
            max_length=1500,
            num_return_sequences=1,
            temperature=0.7
        )
        return response[0]['generated_text']
    except Exception as e:
        print(f"❌ Generation error: {str(e)}")
        return f"Error: {str(e)}\n\nData that was analyzed:\n{samples_text}"

def save_persona(username, persona, output_dir="output"):
    """Save the generated persona to a text file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"{output_dir}/{username}_persona.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(persona)
    
    print(f"✅ Persona saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Reddit User Persona Generator')
    parser.add_argument('url', help='Reddit user profile URL')
    args = parser.parse_args()
    
    # Extract username from URL
    url_parts = args.url.strip('/').split('/')
    if 'user' not in url_parts:
        print("❌ Invalid Reddit user profile URL")
        return
    
    username = url_parts[url_parts.index('user') + 1]
    
    # Initialize clients
    print("🔧 Initializing systems...")
    reddit = setup_reddit_client()
    generator = initialize_gpt2_large()
    
    # Get user content
    print(f"🔍 Fetching content for u/{username}...")
    user_content = fetch_user_content(reddit, username)
    
    if not user_content:
        print("❌ No content found or error fetching content.")
        return
    
    # Generate persona
    persona = generate_persona(generator, user_content)
    
    if persona:
        save_persona(username, persona)
    else:
        print("❌ Failed to generate persona.")

if __name__ == "__main__":
    main()