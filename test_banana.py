import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, Modality

def main():
    # Load environment variables from .env
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found in .env or system environment.")
        sys.exit(1)
        
    print("Initializing GenAI client...")
    client = genai.Client(api_key=api_key)
    
    prompt = "A high-quality, realistic photo of a single ripe yellow banana lying on a clean white surface with soft shadows."
    model_name = "gemini-3.1-flash-image"
    
    print(f"Sending request to model '{model_name}'...")
    print(f"Prompt: '{prompt}'")
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=GenerateContentConfig(
                response_modalities=[Modality.IMAGE],
            ),
        )
    except Exception as e:
        print(f"Error during API call: {e}")
        sys.exit(1)
        
    print("API call successful. Processing response candidates...")
    
    if not response.candidates:
        print("Error: No candidates returned in response.")
        sys.exit(1)
        
    image_saved = False
    for i, candidate in enumerate(response.candidates):
        if not candidate.content or not candidate.content.parts:
            continue
            
        for j, part in enumerate(candidate.content.parts):
            if part.inline_data:
                data = part.inline_data.data
                mime_type = part.inline_data.mime_type
                
                # Determine correct extension
                ext = "png"
                if "jpeg" in mime_type or "jpg" in mime_type:
                    ext = "jpg"
                elif "webp" in mime_type:
                    ext = "webp"
                
                filename = f"banana.{ext}"
                print(f"Found image part with MIME type: {mime_type}. Saving to {filename}...")
                
                try:
                    with open(filename, "wb") as f:
                        f.write(data)
                    print(f"Successfully generated and saved image as: {os.path.abspath(filename)}")
                    image_saved = True
                except Exception as e:
                    print(f"Error saving image file: {e}")
                    
    if not image_saved:
        print("Warning: No inline image data found in response parts.")
        # Print parts structure for debugging
        print("Response content parts:", response.candidates[0].content.parts if response.candidates[0].content else "No content")

if __name__ == "__main__":
    main()
