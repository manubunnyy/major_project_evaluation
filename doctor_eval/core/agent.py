import os, re
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

class UnifiedAgent:
    """
    Simple, Reliable Agentic System
    - No complex voting (reduces errors)
    - Clean prompts that work
    - Focus on what actually helps
    """
    def __init__(self, model_name: str):
        self.mn = model_name
        self.llm = self._init_llm(temperature=0.3)  # Balanced

    def _init_llm(self, temperature: float):
        if "gemini" in self.mn.lower():
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=self.mn, google_api_key=os.getenv("GEMINI_API_KEY"), temperature=temperature, max_retries=0)
        
        gn = "llama-3.3-70b-versatile" if "gpt-oss" in self.mn else self.mn
        return ChatGroq(model_name=gn, groq_api_key=os.getenv("GROQ_API_KEY"), temperature=temperature)

    async def process_async(self, prompt: str) -> str:
        """Simple but effective agentic prompt"""
        try:
            # Enhanced prompt that helps without overcomplicating
            system_prompt = """You are an expert medical AI assistant.

{input}

Important guidelines:
- For medical questions: Think step-by-step, then select the BEST answer
- For writing constraints: Follow them EXACTLY (word counts, forbidden letters, format)
- Output ONLY your final answer - no explanations unless asked

Your response:"""
            
            response = await (ChatPromptTemplate.from_template(system_prompt) | self.llm | StrOutputParser()).ainvoke({"input": prompt})
            
            # Simple cleaning
            cleaned = response.strip()
            
            # Remove think tags if present
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = cleaned.strip()
            
            # Remove obvious markdown wrappers
            if cleaned.startswith("```") and cleaned.endswith("```"):
                lines = cleaned.split('\n')
                if len(lines) >= 3:
                    cleaned = '\n'.join(lines[1:-1]).strip()
            
            return cleaned

        except Exception as e:
            return f"Error: {e}"
