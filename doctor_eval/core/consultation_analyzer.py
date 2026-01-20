import os, re
from typing import Dict
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

class MedicalConsultationAnalyzer:
    def __init__(self, model_name: str):
        self.mn = model_name
        self.llm = self._init_llm(temperature=0.0)
        self.stylist_llm = self._init_llm(temperature=0.2)
        self.refiner_llm = self._init_llm(temperature=0.1) # Precision temp for critique

    def _init_llm(self, temperature: float):
        if any(x in self.mn.lower() for x in ["gemini", "gemma"]):
            return ChatGoogleGenerativeAI(model=self.mn, google_api_key=os.getenv("GEMINI_API_KEY"), temperature=temperature, max_retries=0)
        
        # Mapping Logic Cleaned:
        gn = "llama-3.3-70b-versatile" if "gpt-oss" in self.mn else self.mn
        # No more qwen mapping - letting it pass through
        
        return ChatGroq(model_name=gn, groq_api_key=os.getenv("GROQ_API_KEY"), temperature=temperature)

    def _clean_output(self, text: str) -> str:
        """Removes reasoning tags and meta-talk that hurt BERTScore F1."""
        # Strip <think> tags (common in reasoning models like Qwen/DeepSeek)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Strip any words like "(110 words)" or "Word count:"
        text = re.sub(r'\(?\d+\s+words\)?', '', text, flags=re.IGNORECASE)
        # Strip preamble/outro meta-talk
        text = text.replace("Final Optimized Response:", "").replace("Draft:", "").replace("Optimized Output:", "")
        return text.strip()

    async def analyze_consultation_async(self, instruction: str, patient_input: str) -> Dict[str, str]:
        """
        Hyper-Optimized Physician Agent. 
        Focuses on Semantic Precision (BERTScore) and Protocol Adherence.
        """
        try:
            # Step 1: Clinical Extraction
            reason_prompt = "Identify diagnosis and plan from: {input}\nInstruction: {instruction}"
            facts = await (ChatPromptTemplate.from_template(reason_prompt) | self.llm | StrOutputParser()).ainvoke({"input": patient_input, "instruction": instruction})
            
            # Step 2: Initial Draft (Junior Physician)
            synth_prompt = f"""
            Write a physician's reply based on: {facts}
            
            REFERENCE EXAMPLES (Use these to guide your TONE and STRUCTURE):
            
            Example 1 (Standard Query):
            "Hello, Thank you for posting your query. Based on your symptoms of [Condition], it is likely you are experiencing [Diagnosis]. I recommend starting with [Treatment]. Please ensure you stay hydrated and rest. If symptoms persist for more than 3 days, consult a specialist. Hope this helps. Thank you."
            
            Example 2 (Complex/Unknown):
            "Hello, Thank you for posting your query. Your symptoms are quite specific, but a definitive diagnosis requires a physical exam. It could be related to [Possibility A] or [Possibility B]. I strongly advise visiting a clinic for blood work. In the meantime, avoid [Aggravating Factor]. Hope this helps. Thank you."

            RULES:
            1. Start with "Hello, Thank you for posting your query."
            2. End with "Hope this helps. Thank you."
            3. Paragraph form ONLY. No lists.
            4. Clinical, professional tone. 
            5. LIMIT TO 100 WORDS.
            
            DRAFT:"""
            
            draft = await (ChatPromptTemplate.from_template(synth_prompt) | self.stylist_llm | StrOutputParser()).ainvoke({})
            
            # Step 3: Senior Physician Critique & Refine (The "Agentic" Boost)
            # This step specifically targets BERTScore F1 by aligning strictly with professional standards
            critique_prompt = f"""
            You are a Senior Physician reviewing a Junior Doctor's draft.
            
            Patient Input: "{patient_input}"
            Junior Draft:
            {draft}
            
            TASK:
            1. Check for missed key symptoms from the input.
            2. Ensure specific medical terminology is used correctly.
            3. Verify the tone is empathetic but professional.
            4. STRICTLY enforce the "Hello..." and "Hope this helps..." structure.
            
            Rewrite the response to be the absolute best clinical version. 
            ONLY output the final rewritten response.
            """
            
            refined_response = await (ChatPromptTemplate.from_template(critique_prompt) | self.refiner_llm | StrOutputParser()).ainvoke({})
            
            # Post-process: remove ALL artifacts
            final = refined_response.strip()
            
            # Remove think tags (multi-pass)
            while '<think>' in final.lower():
                final = re.sub(r'<think>.*?</think>', '', final, flags=re.DOTALL | re.IGNORECASE)
            
            final = final.strip()
            
            # Remove markdown wrappers
            if final.startswith("```") and final.endswith("```"):
                lines = final.split('\n')
                if len(lines) >= 3:
                    final = '\n'.join(lines[1:-1]).strip()
            
            return {"output": final} # Ensure the return type matches the method signature
            
        except Exception as e:
            # Fallback to a simpler prompt if the chain breaks (prevents NULL results)
            try:
                res = await self.llm.ainvoke(f"Answer as a physician: {patient_input}")
                return {"output": res.content.strip()}
            except:
                return {"output": "Error: Processing Failed"}
