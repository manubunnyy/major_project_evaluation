import os
from typing import Dict, List, Optional
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class HealthReportAnalyzer:
    """Enhanced health report analysis system with specialized agents, decoupled from Streamlit."""
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self._initialize_agents()

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on the model name."""
        if "gemini" in self.model_name.lower():
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
                temperature=0.3
            )
        else:
            # Assume Groq for other models (llama, qwen, openai-oss, etc.)
            return ChatGroq(
                temperature=0.3,
                model_name=self.model_name,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )

    def _initialize_agents(self):
        """Initialize specialized medical analysis agents"""
        self.agents = {
            'document_processor': self._create_agent("""
                You are a medical document processor specialized in health reports.
                Extract all relevant medical information, organize it clearly, and maintain accuracy.
                Focus on blood work, vital signs, and other measurable health metrics.
            """),
            
            'positive_analyzer': self._create_agent("""
                You are a positive health findings specialist.
                Identify and explain all positive health indicators in the report.
                
                IMPORTANT FORMATTING INSTRUCTIONS:
                1. Each finding MUST start on a new line with a checkmark symbol (✓)
                2. After each finding value and range, add a new line with two spaces of indentation for the significance
                3. Add a blank line between each complete finding
                
                Format each finding exactly like this:
                
                ✓ [Test Name]: [Value] [Unit] (normal range: [range])
                  Significance: [Brief explanation of why this is positive]
                
                [blank line here]
                ✓ [Next Test Name]: [Value] [Unit] (normal range: [range])
                  Significance: [Brief explanation of why this is positive]
                
                Example:
                ✓ Hemoglobin: 14.5 g/dL (normal range: 13.0-17.0 g/dL)
                  Significance: Excellent oxygen-carrying capacity, indicating good red blood cell function
                
                ✓ White Blood Cells: 7.2 x 10^9/L (normal range: 4.0-10.0 x 10^9/L)
                  Significance: Strong immune system function within optimal range
                
                Ensure each finding has:
                - Checkmark symbol at start
                - Clear test name and value
                - Normal range in parentheses
                - Significance on next line with indentation
                - Blank line after each complete finding
            """),
            
            'negative_analyzer': self._create_agent("""
                You are a health risk assessment specialist.
                Identify concerning findings and potential health risks.
                Format findings as bullet points starting with "⚠".
                Each finding must be on a new line.
                Include severity levels and recommended actions.
            """),
            
            'summary_agent': self._create_agent("""
                You are a medical report summarizer.
                Create a comprehensive yet concise summary of all findings.
                Include key metrics, trends, and important observations.
                Use clear, patient-friendly language.
                Format with clear sections and bullet points.
            """),
            
            'recommendation_agent': self._create_agent("""
                You are a healthcare recommendations specialist.
                Provide actionable advice based on the report findings.
                Include lifestyle, diet, and exercise recommendations.
                Prioritize suggestions by importance and urgency.
                Format each recommendation on a new line with clear categorization.
            """),
            
            'diet_planner': self._create_agent("""
                You are a specialized medical nutritionist who creates personalized diet plans.
                
                INSTRUCTIONS:
                1. Analyze the abnormal and low conditions in the medical report
                2. For each identified condition, provide specific dietary recommendations
                3. Create a complete 7-day meal plan addressing all health concerns
                4. Include specific foods to eat and avoid for each condition
                5. Prioritize evidence-based nutritional recommendations
                
                FORMAT YOUR RESPONSE:
                
                ## CONDITIONS REQUIRING DIETARY INTERVENTION
                - [List each condition with brief explanation]
                
                ## DIETARY RECOMMENDATIONS BY CONDITION
                ### [Condition 1]
                - Foods to include: [list with benefits]
                - Foods to avoid: [list with explanation]
                
                ### [Condition 2]
                - Foods to include: [list with benefits]
                - Foods to avoid: [list with explanation]
                
                ## 7-DAY OPTIMAL MEAL PLAN
                ### Day 1
                - Breakfast: [specific meal with ingredients]
                - Lunch: [specific meal with ingredients]
                - Dinner: [specific meal with ingredients]
                - Snacks: [options]
                
                [Continue for all 7 days]
                
                ## NUTRITIONAL SUPPLEMENTS
                - [List recommended supplements if needed]
                
                ## HYDRATION RECOMMENDATIONS
                - [Specific recommendations]
            """)
        }

    def _create_agent(self, system_prompt: str):
        """Create an agent with specific system prompt"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        return prompt | self.llm | StrOutputParser()

    async def analyze_report_async(self, report_text: str) -> Dict[str, str]:
        """Run a sequential agentic chain with token optimization."""
        results = {}
        try:
            # Step 1: Data Extraction & Processing
            # Truncate raw text to stay within request limits (Strict 4000 char limit)
            safe_text = report_text[:4000] if len(report_text) > 4000 else report_text
            
            processor_prompt = """
            Extract all medical lab results, values, and reference ranges.
            Highlight any values outside normal limits with '(!)'
            Be extremely concise. Output ONLY the findings list.
            """
            extraction = await self._create_agent(processor_prompt).ainvoke({"input": safe_text})
            results['extracted_data'] = extraction

            # Step 2 & 3: Focused Analysis
            # We NO LONGER send report_text as 'input'. We send the extraction.
            risk_prompt = """
            Analyze the following extracted medical findings.
            Identify only the abnormal or concerning results and explain their clinical significance.
            Input: {input}
            """
            risks = await self._create_agent(risk_prompt).ainvoke({"input": extraction})
            results['negative_analyzer'] = risks

            positive_prompt = """
            Identify the key healthy metrics from these findings.
            Input: {input}
            """
            positives = await self._create_agent(positive_prompt).ainvoke({"input": extraction})
            results['positive_analyzer'] = positives

            # Step 4: Final Clinical Synthesis
            # Matches ground truth style (concise sentences).
            summary_prompt = """
            Synthesize a 1-3 sentence clinical summary.
            Focus on the primary health status and significant findings.
            Clinical Analysis: {input}
            """
            combined_analysis = f"Abnormals: {risks}\nNormals: {positives}"
            final_summary = await self._create_agent(summary_prompt).ainvoke({"input": combined_analysis})
            
            results['summary'] = final_summary
            results['summary_agent'] = final_summary
                
            return results

        except Exception as e:
            return {"summary": f"Error during analysis: {str(e)}"}


