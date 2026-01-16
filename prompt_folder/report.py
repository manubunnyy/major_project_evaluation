import os
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
import requests
import json
import re

class HealthReportAnalyzer:
    """Enhanced health report analysis system with specialized agents"""
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = None
        self._initialize_agents()

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

    def _format_findings(self, response: str) -> str:
        """Format the findings to ensure proper line breaks and spacing"""
        # First, ensure we're working with a string
        if not isinstance(response, str):
            return str(response)

        # Split on checkmark symbol, preserving the symbol
        findings = [f.strip() for f in response.split('✓') if f.strip()]
        
        formatted_findings = []
        for finding in findings:
            # Split finding into main result and significance (if present)
            lines = [line.strip() for line in finding.split('\n') if line.strip()]
            
            if lines:
                # Format main finding line
                main_finding = lines[0]
                formatted_finding = [f"✓ {main_finding}"]
                
                # Format significance and any additional lines
                for line in lines[1:]:
                    if line.startswith('Significance:'):
                        formatted_finding.append(f"  {line}")
                    else:
                        formatted_finding.append(f"  {line}")
                
                # Join the lines for this finding
                formatted_findings.append('\n'.join(formatted_finding))
        
        # Join all findings with double newlines for spacing
        return '\n\n'.join(formatted_findings)

    async def process_document(self, file_content: str):
        """Process document and create vector store"""
        try:
            chunks = self.text_splitter.split_text(file_content)
            
            if not chunks:
                raise ValueError("No text chunks were created from the document")

            self.vectorstore = await FAISS.afrom_texts(
                texts=chunks,
                embedding=self.embeddings,
                normalize_L2=True
            )
            
            return chunks
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            raise

    async def analyze_report(self, report_text: str, agent_status: AgentStatus):
        """Analyze report using multiple agents with dynamic status updates"""
        results = {}
        
        agent_status.update_status(
            'document_processor',
            'working',
            0.0,
            'Processing document...'
        )
        
        try:
            await self.process_document(report_text)
            agent_status.update_status(
                'document_processor',
                'completed',
                1.0,
                'Document processed'
            )
            
            agents_list = list(self.agents.items())
            for idx, (agent_name, agent) in enumerate(agents_list[1:], 1):
                agent_status.update_status(
                    agent_name,
                    'working',
                    0.0,
                    f'Starting analysis...'
                )
                
                start_time = time.time()
                
                try:
                    # Get relevant context
                    if self.vectorstore is not None:
                        relevant_docs = await self.vectorstore.asimilarity_search(
                            agent_name,
                            k=3
                        )
                        context = "\n".join(doc.page_content for doc in relevant_docs)
                        augmented_text = f"Context: {context}\n\nReport: {report_text}"
                    else:
                        augmented_text = report_text
                    
                    agent_status.update_status(
                        agent_name,
                        'working',
                        0.5,
                        'Analyzing content...'
                    )
                    
                    # Get response and apply formatting for positive_analyzer
                    response = await agent.ainvoke({"input": augmented_text})
                    if agent_name == 'positive_analyzer':
                        response = self._format_findings(response)
                    
                    processing_time = time.time() - start_time
                    
                    results[agent_name] = AgentResponse(
                        agent_name=agent_name,
                        content=response,
                        confidence=0.9,
                        processing_time=processing_time
                    )
                    
                    agent_status.update_status(
                        agent_name,
                        'completed',
                        1.0,
                        'Analysis complete'
                    )
                    
                except Exception as e:
                    results[agent_name] = AgentResponse(
                        agent_name=agent_name,
                        content=f"Error: {str(e)}",
                        confidence=0.0,
                        processing_time=0.0
                    )
                    
                    agent_status.update_status(
                        agent_name,
                        'error',
                        1.0,
                        f'Error: {str(e)}'
                    )
            
            return results
        except Exception as e:
            st.error(f"Error in analyze_report: {str(e)}")
            raise

    async def generate_chat_response(self, query: str, context: str) -> str:
        """Generate chat response using RAG"""
        try:
            if self.vectorstore is not None:
                relevant_chunks = await self.vectorstore.asimilarity_search(
                    query,
                    k=3
                )
                additional_context = "\n".join(chunk.page_content for chunk in relevant_chunks)
                full_context = f"{context}\n\nAdditional Context: {additional_context}"
            else:
                full_context = context

            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a medical report assistant. Use the provided context to:
                    1. Answer questions accurately
                    2. Explain medical terms simply
                    3. Provide evidence-based responses
                    4. Maintain a helpful, professional tone"""),
                ("human", "{query}"),
                ("system", "Context: {context}")
            ])
            
            chain = chat_prompt | self.llm | StrOutputParser()
            
            response = await chain.ainvoke({
                "query": query,
                "context": full_context
            })
            
            return response
        except Exception as e:
            error_message = f"Error generating chat response: {str(e)}"
            st.error(error_message)
            return f"I apologize, but I encountered an error: {str(e)}"

    async def web_search_diet_info(self, abnormal_conditions: List[str]) -> str:
        """Search the web for diet recommendations based on abnormal conditions"""
        try:
            search_results = []
            
            for condition in abnormal_conditions:
                search_query = f"evidence based diet recommendations for {condition}"
                
                # Simulate web search results
                search_result = f"### Diet Information for {condition}\n"
                search_result += "Based on recent medical research:\n"
                search_result += "- Recommended foods: [would be populated from actual search]\n"
                search_result += "- Foods to avoid: [would be populated from actual search]\n"
                search_result += "- Recent studies suggest: [would be populated from actual search]\n\n"
                
                search_results.append(search_result)
            
            return "\n".join(search_results)
        except Exception as e:
            return f"Error searching for diet information: {str(e)}"

    async def extract_abnormal_conditions(self, report_text: str) -> List[str]:
        """Extract abnormal conditions from the report text"""
        try:
            # Create a specialized prompt for extracting abnormal conditions
            extract_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a medical condition extractor.
                    Extract all abnormal test results and conditions from the provided medical report.
                    Return ONLY a list of specific conditions, one per line.
                    DO NOT include normal results.
                    Example output:
                    Low Vitamin D
                    Elevated LDL cholesterol
                    Hypothyroidism"""),
                ("human", "{report}")
            ])
            
            chain = extract_prompt | self.llm | StrOutputParser()
            
            conditions_text = await chain.ainvoke({"report": report_text})
            
            # Split by newlines and clean up
            conditions = [
                cond.strip() for cond in conditions_text.split('\n')
                if cond.strip() and not cond.startswith("Normal")
            ]
            
            return conditions
        except Exception as e:
            st.error(f"Error extracting conditions: {str(e)}")
            return []

    async def generate_diet_plan(self, report_text: str, agent_status: AgentStatus) -> str:
        """Generate comprehensive diet plan based on report findings"""
        try:
            agent_status.update_status(
                'diet_planner',
                'working',
                0.2,
                'Analyzing health conditions...'
            )
            
            # Extract abnormal conditions
            conditions = await self.extract_abnormal_conditions(report_text)
            
            agent_status.update_status(
                'diet_planner',
                'working',
                0.4,
                'Formulating diet recommendations...'
            )
            
            # Get diet recommendations
            diet_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a specialized medical nutritionist.
                    Create a comprehensive diet plan addressing the specific abnormal conditions listed.
                    Include scientific rationale for each recommendation.
                    Format as:
                    1. Analysis of each condition and its nutritional implications
                    2. Specific foods to eat and avoid for each condition
                    3. A detailed 7-day meal plan with recipes
                    4. Supplement recommendations if needed"""),
                ("human", "Create a personalized diet plan for these conditions: {conditions}")
            ])
            
            chain = diet_prompt | self.llm | StrOutputParser()
            
            diet_plan = await chain.ainvoke({"conditions": "\n".join(conditions)})
            
            agent_status.update_status(
                'diet_planner',
                'working',
                0.7,
                'Searching for additional information...'
            )
            
            # Get web search results
            agent_status.update_status(
                'web_search',
                'working',
                0.5,
                'Searching for diet information...'
            )
            
            web_results = await self.web_search_diet_info(conditions)
            
            agent_status.update_status(
                'web_search',
                'completed',
                1.0,
                'Search completed'
            )
            
            # Combine diet plan with web results
            combined_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a medical nutritionist creating the optimal diet plan.
                    Combine the AI-generated diet plan with web research to create the most 
                    comprehensive and evidence-based recommendations.
                    Keep formatting clear with headers, bullet points, and a 7-day meal plan."""),
                ("human", """
                AI Diet Plan:
                {diet_plan}
                
                Web Research:
                {web_results}
                
                Create an optimized diet plan combining this information.
                """)
            ])
            
            chain = combined_prompt | self.llm | StrOutputParser()
            
            final_diet_plan = await chain.ainvoke({
                "diet_plan": diet_plan,
                "web_results": web_results
            })
            
            agent_status.update_status(
                'diet_planner',
                'completed',
                1.0,
                'Diet plan completed'
            )
            
            return final_diet_plan
        except Exception as e:
            agent_status.update_status(
                'diet_planner',
                'error',
                1.0,
                f'Error: {str(e)}'
            )
            return f"Error creating diet plan: {str(e)}"

def handle_chat_input():
    """Handle chat input and response"""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    if "chat_input_key" not in st.session_state:
        st.session_state.chat_input_key = 0
    
    if "processing_message" not in st.session_state:
        st.session_state.processing_message = False
    
    # Display existing chat messages
    for message in st.session_state.chat_messages:
        if isinstance(message, HumanMessage):
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message.content}
                </div>
            """, unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message.content}
                </div>
            """, unsafe_allow_html=True)
    
    # Chat input area
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question about your report:",
            key=f"chat_input_{st.session_state.chat_input_key}"
        )
    with col2:
        send_button = st.button("Send")
    
    if send_button and user_input and not st.session_state.processing_message:
        st.session_state.processing_message = True
        
        # Update chat assistant status
        st.session_state.agent_status.update_status(
            'chat_assistant',
            'working',
            0.5,
            'Processing your question...'
        )
        
        human_message = HumanMessage(content=user_input)
        st.session_state.chat_messages.append(human_message)
        
        if st.session_state.report_text:
            response = asyncio.run(
                st.session_state.analyzer.generate_chat_response(
                    user_input,
                    st.session_state.report_text
                )
            )
            
            ai_message = AIMessage(content=response)
            st.session_state.chat_messages.append(ai_message)
        
        # Update chat assistant status to completed
        st.session_state.agent_status.update_status(
            'chat_assistant',
            'completed',
            1.0,
            'Response generated'
        )
        
        st.session_state.processing_message = False
        st.session_state.chat_input_key += 1
        st.rerun()
        
def display_workflow():
    """Display the analysis workflow"""
    with st.container():
        st.markdown("""
            <div class="workflow-container">
                <h3>How It Works</h3>
                <div class="workflow-step">
                    1. Upload Your Health Report (PDF/TXT)
                </div>
                <div class="workflow-step">
                    2. AI Agents Analyze Your Report
                </div>
                <div class="workflow-step">
                    3. Get Comprehensive Analysis & Insights
                </div>
                <div class="workflow-step">
                    4. Chat with AI About Your Results
                </div>
            </div>
        """, unsafe_allow_html=True)

def main():
    """Main application with enhanced UI and dark theme"""
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = HealthReportAnalyzer()
    if 'report_results' not in st.session_state:
        st.session_state.report_results = None
    if 'report_text' not in st.session_state:
        st.session_state.report_text = None
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = AgentStatus()
    if 'diet_plan' not in st.session_state:
        st.session_state.diet_plan = None
    
    # Sidebar
    with st.sidebar:
        st.title("🏥 Health Report Analyzer")
        st.markdown("---")
        
        # File upload in sidebar
        uploaded_file = st.file_uploader(
            "Upload your health report",
            type=['pdf', 'txt'],
            key='file_uploader'
        )
        
        if uploaded_file:
            if st.button("🔍 Analyze Report", key='analyze_btn'):
                try:
                    # Read document
                    if uploaded_file.type == "application/pdf":
                        pdf_reader = PdfReader(uploaded_file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                    else:
                        text = uploaded_file.getvalue().decode()
                    
                    st.session_state.report_text = text
                    
                    # Initialize agent status display after file upload
                    st.session_state.agent_status.initialize_sidebar_placeholder()
                    
                    # Reset all agent statuses to idle
                    for agent_name in st.session_state.agent_status.agents:
                        st.session_state.agent_status.update_status(
                            agent_name,
                            'idle',
                            0.0,
                            'Waiting to start...'
                        )
                    
                    # Analyze report with dynamic status updates
                    st.session_state.report_results = asyncio.run(
                        st.session_state.analyzer.analyze_report(
                            text,
                            st.session_state.agent_status
                        )
                    )
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing report: {str(e)}")
        
        # Display agent status after file upload section
        if st.session_state.report_text:
            st.session_state.agent_status.initialize_sidebar_placeholder()
    
    # Main content area
    st.title("Health Report Analysis")
    
    if not st.session_state.report_results:
        display_workflow()
    else:
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "✅ Positive Findings",
            "⚠️ Areas of Concern",
            "🥗 Personalized Diet Plan",
            "📊 Full Report",
            "💬 Chat Assistant"
        ])
        
        with tab1:
            if 'positive_analyzer' in st.session_state.report_results:
                result = st.session_state.report_results['positive_analyzer']
                st.markdown(result.content)
        
        with tab2:
            if 'negative_analyzer' in st.session_state.report_results:
                result = st.session_state.report_results['negative_analyzer']
                st.markdown(result.content)
                
                # Add button to generate diet plan
                if st.button("🥗 Generate Personalized Diet Plan", key="generate_diet_btn"):
                    st.session_state.agent_status.update_status(
                        'diet_planner',
                        'working',
                        0.1,
                        'Starting diet plan generation...'
                    )
                    
                    with st.spinner("Generating your personalized diet plan..."):
                        diet_plan = asyncio.run(
                            st.session_state.analyzer.generate_diet_plan(
                                st.session_state.report_text,
                                st.session_state.agent_status
                            )
                        )
                        st.session_state.diet_plan = diet_plan
                    
                    # Switch to diet plan tab
                    st.rerun()
        
        with tab3:
            if st.session_state.diet_plan:
                st.markdown(st.session_state.diet_plan)
            else:
                st.info("No diet plan generated yet. Go to 'Areas of Concern' tab and click 'Generate Personalized Diet Plan'.")
        
        with tab4:
            if 'document_processor' in st.session_state.report_results:
                st.subheader("Document Analysis")
                st.markdown(st.session_state.report_results['document_processor'].content)
            
            if 'summary_agent' in st.session_state.report_results:
                st.subheader("Summary")
                st.markdown(st.session_state.report_results['summary_agent'].content)
            
            if 'recommendation_agent' in st.session_state.report_results:
                st.subheader("Recommendations")
                st.markdown(st.session_state.report_results['recommendation_agent'].content)
        
        with tab5:
            handle_chat_input()

if __name__ == "__main__":
    main()
