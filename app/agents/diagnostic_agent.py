from typing import Any, Dict, List, Optional
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.services.scraper_service import ScraperService
from app.CRUD.car_crud import CarCRUD
from app.utils.diagnosis_tree import DiagnosisTreeNode
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import asyncio
import numpy as np
import traceback
from app.core.interfaces import IWebSocketManager
import json
from datetime import datetime
from app.services.search_engine_service import SearchEngineService
from app.agents.base_agent import BaseAgent
from app.utils.message_types import MessageSource
from app.utils.monitoring import monitor_and_handle
from app.agents.session_context_agent import SessionContextAgent

class DiagnosisAgent(BaseAgent):
    """
    Agentic RAG for generating a diagnosis using the diagnosis tree, user message, and multi-source context.
    """
    def __init__(
        self,
        car_id: str,
        diagnosis_tree: DiagnosisTreeNode,
        car_make: Optional[str] = None,
        car_model: Optional[str] = None,
        car_year: Optional[str] = None,
        search_engine_service: Optional[SearchEngineService] = None,
        websocket_manager: Optional[IWebSocketManager] = None,
        llm_service: Optional[LLMService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        scraper_service: Optional[ScraperService] = None,
        session_context_manager: Optional[SessionContextAgent] = None,
        **kwargs
    ):
        """
        Initialize the DiagnosisAgent with all dependencies injected for testability.
        """
        super().__init__(car_crud=CarCRUD(), car_id=car_id, car_make=car_make, car_model=car_model, car_year=car_year, logger_name="DiagnosisAgent", websocket_manager=websocket_manager)
        self.diagnosis_tree = diagnosis_tree
        self.llm_service = llm_service or LLMService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.scraper_service = scraper_service or ScraperService(headless=True)
        self.search_engine_service = search_engine_service or SearchEngineService()
        self.session_context_manager = session_context_manager or SessionContextAgent()
        self.car_make = car_make
        self.car_model = car_model
        self.car_year = car_year
        self.prompt = PromptTemplate.from_template(
            """
            You are an expert automotive diagnostician with access to an extensive knowledge base of 38,936 automotive documents. Your primary goal is to EMPOWER users to diagnose and fix automotive issues themselves by providing comprehensive, detailed technical guidance. Most users want to perform their own repairs, so provide in-depth, step-by-step instructions that enable successful DIY repairs.

            Focus on DIY empowerment - assume the user wants to fix the issue themselves and provide the knowledge and confidence to do so safely. Only recommend professional help when absolutely necessary for safety or when specialized equipment is required.

            CRITICAL: ALWAYS USE THE DIAGNOSIS TREE AS YOUR PRIMARY REFERENCE
            The diagnosis tree contains structured symptom data extracted from the user's descriptions. This tree represents the most likely issues based on the user's reported symptoms, with likelihood percentages indicating priority. ALWAYS prioritize your diagnosis and recommendations based on the tree's highest likelihood symptoms first.

            SESSION CONTEXT AWARENESS:
            - This conversation is part of a focused session on a specific automotive issue
            - ALWAYS prioritize the original issue context when interpreting new messages
            - If session context is provided, use it to maintain diagnostic focus
            - Connect all symptoms and follow-ups to the original issue category
            - Do not drift to unrelated automotive topics

            CORE PRINCIPLES:
            - START with the highest likelihood symptoms from the diagnosis tree
            - Use the tree's hierarchical structure to understand symptom relationships
            - Maintain focus on the session's original issue context
            - Provide detailed technical explanations that build user understanding
            - Give specific part numbers, specifications, and technical details when possible
            - Explain the WHY behind each step so users understand the reasoning
            - Include common pitfalls and how to avoid them
            - Emphasize safety but don't overstate risks for standard procedures

            IMPORTANT:
            - You have access to the car's make, model, and year: Make: {car_make}, Model: {car_model}, Year: {car_year}.
            - Do NOT ask the user for car make, model, or year; you already have this information.
            - Do NOT tell the user to refer to the owner's manual. Extract and use information directly.
            - You have access to a comprehensive automotive knowledge base with nearly 39,000 documents covering all automotive systems, repair procedures, and technical specifications.
            - ASSUME the user wants to do the repair themselves and provide detailed guidance to make that possible.
            - If session context is provided, use it to interpret user messages within the original issue scope.

            CONTEXT SOURCES PRIORITY ORDER:
            1. SESSION CONTEXT: Original issue context and accumulated symptoms from this conversation
            2. DIAGNOSIS TREE: Primary structured symptom and issue data (USE THIS FIRST)
            3. Knowledge base: Comprehensive automotive repair and diagnostic information (38,936 documents)
            4. Owner's manual/context: Official technical specifications and procedures
            5. Online context: Recent technical discussions and repair experiences

            INSTRUCTIONS FOR SESSION-AWARE TREE-GUIDED DIY-FOCUSED ANALYSIS:
            1. EXAMINE session context first if provided - understand the original issue scope
            2. EXAMINE the diagnosis tree carefully - identify the highest likelihood symptoms (>70%)
            3. PRIORITIZE your diagnosis around these high-likelihood symptoms from the tree
            4. CONNECT all symptoms to the original issue context if session context is available
            5. For each tree symptom, provide detailed technical explanations with specific repair procedures
            6. Use the tree's hierarchical relationships to understand symptom connections
            7. Include multiple diagnostic approaches - visual inspection, electrical testing, mechanical testing, etc.
            8. Provide specific torque specifications, part numbers, fluid specifications, and technical details.
            9. Explain the underlying automotive systems so users understand what they're working on.
            10. Give detailed step-by-step repair procedures with professional-level detail.
            11. Include troubleshooting steps for when things don't go as expected.
            12. Explain how to verify the repair was successful and prevent recurrence.
            13. Include tips and tricks from professional mechanics.
            14. Include maintenance schedules and inspection points to prevent similar issues.

            OUTPUT (COMPREHENSIVE SESSION-AWARE DIY-FOCUSED JSON):
            {{
                "diagnosis_summary": "SESSION-AWARE technical diagnosis starting with the highest likelihood symptoms from the diagnosis tree, focused on the original issue context, with detailed repair-focused analysis and system explanations",
                "session_context_analysis": {{
                    "original_issue_focus": "How this diagnosis relates to the session's original issue",
                    "symptom_connections": "How new symptoms connect to the original issue context",
                    "diagnostic_focus": "Maintained focus on the original issue category"
                }},
                "tree_analysis": {{
                    "primary_symptoms": ["List the highest likelihood symptoms from the tree (>70%)"],
                    "secondary_symptoms": ["Medium likelihood symptoms from the tree (30-70%)"],
                    "symptom_relationships": "Explain how the tree symptoms relate to each other and point to common root causes",
                    "tree_guided_diagnosis": "Primary diagnosis based specifically on the tree's highest likelihood symptoms"
                }},
                "supporting_evidence": [
                    {{"source": "session_context", "evidence": "How session context informs this diagnosis", "confidence": "High/Medium/Low", "context_relevance": "Connection to original issue"}},
                    {{"source": "diagnosis_tree", "evidence": "Technical evidence FROM THE TREE", "confidence": "High/Medium/Low", "diy_relevance": "How this tree data helps DIY diagnosis", "tree_symptom": "Specific symptom from tree"}},
                    {{"source": "owner_manual", "evidence": "Technical specifications and procedures", "confidence": "High/Medium/Low", "specific_details": "Torque specs, part numbers, etc."}},
                    {{"source": "knowledge_base", "evidence": "Detailed repair procedures", "book_title": "...", "page": "...", "confidence": "High/Medium/Low", "diy_tips": "Professional insights for DIYers"}},
                    {{"source": "online", "evidence": "Real-world repair experiences", "url": "...", "confidence": "High/Medium/Low", "practical_insights": "What actually works in practice"}}
                ],
                "diy_repair_procedures": [
                    {{
                        "procedure_name": "Primary repair procedure",
                        "difficulty_level": "Beginner/Intermediate/Advanced",
                        "estimated_time": "Detailed time breakdown",
                        "required_tools": ["Specific tool with size/type", "Tool 2 with specifications"],
                        "required_parts": [
                            {{
                                "part_name": "Specific part name",
                                "part_number": "OEM or aftermarket part number",
                                "specifications": "Technical specifications",
                                "cost_range": "Typical cost range",
                                "where_to_buy": "Best sources for parts"
                            }}
                        ],
                        "detailed_steps": [
                            {{
                                "step_number": 1,
                                "action": "Detailed step description",
                                "technical_details": "Torque specs, measurements, etc.",
                                "safety_notes": "Specific safety considerations",
                                "common_mistakes": "What to avoid",
                                "verification": "How to verify this step was done correctly"
                            }}
                        ],
                        "troubleshooting": [
                            {{
                                "problem": "What might go wrong",
                                "causes": ["Possible causes"],
                                "solutions": ["How to fix it"]
                            }}
                        ]
                    }}
                ],
                "alternative_diagnoses": [
                    {{
                        "name": "Alternative technical diagnosis",
                        "likelihood": "Percentage with technical reasoning",
                        "distinguishing_features": ["Technical indicators"],
                        "diagnostic_procedures": [
                            {{
                                "test_name": "Specific diagnostic test",
                                "tools_needed": ["Required tools"],
                                "procedure": "Step-by-step testing procedure",
                                "expected_results": "What results indicate this diagnosis",
                                "interpretation": "How to interpret the results"
                            }}
                        ],
                        "difficulty_assessment": "Why this might be easier/harder to fix"
                    }}
                ],
                "diagnostic_procedures": [
                    {{
                        "test_name": "Comprehensive diagnostic test",
                        "purpose": "What this test determines",
                        "tools_required": ["Specific tools with specifications"],
                        "step_by_step_procedure": ["Detailed testing steps"],
                        "normal_values": "Expected readings/results",
                        "interpretation_guide": "How to interpret different results",
                        "next_steps_based_on_results": "What to do based on what you find"
                    }}
                ],
                "quality_assurance": [
                    {{
                        "checkpoint": "What to verify",
                        "testing_procedure": "How to test the repair",
                        "success_criteria": "How to know it's working correctly",
                        "common_issues": "Problems that might still exist"
                    }}
                ],
                "maintenance_schedule": {{
                    "immediate_follow_up": ["What to check in first week/month"],
                    "ongoing_maintenance": ["Regular maintenance to prevent recurrence"],
                    "inspection_points": ["What to monitor long-term"],
                    "replacement_intervals": ["When to replace preventively"]
                }},
                "professional_consultation_indicators": [
                    {{
                        "scenario": "Specific situation requiring professional help",
                        "why_professional_needed": "Technical reason why DIY isn't recommended",
                        "what_to_tell_mechanic": "How to communicate the issue effectively",
                        "estimated_shop_cost": "What to expect to pay"
                    }}
                ],
                "confidence": "Technical confidence assessment with detailed reasoning"
            }}

            INPUT:
            - Car make: {car_make}
            - Car model: {car_model}
            - Car year: {car_year}
            - Session context: {session_context}
            - User messages (last 5, most recent last): {user_message}
            - Diagnosis tree: {tree_summary}
            - Owner's manual/context: {manual_context}
            - Knowledge base (from 38,936 documents): {kb_context}
            - Online context: {online_context}

            Ensure the JSON is valid and well-formed. Provide comprehensive technical details that enable successful DIY repairs while maintaining focus on the session's original issue.
            Return ONLY a valid JSON object. Do NOT include any extra text, markdown, or explanations outside the JSON.
            """
        )

    async def retrieve_context(self, user_message: str) -> Dict[str, Any]:
        stage_msg = {"type": "stage", "stage": "Retrieving context"}
        await self.logger.info(f"Broadcasting stage: {stage_msg['stage']}")
        await self.broadcast_stage(json.dumps(stage_msg))
        await self._ensure_car_info()
        
        # PERFORMANCE OPTIMIZATION: Run searches in parallel
        async def get_manual_context():
            try:
                # Try to get manual chunks using direct Milvus search - REDUCED for performance
                manual_chunks = await self.search_engine_service.embed_and_vector_search(
                    car_id=self.car_id, query=user_message, top_k=2  # Reduced from 3 to 2 for performance
                )
                if manual_chunks:
                    # Combine the top chunks into context - REDUCED content length
                    return "\n\n".join([
                        f"[Manual Section {i+1}]: {chunk['chunk'][:600]}"  # Reduced from full chunk to 600 chars
                        for i, chunk in enumerate(manual_chunks)
                    ])
                else:
                    # Try normalized car ID if original failed
                    normalized_car_id = self.car_id
                    parts = self.car_id.lower().split('-')
                    if len(parts) >= 3:
                        common_makes = ['toyota', 'honda', 'ford', 'chevrolet', 'bmw', 'audi', 'mercedes', 'nissan', 
                                      'mazda', 'subaru', 'hyundai', 'kia', 'lexus', 'acura', 'volkswagen', 'vw']
                        if parts[0] not in common_makes and parts[1] in common_makes:
                            normalized_car_id = f"{parts[1]}-{parts[0]}-{parts[2]}"
                            
                            # Try again with normalized ID if different
                            if normalized_car_id != self.car_id:
                                await self.logger.info(f"Trying normalized car_id: {normalized_car_id}")
                                manual_chunks = await self.search_engine_service.embed_and_vector_search(
                                    car_id=normalized_car_id, query=user_message, top_k=2
                                )
                                if manual_chunks:
                                    return "\n\n".join([
                                        f"[Manual Section {i+1}]: {chunk['chunk'][:600]}"
                                        for i, chunk in enumerate(manual_chunks)
                                    ])
                return ""
            except Exception as e:
                await self.logger.error(f"Error retrieving manual text: {str(e)}")
                return ""
        
        async def get_knowledge_context():
            # MAJOR PERFORMANCE IMPROVEMENT: Drastically reduce search scope
            docs = await self.search_engine_service.search(self.car_id, user_message, top_k=15)  # Reduced from 30 to 15
            
            # Process only knowledge base docs - FURTHER REDUCED for performance
            kb_docs = [d for d in docs if d.metadata.get("source") == "ground_knowledge"]
            
            # Create knowledge base context - HEAVILY REDUCED for performance
            kb_context_parts = []
            for i, doc in enumerate(kb_docs[:4]):  # Reduced from 8 to 4 for much faster processing
                book_title = doc.metadata.get("book_title", "Unknown Source")
                page_num = doc.metadata.get("page_number", "N/A")
                content = doc.page_content[:500]  # Reduced from 1000 to 500 chars
                kb_context_parts.append(f"[Source: {book_title}, Page: {page_num}]\n{content}")
            
            return "\n\n---KNOWLEDGE BASE ENTRY---\n\n".join(kb_context_parts)
        
        async def get_online_context():
            # Get online context - REDUCED for performance
            docs = await self.search_engine_service.search(self.car_id, user_message, top_k=10)  # Reduced search scope
            online_docs = [d for d in docs if d.metadata.get("source") == "car_guide_link"]
            
            online_context = []
            for doc in online_docs[:2]:  # Reduced from 3 to 2 online sources
                url = doc.metadata.get("url", "Unknown URL")
                content = doc.page_content[:400]  # Reduced from 800 to 400 chars
                online_context.append(f"[Online Source: {url}]\n{content}")
            return online_context
        
        # PERFORMANCE OPTIMIZATION: Run all context retrieval in parallel
        import asyncio
        manual_context, kb_context, online_context = await asyncio.gather(
            get_manual_context(),
            get_knowledge_context(), 
            get_online_context(),
            return_exceptions=True
        )
        
        # Handle any exceptions from parallel execution
        if isinstance(manual_context, Exception):
            await self.logger.error(f"Manual context error: {manual_context}")
            manual_context = ""
        if isinstance(kb_context, Exception):
            await self.logger.error(f"KB context error: {kb_context}")
            kb_context = ""
        if isinstance(online_context, Exception):
            await self.logger.error(f"Online context error: {online_context}")
            online_context = []
        
        # Log context sizes for performance monitoring
        manual_size = len(manual_context) if manual_context else 0
        kb_size = len(kb_context) if kb_context else 0  
        online_size = len("\n".join(online_context)) if online_context else 0
        total_context_size = manual_size + kb_size + online_size
        
        await self.logger.info(f"OPTIMIZED Context sizes - Manual: {manual_size}, KB: {kb_size}, Online: {online_size}, Total: {total_context_size} chars (Target <5000)")
        
        return {
            "manual_context": manual_context,
            "kb_context": kb_context,
            "online_context": online_context
        }

    def summarize_tree(self) -> str:
        """
        Summarize the diagnosis tree for LLM input with enhanced diagnosis-focused formatting.
        """
        if self.diagnosis_tree is None:
            return "No diagnosis tree available."
            
        # Debug logging to understand tree state
        children_count = len(self.diagnosis_tree.children)
        
        if children_count == 0:
            return "Diagnosis tree is empty - no symptoms have been extracted yet."
        
        def format_tree_for_diagnosis(node, depth=0):
            """Format tree in a diagnosis-friendly way with priority and context"""
            indent = "  " * depth
            
            # Format the node information
            issue_name = getattr(node, "issue_name", "Unknown")
            likelihood = getattr(node, "likelyhood", 0.0)
            data = getattr(node, "data", None)
            
            # Build node description
            node_desc = f"{indent}- {issue_name} (Likelihood: {likelihood:.0%})"
            
            # Add additional context from data if available
            if data and isinstance(data, dict):
                context_parts = []
                if 'category' in data:
                    context_parts.append(f"Category: {data['category']}")
                if 'severity' in data:
                    context_parts.append(f"Severity: {data['severity']}")
                if 'urgency' in data:
                    context_parts.append(f"Urgency: {data['urgency']}")
                if 'issue_type' in data:
                    context_parts.append(f"Type: {data['issue_type']}")
                if 'component' in data:
                    context_parts.append(f"Component: {data['component']}")
                
                if context_parts:
                    node_desc += f" [{', '.join(context_parts)}]"
            
            # Add children recursively
            children = getattr(node, "children", [])
            if children:
                # Sort children by likelihood for better diagnosis priority
                sorted_children = sorted(children, key=lambda x: getattr(x, "likelyhood", 0.0), reverse=True)
                for child in sorted_children:
                    node_desc += "\n" + format_tree_for_diagnosis(child, depth + 1)
            
            return node_desc
        
        # Create diagnosis-focused summary
        tree_formatted = format_tree_for_diagnosis(self.diagnosis_tree)
        
        # Add summary statistics for diagnosis context
        def count_symptoms_by_likelihood(node):
            """Count symptoms by likelihood ranges"""
            high_likelihood = 0  # >70%
            medium_likelihood = 0  # 30-70%
            low_likelihood = 0  # <30%
            
            def traverse_and_count(n):
                nonlocal high_likelihood, medium_likelihood, low_likelihood
                likelihood = getattr(n, "likelyhood", 0.0)
                if n != self.diagnosis_tree:  # Don't count root
                    if likelihood > 0.7:
                        high_likelihood += 1
                    elif likelihood > 0.3:
                        medium_likelihood += 1
                    else:
                        low_likelihood += 1
                
                for child in getattr(n, "children", []):
                    traverse_and_count(child)
            
            traverse_and_count(node)
            return high_likelihood, medium_likelihood, low_likelihood
        
        high, medium, low = count_symptoms_by_likelihood(self.diagnosis_tree)
        
        diagnosis_summary = f"""DIAGNOSIS TREE ANALYSIS:
{tree_formatted}

SYMPTOM PRIORITY SUMMARY:
- High Priority (>70% likelihood): {high} symptoms
- Medium Priority (30-70% likelihood): {medium} symptoms  
- Low Priority (<30% likelihood): {low} symptoms
- Total symptoms identified: {high + medium + low}

DIAGNOSIS GUIDANCE:
Focus your diagnosis on the highest likelihood symptoms first, then use supporting evidence from medium and low priority symptoms to confirm or refine the diagnosis. Consider the hierarchical relationships shown above - child symptoms often provide specific details about their parent categories."""
        
        return diagnosis_summary

    async def diagnose(self, user_messages: List[str], session_id: Optional[str] = None) -> dict:
        stage_msg = {"type": "stage", "stage": "Starting diagnosis"}
        await self.logger.info(f"Broadcasting stage: {stage_msg['stage']}")
        await self.broadcast_stage(json.dumps(stage_msg))
        """
        Main entry: generate diagnosis using tree, last 5 user messages, and multi-source context.
        Session-aware: integrates with session context for focused diagnosis.
        Improved error handling.
        Args:
            user_messages (List[str]): List of user messages (use only the last 5).
            session_id (Optional[str]): Session identifier for context management.
        """
        await self._ensure_car_info()
        try:
            # Use only the last 5 messages
            last_messages = user_messages[-5:] if isinstance(user_messages, list) else [user_messages]
            user_message_concat = "\n".join(last_messages)
            
            # SESSION CONTEXT INTEGRATION
            session_context_str = ""
            if session_id:
                session_context = self.session_context_manager.get_original_context(session_id)
                if session_context:
                    session_context_str = self.session_context_manager.get_context_reminder(session_id)
                    await self.logger.info(f"Session {session_id}: Using context for diagnosis - {session_context.issue_category} issue")
                else:
                    await self.logger.info(f"Session {session_id}: No existing context found for diagnosis")
            
            context = await self.retrieve_context(user_message_concat)
            stage_msg = {"type": "stage", "stage": "Context retrieved"}
            await self.logger.info(f"Broadcasting stage: {stage_msg['stage']}")
            await self.broadcast_stage(json.dumps(stage_msg))
            tree_summary = self.summarize_tree()
            prompt_vars = {
                "user_message": user_message_concat,
                "session_context": session_context_str,
                "tree_summary": tree_summary,
                "manual_context": context["manual_context"],
                "kb_context": context["kb_context"],
                "online_context": "\n".join(context["online_context"]),
                "car_make": self.car_make or "",
                "car_model": self.car_model or "",
                "car_year": self.car_year or ""
            }
            stage_msg = {"type": "stage", "stage": "Invoking LLM for diagnosis"}
            await self.logger.info(f"Broadcasting stage: {stage_msg['stage']}")
            await self.broadcast_stage(json.dumps(stage_msg))
            llm = self.llm_service.get_llm()
            prompt = self.prompt.format(**prompt_vars)
            response = await llm.ainvoke(prompt) if hasattr(llm, "ainvoke") else llm.invoke(prompt)
            # Sanitize output
            if isinstance(response, str):
                response = self._sanitize_output(response)
                # Remove markdown/code block wrappers if present
                response = response.strip()
                if response.startswith('```json'):
                    response = response[len('```json'):].strip()
                if response.startswith('```'):
                    response = response[len('```'):].strip()
                if response.endswith('```'):
                    response = response[:-3].strip()
            elif isinstance(response, dict):
                response = json.loads(self._sanitize_output(json.dumps(response)))
            stage_msg = {"type": "stage", "stage": "LLM response received"}
            await self.logger.info(f"Broadcasting stage: {stage_msg['stage']}")
            await self.broadcast_stage(json.dumps(stage_msg))
            # --- Symptom extraction trigger logic ---
            need_symptom_extraction = False
            if isinstance(response, str) and ("need more symptom" in response.lower() or "provide more symptoms" in response.lower()):
                need_symptom_extraction = True
            # Try to parse the response as JSON and extract the step_by_step_guide and other fields
            parsed_response = None
            step_by_step_guide = None
            try:
                # Handle AIMessage objects from LangChain
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = response
                
                # Handle JSON string or object
                parsed_response = json.loads(response_text) if isinstance(response_text, str) else response_text
                step_by_step_guide = parsed_response.get("step_by_step_guide")
            except Exception as e:
                await self.logger.error(f"[diagnosis] Failed to parse LLM response as JSON: {e}\nRaw response: {response}")
                parsed_response = None
            # Ensure all expected fields are present for downstream use
            default_response = {
                "diagnosis_summary": "Could not parse response.",
                "supporting_evidence": [],
                "diy_repair_procedures": [],
                "alternative_diagnoses": [],
                "diagnostic_procedures": [],
                "quality_assurance": [],
                "maintenance_schedule": {},
                "professional_consultation_indicators": [],
                "confidence": "Low: Could not parse response."
            }
            if not parsed_response or not isinstance(parsed_response, dict):
                parsed_response = default_response
                step_by_step_guide = []
            else:
                # Fill in any missing fields with defaults
                for k, v in default_response.items():
                    if k not in parsed_response:
                        parsed_response[k] = v
                # step_by_step_guide for legacy downstream use
                step_by_step_guide = parsed_response.get("step_by_step_guide", [])
            return {
                "diagnosis": parsed_response,
                "success": True,
                "need_symptom_extraction": need_symptom_extraction,
                "step_by_step_guide": step_by_step_guide
            }
        except Exception as e:
            tb = traceback.format_exc()
            error_stage = f"Error occurred - {type(e).__name__}"
            stage_msg = {"type": "stage", "stage": error_stage}
            await self.logger.info(f"Broadcasting error stage: {stage_msg['stage']}")
            await self.broadcast_stage(json.dumps(stage_msg))
            await self.logger.error(f"DiagnosisAgent error: {e}\n{tb}")
            user_friendly = "An internal error occurred while generating the diagnosis. Please try again later."
            return {
                "diagnosis": None,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "user_message": user_friendly
            }

    @monitor_and_handle("DiagnosisAgent")
    async def process(self, user_message: str, websocket=None, session_id=None):
        """
        Accepts the user messages (list), runs the diagnosis, and returns the result and success status.
        Session-aware: integrates with session context management.
        Uses only the last 5 user messages.
        """
        await self._log_entry("process", message_length=len(str(user_message)), session_id=session_id)
        
        await self._ensure_car_info()
        try:
            # Stage 1: Diagnosis started
            if websocket:
                await self.logger.info(f"Sending diagnosis start stage via WebSocket - session_id={session_id}")
                await self.send_ws_stage(websocket, "Diagnosis started", MessageSource.DIAGNOSTIC_AGENT, session_id=session_id)
            
            # Stage 2: Analyzing diagnosis tree
            if websocket:
                tree_summary = None
                if self.diagnosis_tree:
                    tree_summary = {
                        "root_issue": self.diagnosis_tree.issue_name,
                        "total_symptoms": len(self.diagnosis_tree.children),
                        "main_symptoms": [
                            {
                                "name": child.issue_name,
                                "likelihood": round(child.likelyhood * 100, 1),
                                "type": child.data.get("issue_type") if child.data else "Unknown"
                            }
                            for child in sorted(self.diagnosis_tree.children, key=lambda x: x.likelyhood, reverse=True)[:3]
                        ] if self.diagnosis_tree.children else []
                    }
                
                await self.send_ws_stage(
                    websocket, 
                    "Analyzing diagnosis tree and symptoms", 
                    MessageSource.DIAGNOSTIC_AGENT, 
                    session_id=session_id,
                    details={"tree_analysis": tree_summary}
                )
            
            # SESSION CONTEXT INTEGRATION
            if session_id:
                # Check if message is relevant to session context
                session_context = self.session_context_manager.get_original_context(session_id)
                if session_context:
                    is_relevant = await self.session_context_manager.is_message_relevant(user_message, session_id)
                    if not is_relevant:
                        await self.logger.info(f"Session {session_id}: Message not relevant to original issue - {session_context.primary_issue}")
                        # Return focused response instead of general diagnosis
                        return {
                            "success": True,
                            "result": {
                                "diagnosis_summary": f"This session is focused on your {session_context.issue_category} issue: '{session_context.primary_issue}'. Your question seems unrelated to this original issue. Would you like to start a new session for a different automotive concern, or would you like to continue with the current {session_context.issue_category} diagnosis?",
                                "session_context_analysis": {
                                    "original_issue_focus": session_context.primary_issue,
                                    "symptom_connections": "Message not connected to original symptoms",
                                    "diagnostic_focus": f"Please stay focused on the {session_context.issue_category} issue"
                                },
                                "tree_analysis": {},
                                "supporting_evidence": [],
                                "diy_repair_procedures": [],
                                "alternative_diagnoses": [],
                                "diagnostic_procedures": [],
                                "quality_assurance": [],
                                "maintenance_schedule": {},
                                "professional_consultation_indicators": [],
                                "confidence": "High - Maintaining session focus"
                            },
                            "step_by_step_guide": [],
                            "error": None,
                            "error_type": None,
                            "user_message": None
                        }
                    else:
                        # Message is relevant to session context - continue with diagnosis
                        await self.logger.info(f"Session {session_id}: Processing relevant follow-up message")
            
            # Stage 3: Gathering car information and context
            if websocket:
                await self.send_ws_stage(websocket, "Gathering car information and manual context", MessageSource.DIAGNOSTIC_AGENT, session_id=session_id)
            
            # Fetch the full car object from the DB to check for vectorization status
            car = await self.car_crud.get_car_by_id(self.car_id) if self.car_crud and self.car_id else None
            if car:
                # Check new vectorization fields instead of old 'vector' field
                is_vectorized = car.get('is_vectorized', False)
                chunk_count = car.get('vector_chunk_count', 0)
                if is_vectorized and chunk_count > 0:
                    await self.logger.info(f"Vector data present for car {self.car_id}: vectorized={is_vectorized}, chunks={chunk_count}")
                else:
                    await self.logger.info(f"No vector data present for car {self.car_id} (vectorized={is_vectorized}, chunks={chunk_count})")
            
            # Stage 4: Running diagnosis analysis
            if websocket:
                await self.send_ws_stage(websocket, "Running comprehensive diagnosis analysis", MessageSource.DIAGNOSTIC_AGENT, session_id=session_id)
            
            result = await self.diagnose(user_message, session_id=session_id)
            
            # Stage 5: Processing diagnosis results
            if websocket:
                diagnosis_summary = None
                if result.get("success", False) and result.get("diagnosis"):
                    diagnosis_data = result.get("diagnosis", {})
                    diagnosis_summary = {
                        "primary_diagnosis": diagnosis_data.get("diagnosis_summary", "Unknown"),
                        "confidence": diagnosis_data.get("confidence", "Unknown"),
                        "repair_procedures_count": len(diagnosis_data.get("diy_repair_procedures", [])),
                        "alternatives_count": len(diagnosis_data.get("alternative_diagnoses", [])),
                        "requires_professional": len(diagnosis_data.get("professional_consultation_indicators", [])) > 0
                    }
                
                await self.send_ws_stage(
                    websocket, 
                    "Processing diagnosis results and generating recommendations", 
                    MessageSource.DIAGNOSTIC_AGENT, 
                    session_id=session_id,
                    details={"diagnosis_summary": diagnosis_summary}
                )
            
            # Update session context with diagnosis if available
            if session_id and result.get("success", False) and result.get("diagnosis"):
                # Note: We don't update the session context from diagnosis - that's mixing responsibilities
                # The session context agent manages session focus, not diagnosis outcomes
                await self.logger.info(f"Session {session_id}: Diagnosis completed successfully")
            
            # Send tree data immediately after diagnosis completion
            if websocket and self.diagnosis_tree:
                try:
                    tree_data = {
                        "full_tree": self.diagnosis_tree.to_dict(),
                        "summary": {
                            "total_nodes": len(self.diagnosis_tree.children),
                            "root_issue": self.diagnosis_tree.issue_name,
                            "high_likelihood_symptoms": [
                                child.issue_name for child in self.diagnosis_tree.children 
                                if getattr(child, "likelyhood", 0) > 0.7
                            ]
                        }
                    }
                    
                    tree_message = {
                        "type": "tree_data",
                        "source": "diagnosis_agent",
                        "content": "Complete tree data after diagnosis",
                        "timestamp": datetime.now().isoformat() + "Z",
                        "data": {
                            "tree_data": tree_data,
                            "stage": "diagnosis_complete",
                            "diagnosis_available": result.get("success", False)
                        }
                    }
                    
                    await websocket.send_text(json.dumps(tree_message))
                    await self.logger.info(f"Sent complete tree data via WebSocket after diagnosis")
                except Exception as e:
                    await self.logger.error(f"Failed to send tree data via WebSocket: {e}")
            
            # Stage 6: Diagnosis complete
            if websocket:
                await self.logger.info(f"Sending diagnosis completion result via WebSocket - session_id={session_id}, result_success={result.get('success', False)}")
                
                # Include final tree data in completion
                final_tree_data = None
                if self.diagnosis_tree:
                    final_tree_data = {
                        "total_nodes": len(self.diagnosis_tree.children),
                        "root_issue": self.diagnosis_tree.issue_name,
                        "children": [
                            {
                                "issue_name": child.issue_name,
                                "likelihood": round(child.likelyhood * 100, 1),
                                "type": child.data.get("issue_type") if child.data else "Unknown",
                                "category": child.data.get("issue_category") if child.data else "Unknown",
                                "description": child.data.get("description") if child.data else None,
                                "severity": child.data.get("severity") if child.data else "Unknown"
                            }
                            for child in self.diagnosis_tree.children
                        ]
                    }
                
                result_details = {
                    "success": result.get("success", False),
                    "has_diagnosis": "diagnosis" in result and result["diagnosis"] is not None,
                    "has_step_guide": "step_by_step_guide" in result and len(result.get("step_by_step_guide", [])) > 0,
                    "diagnosis_data": diagnosis_summary if 'diagnosis_summary' in locals() else None,
                    "final_tree_data": final_tree_data  # Include complete tree data
                }
                await self.send_ws_result(websocket, "Diagnosis complete", MessageSource.DIAGNOSTIC_AGENT, session_id=session_id, details=result_details)
            
            process_result = {
                "success": result.get("success", False),
                "result": result.get("diagnosis"),
                "step_by_step_guide": result.get("step_by_step_guide"),
                "error": result.get("error") if not result.get("success", False) else None,
                "error_type": result.get("error_type") if not result.get("success", False) else None,
                "user_message": result.get("user_message") if not result.get("success", False) else None
            }
            
            await self._log_exit("process", success=process_result["success"], has_session_context=session_id is not None)
            return process_result
            
        except Exception as e:
            tb = traceback.format_exc()
            await self.logger.error(f"Process error: {e}\n{tb}")
            user_friendly = "An unexpected error occurred while processing your request. Please try again later."
            if websocket:
                await self.logger.info(f"Sending diagnosis error via WebSocket - session_id={session_id}, error_type={type(e).__name__}")
                await self.send_ws_error(websocket, user_friendly, MessageSource.DIAGNOSTIC_AGENT, session_id=session_id, details={"error": str(e)})
            
            await self._log_exit("process", success=False, error=str(e))
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "error_type": type(e).__name__,
                "user_message": user_friendly
            }

    async def generate_diagnosis(self, user_message: str, tree_summary: str, manual_context: str, kb_context: str = "", online_context: str = "") -> str:
        stage_msg = {"type": "stage", "stage": "Generating diagnosis"}
        await self.logger.info(f"Broadcasting stage: {stage_msg['stage']}")
        await self.broadcast_stage(json.dumps(stage_msg))
        """
        Generate diagnosis using the LLM service.
        """
        await self._ensure_car_info()
        prompt = self.prompt.format(user_message=user_message, tree_summary=tree_summary, manual_context=manual_context, kb_context=kb_context, online_context=online_context)
        response = await self.llm_service.generate_response(prompt)
        response = self._sanitize_output(response)
        return response

    def get_langchain_llm(self):
        """
        For advanced LangChain integrations (e.g., chains), use this accessor.
        """
        return self.llm_service.get_llm()

    def close(self) -> None:
        """
        Optional cleanup method for the agent.
        """
        pass
