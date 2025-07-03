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
from app.services.search_engine_service import SearchEngineService
from app.agents.base_agent import BaseAgent
from app.utils.message_types import MessageSource
from app.utils.monitoring import monitor_and_handle

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
        self.car_make = car_make
        self.car_model = car_model
        self.car_year = car_year
        self.prompt = PromptTemplate.from_template(
            """
            You are an expert automotive diagnostician with access to an extensive knowledge base of 38,936 automotive documents. Your primary goal is to EMPOWER users to diagnose and fix automotive issues themselves by providing comprehensive, detailed technical guidance. Most users want to perform their own repairs, so provide in-depth, step-by-step instructions that enable successful DIY repairs.

            Focus on DIY empowerment - assume the user wants to fix the issue themselves and provide the knowledge and confidence to do so safely. Only recommend professional help when absolutely necessary for safety or when specialized equipment is required.

            CORE PRINCIPLES:
            - Provide detailed technical explanations that build user understanding
            - Include multiple diagnostic approaches so users can choose what works for them
            - Give specific part numbers, specifications, and technical details when possible
            - Explain the WHY behind each step so users understand the reasoning
            - Provide troubleshooting alternatives if the first approach doesn't work
            - Include common pitfalls and how to avoid them
            - Emphasize safety but don't overstate risks for standard procedures

            IMPORTANT:
            - You have access to the car's make, model, and year: Make: {car_make}, Model: {car_model}, Year: {car_year}.
            - Do NOT ask the user for car make, model, or year; you already have this information.
            - Do NOT tell the user to refer to the owner's manual. Extract and use information directly.
            - You have access to a comprehensive automotive knowledge base with nearly 39,000 documents covering all automotive systems, repair procedures, and technical specifications.
            - ASSUME the user wants to do the repair themselves and provide detailed guidance to make that possible.

            CONTEXT SOURCES:
            - Diagnosis tree: Structured symptom and issue data
            - Owner's manual/context: Official technical specifications and procedures
            - Knowledge base: Comprehensive automotive repair and diagnostic information (38,936 documents)
            - Online context: Recent technical discussions and repair experiences

            INSTRUCTIONS FOR DIY-FOCUSED COMPREHENSIVE ANALYSIS:
            1. Analyze symptoms to identify ALL possible root causes, prioritizing those that can be fixed by a DIY enthusiast.
            2. For each scenario, provide detailed technical explanations with specific repair procedures, part specifications, and tool requirements.
            3. Include multiple diagnostic approaches - visual inspection, electrical testing, mechanical testing, etc.
            4. Provide specific torque specifications, part numbers, fluid specifications, and technical details.
            5. Explain the underlying automotive systems so users understand what they're working on.
            6. Give detailed step-by-step repair procedures with professional-level detail.
            7. Include troubleshooting steps for when things don't go as expected.
            8. Provide cost-effective alternatives and workarounds where appropriate.
            9. Explain how to verify the repair was successful and prevent recurrence.
            10. Include tips and tricks from professional mechanics.
            11. Provide detailed safety guidance specific to each procedure, not generic warnings.
            12. Include maintenance schedules and inspection points to prevent similar issues.

            OUTPUT (COMPREHENSIVE DIY-FOCUSED JSON):
            {{
                "diagnosis_summary": "Detailed technical diagnosis with repair-focused analysis and system explanations",
                "supporting_evidence": [
                    {{"source": "diagnosis_tree", "evidence": "Technical evidence", "confidence": "High/Medium/Low", "diy_relevance": "How this helps DIY diagnosis"}},
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
                        "repair_approach": "Different repair strategy for this scenario",
                        "cost_estimate": "Parts and time cost",
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
                "system_education": {{
                    "affected_system": "Primary automotive system involved",
                    "how_it_works": "Technical explanation of system operation",
                    "common_failure_modes": ["How this system typically fails"],
                    "preventive_maintenance": ["How to prevent future problems"],
                    "related_components": ["Other parts that might be affected"],
                    "upgrade_opportunities": ["Performance or reliability improvements possible"]
                }},
                "cost_breakdown": {{
                    "parts_cost": "Detailed parts cost analysis",
                    "tool_investment": "One-time tool costs if tools need to be purchased",
                    "shop_cost_comparison": "What this would cost at a shop vs DIY",
                    "cost_saving_tips": ["How to reduce costs while maintaining quality"]
                }},
                "safety_protocols": [
                    {{
                        "procedure": "Specific repair procedure",
                        "safety_equipment": ["Required safety gear"],
                        "environmental_considerations": ["Workspace requirements"],
                        "specific_hazards": ["Procedure-specific risks"],
                        "emergency_procedures": ["What to do if something goes wrong"]
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
            - User messages (last 5, most recent last): {user_message}
            - Diagnosis tree: {tree_summary}
            - Owner's manual/context: {manual_context}
            - Knowledge base (from 38,936 documents): {kb_context}
            - Online context: {online_context}

            Ensure the JSON is valid and well-formed. Provide comprehensive technical details that enable successful DIY repairs.
            Return ONLY a valid JSON object. Do NOT include any extra text, markdown, or explanations outside the JSON.
            """
        )

    async def retrieve_context(self, user_message: str) -> Dict[str, Any]:
        await self.broadcast_stage(json.dumps({"type": "stage", "stage": "Retrieving context"}))
        await self._ensure_car_info()
        # Vector search against owner manual text using user_message
        manual_chunks = await self.search_engine_service.embed_and_vector_search(
            content_path=f"car_data/{self.car_id}_manual.pdf", query=user_message, top_k=1
        )
        manual_context = manual_chunks[0]["chunk"] if manual_chunks else ""
        docs = await self.search_engine_service.search(self.car_id, user_message, top_k=80)
        
        # Separate and process knowledge base docs more comprehensively
        kb_docs = [d for d in docs if d.metadata.get("source") == "ground_knowledge"]
        
        # Create rich knowledge base context with source attribution
        kb_context_parts = []
        for i, doc in enumerate(kb_docs[:25]):  # Increased from 15 to 25 for richer context
            book_title = doc.metadata.get("book_title", "Unknown Source")
            page_num = doc.metadata.get("page_number", "N/A")
            content = doc.page_content
            kb_context_parts.append(f"[Source: {book_title}, Page: {page_num}]\n{content}")
        
        kb_context = "\n\n---KNOWLEDGE BASE ENTRY---\n\n".join(kb_context_parts)
        
        # Enhanced online context processing
        online_docs = [d for d in docs if d.metadata.get("source") == "car_guide_link"]
        online_context = []
        for doc in online_docs:
            url = doc.metadata.get("url", "Unknown URL")
            content = doc.page_content
            online_context.append(f"[Online Source: {url}]\n{content}")
            
        # Additional context searches for specific automotive areas
        additional_searches = [
            f"{user_message} symptoms causes diagnosis",
            f"{self.car_make} {self.car_model} {self.car_year} common problems",
            f"automotive troubleshooting {user_message}",
            f"repair guide {user_message}"
        ]
        
        # Perform additional targeted searches for comprehensive coverage
        additional_docs = []
        for search_query in additional_searches:
            extra_docs = await self.search_engine_service.vector_search_ground_knowledge(search_query, top_k=15)
            additional_docs.extend(extra_docs)
        
        # Add additional context from targeted searches
        if additional_docs:
            additional_context_parts = []
            seen_content = set()  # Avoid duplicates
            for doc in additional_docs:
                content = doc.get("chunk", "")
                if content and content not in seen_content:
                    book_title = doc.get("book_title", "Unknown Source")
                    page_num = doc.get("page_number", "N/A")
                    additional_context_parts.append(f"[Additional Source: {book_title}, Page: {page_num}]\n{content}")
                    seen_content.add(content)
            
            if additional_context_parts:
                kb_context += "\n\n---ADDITIONAL RELEVANT KNOWLEDGE---\n\n" + "\n\n---\n\n".join(additional_context_parts[:15])
        return {
            "manual_context": manual_context,
            "kb_context": kb_context,
            "online_context": online_context
        }

    def summarize_tree(self) -> str:
        """
        Summarize the diagnosis tree for LLM input.
        """
        if self.diagnosis_tree is None:
            return "No diagnosis tree available."
            
        def node_to_dict(node):
            return {
                "issue_name": getattr(node, "issue_name", "Unknown"),
                "likelyhood": getattr(node, "likelyhood", 0.0),
                "data": getattr(node, "data", None),
                "children": [node_to_dict(child) for child in getattr(node, "children", [])]
            }
        return str(node_to_dict(self.diagnosis_tree))

    async def diagnose(self, user_messages: List[str]) -> dict:
        await self.broadcast_stage(json.dumps({"type": "stage", "stage": "Starting diagnosis"}))
        """
        Main entry: generate diagnosis using tree, last 5 user messages, and multi-source context.
        Improved error handling.
        Args:
            user_messages (List[str]): List of user messages (use only the last 5).
        """
        await self._ensure_car_info()
        try:
            # Use only the last 5 messages
            last_messages = user_messages[-5:] if isinstance(user_messages, list) else [user_messages]
            user_message_concat = "\n".join(last_messages)
            context = await self.retrieve_context(user_message_concat)
            await self.broadcast_stage(json.dumps({"type": "stage", "stage": "Context retrieved"}))
            tree_summary = self.summarize_tree()
            prompt_vars = {
                "user_message": user_message_concat,
                "tree_summary": tree_summary,
                "manual_context": context["manual_context"],
                "kb_context": context["kb_context"],
                "online_context": "\n".join(context["online_context"]),
                "car_make": self.car_make or "",
                "car_model": self.car_model or "",
                "car_year": self.car_year or ""
            }
            await self.broadcast_stage(json.dumps({"type": "stage", "stage": "Invoking LLM for diagnosis"}))
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
            await self.broadcast_stage(json.dumps({"type": "stage", "stage": "LLM response received"}))
            # --- Symptom extraction trigger logic ---
            need_symptom_extraction = False
            if isinstance(response, str) and ("need more symptom" in response.lower() or "provide more symptoms" in response.lower()):
                need_symptom_extraction = True
            # Try to parse the response as JSON and extract the step_by_step_guide and other fields
            parsed_response = None
            step_by_step_guide = None
            try:
                parsed_response = json.loads(response) if isinstance(response, str) else response
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
                "system_education": {},
                "cost_breakdown": {},
                "safety_protocols": [],
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
            await self.broadcast_stage(json.dumps({"type": "stage", "stage": f"Error occurred - {type(e).__name__}"}))
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
        await self.broadcast_stage(json.dumps({"type": "stage", "stage": "Processing diagnosis"}))
        """
        Accepts the user messages (list), runs the diagnosis, and returns the result and success status.
        Uses only the last 5 user messages.
        """
        await self._ensure_car_info()
        try:
            if websocket:
                await self.send_ws_stage(websocket, "Diagnosis started", MessageSource.DIAGNOSTIC_AGENT, session_id=session_id)
            # Fetch the full car object from the DB to check for vector data
            car = await self.car_crud.get_car_by_id(self.car_id) if self.car_crud and self.car_id else None
            if car:
                vector_data = car.get('vector')
                if vector_data:
                    await self.logger.info(f"[DEBUG] Vector data present for car {self.car_id}: type={type(vector_data)}, length={len(vector_data) if hasattr(vector_data, '__len__') else 'N/A'}")
                else:
                    await self.logger.info(f"[DEBUG] No vector data present for car {self.car_id}")
            result = await self.diagnose(user_message)
            if websocket:
                await self.send_ws_result(websocket, "Diagnosis complete", MessageSource.DIAGNOSTIC_AGENT, session_id=session_id, details=result)
            return {
                "success": result.get("success", False),
                "result": result.get("diagnosis"),
                "step_by_step_guide": result.get("step_by_step_guide"),
                "error": result.get("error") if not result.get("success", False) else None,
                "error_type": result.get("error_type") if not result.get("success", False) else None,
                "user_message": result.get("user_message") if not result.get("success", False) else None
            }
        except Exception as e:
            tb = traceback.format_exc()
            await self.logger.error(f"Process error: {e}\n{tb}")
            user_friendly = "An unexpected error occurred while processing your request. Please try again later."
            if websocket:
                await self.send_ws_error(websocket, user_friendly, MessageSource.DIAGNOSTIC_AGENT, session_id=session_id, details={"error": str(e)})
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "error_type": type(e).__name__,
                "user_message": user_friendly
            }

    async def generate_diagnosis(self, user_message: str, tree_summary: str, manual_context: str, kb_context: str = "", online_context: str = "") -> str:
        await self.broadcast_stage(json.dumps({"type": "stage", "stage": "Generating diagnosis"}))
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
