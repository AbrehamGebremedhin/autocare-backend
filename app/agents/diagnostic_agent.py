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
    def __init__(self, car_id: str, diagnosis_tree: DiagnosisTreeNode, car_make: str = None, car_model: str = None, car_year: str = None, search_engine_service: Optional[SearchEngineService] = None, websocket_manager: IWebSocketManager = None, **kwargs):
        super().__init__(car_crud=CarCRUD(), car_id=car_id, car_make=car_make, car_model=car_model, car_year=car_year, logger_name="DiagnosisAgent", websocket_manager=websocket_manager)
        self.diagnosis_tree = diagnosis_tree
        self.llm_service = LLMService()
        self.embedding_service = EmbeddingService()
        self.scraper_service = ScraperService(headless=True)
        self.search_engine_service = search_engine_service or SearchEngineService()
        self.car_make = car_make
        self.car_model = car_model
        self.car_year = car_year
        self.prompt = PromptTemplate.from_template(
            """
            You are an expert automotive diagnostician. Your goal is to help the user diagnose and, if possible, resolve the issue themselves. Provide clear, step-by-step instructions for safe DIY troubleshooting and minor repairs. Only recommend seeing a mechanic if the issue is dangerous, requires specialized tools, or cannot be safely addressed by a typical car owner.

            Always include safety warnings before any potentially hazardous steps. Use simple language and explain technical terms. Do NOT recommend visiting a mechanic unless absolutely necessary. Try to empower the user to understand and address the problem first.

            IMPORTANT:
            - You have access to the car's make, model, and year: Make: {car_make}, Model: {car_model}, Year: {car_year}.
            - Do NOT ask the user for car make, model, or year; you already have this information.
            - Do NOT tell the user to refer to the owner's manual or say "check your owner's manual". Instead, if the information is present in the provided owner's manual/context, extract and use it directly in your answer. If the information is not present in the provided context, state that it is not available, but do NOT refer the user to the manual.

            CONTEXT SOURCES:
            - Diagnosis tree: Structured symptom and issue data (see below)
            - Owner's manual/context: Official documentation and technical details (provided below as context)
            - Knowledge base: Trusted reference material and prior cases
            - Online context: Recent or rare issues from the web and car-specific guides

            INSTRUCTIONS:
            1. Carefully analyze the last 5 user messages (provided below, most recent last) and the diagnosis tree to identify all plausible root causes, including common, rare, and edge-case scenarios.
            2. For each scenario, attribute supporting evidence to each context source (owner's manual, knowledge base, online, tree), and cite specific knowledge base entries or document snippets when possible.
            3. Clearly explain your reasoning, referencing specific evidence from each source.
            4. If information is missing or ambiguous, explicitly state what additional details are needed and ask the user for this information in a clear, friendly way.
            5. Provide a detailed, step-by-step troubleshooting and repair guide for the user, including safety precautions, required tools, and what to check at each step.
            6. If multiple possible causes exist, provide a comparative table or list of possible diagnoses, with distinguishing features, pros/cons, and what to check to rule each in or out.
            7. If the problem is urgent or could cause further damage, highlight this and advise the user accordingly.
            8. Always include actionable next steps, and if the user should consult a professional mechanic, say so.
            9. Consider the conversation context and progression from the last 5 user messages.
            10. If the diagnosis tree exists, mention other possible causes from the tree that may be relevant as "Other Possible Causes" in your output, especially if they have not been ruled out by the current symptoms.
            11. Always include a section for "Uncommon but Important Scenarios" if any rare or edge-case issues are plausible.

            OUTPUT (JSON):
            {{
                "diagnosis_summary": "Main diagnosis and reasoning, with source attributions",
                "supporting_evidence": [
                    {{"source": "diagnosis_tree", "evidence": "..."}},
                    {{"source": "owner_manual", "evidence": "..."}},
                    {{"source": "knowledge_base", "evidence": "..."}},
                    {{"source": "online", "evidence": "..."}}
                ],
                "recommendations": ["Step 1...", "Step 2...", "..."],
                "step_by_step_guide": ["Step 1: ...", "Step 2: ...", "..."],
                "alternative_diagnoses": [
                    {{
                        "name": "...",
                        "likelihood": "High/Medium/Low",
                        "distinguishing_features": ["..."],
                        "evidence_from_knowledge_base": ["..."],
                        "actionable_steps": ["..."],
                        "notes": "..."
                    }}
                ],
                "uncommon_but_important_scenarios": [
                    {{
                        "name": "...",
                        "description": "...",
                        "likelihood": "Low/Medium",
                        "evidence": "..."
                    }}
                ],
                "missing_information": ["..."],
                "next_steps": ["..."],
                "other_possible_causes": ["..."],
                "confidence": "High/Medium/Low"
            }}
            - The 'step_by_step_guide' field must be a clear, numbered, step-by-step guide for the user to follow to fix the problem, separate from general recommendations.
            - The 'alternative_diagnoses' field must include all plausible alternative scenarios, with distinguishing features and evidence.
            - The 'uncommon_but_important_scenarios' field must include rare or edge-case issues that should not be overlooked.
            - If the problem cannot be fixed by the user, explain why and what to do instead.

            INPUT:
            - Car make: {car_make}
            - Car model: {car_model}
            - Car year: {car_year}
            - User messages (last 5, most recent last): {user_message}
            - Diagnosis tree: {tree_summary}
            - Owner's manual/context: {manual_context}
            - Knowledge base: {kb_context}
            - Online context: {online_context}

            Ensure the JSON is valid and well-formed. Do NOT include any additional text, markdown, or explanations outside the JSON object.
            Return ONLY a valid JSON object as specified above. Do NOT include any extra text, markdown, explanations outside the JSON or surrounded by backticks.
            """
        )

    async def retrieve_context(self, user_message: str) -> Dict[str, Any]:
        await self.broadcast_stage(json.dumps({"type": "stage", "stage": "Retrieving context"}))
        """
        Retrieve owner's manual, knowledge base, and online context relevant to the user message and tree using SearchEngineService.
        Uses the new search engine unified interface.
        """
        await self._ensure_car_info()
        # Fetch the full car object from the DB to get the vector
        car = await self.car_crud.get_car_by_id(self.car_id) if self.car_crud and self.car_id else None
        manual_context = ""
        if car and car.get("vector"):
            manual_context = str(car["vector"])
        # Use the new search engine to get a broader set of relevant context as LangChain Documents (except manual)
        # Fetch more documents for richer context (e.g., top_k=40)
        docs = await self.search_engine_service.search(self.car_id, user_message, top_k=40)
        # Cluster or summarize the most relevant knowledge base entries to avoid token overload
        kb_docs = [d.page_content for d in docs if d.metadata.get("source") == "ground_knowledge"]
        # Summarize or select the most diverse 10-15 entries
        kb_context = "\n---\n".join(kb_docs[:15])
        # Optionally, you could implement clustering here for more diversity
        online_context = [d.page_content for d in docs if d.metadata.get("source") == "car_guide_link"]
        return {
            "manual_context": manual_context,
            "kb_context": kb_context,
            "online_context": online_context
        }

    def summarize_tree(self) -> str:
        """
        Summarize the diagnosis tree for LLM input.
        """
        def node_to_dict(node):
            return {
                "name": getattr(node, "name", None),
                "likelihood": getattr(node, "likelihood", None),
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
                "recommendations": [],
                "step_by_step_guide": [],
                "alternative_diagnoses": [],
                "uncommon_but_important_scenarios": [],
                "missing_information": [],
                "next_steps": [],
                "other_possible_causes": [],
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
