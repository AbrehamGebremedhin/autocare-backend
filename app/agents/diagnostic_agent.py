from typing import Any, Dict, List, Optional
from app.utils.logger import Logger
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
from app.utils.websocket import manager  # WebSocket manager for broadcasting stages
import json
from app.services.search_engine_service import SearchEngineService

class DiagnosisAgent:
    """
    Agentic RAG for generating a diagnosis using the diagnosis tree, user message, and multi-source context.
    """
    def __init__(self, car_id: str, diagnosis_tree: DiagnosisTreeNode, search_engine_service: Optional[SearchEngineService] = None, **kwargs):
        self.car_id = car_id
        self.diagnosis_tree = diagnosis_tree
        self.logger = Logger("DiagnosisAgent")
        self.llm_service = LLMService()
        self.car_crud = CarCRUD()
        self.embedding_service = EmbeddingService()
        self.scraper_service = ScraperService(headless=True)
        self.search_engine_service = search_engine_service or SearchEngineService()
        self.prompt = PromptTemplate.from_template(
            """
            You are an expert automotive diagnostician. Your task is to generate a comprehensive, evidence-based diagnosis for the user's vehicle problem.

            CONTEXT SOURCES:
            - Diagnosis tree: Structured symptom and issue data (see below)
            - Owner's manual/context: Official documentation and technical details
            - Knowledge base: Trusted reference material and prior cases
            - Online context: Recent or rare issues from the web and car-specific guides

            INSTRUCTIONS:
            1. Carefully analyze the user's message and the diagnosis tree to identify likely root causes.
            2. Attribute supporting evidence to each context source (owner's manual, knowledge base, online, tree).
            3. Clearly explain your reasoning, referencing specific evidence from each source.
            4. If information is missing or ambiguous, state what additional details are needed.
            5. Provide actionable, step-by-step recommendations for the user.

            INPUT:
            - User message: {user_message}
            - Diagnosis tree: {tree_summary}
            - Owner's manual/context: {manual_context}
            - Knowledge base: {kb_context}
            - Online context: {online_context}

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
                "missing_information": ["..."],
                "next_steps": ["..."],
                "confidence": "High/Medium/Low"
            }}
            """
        )

    async def retrieve_context(self, user_message: str) -> Dict[str, Any]:
        await manager.broadcast(json.dumps({"type": "stage", "stage": "Retrieving context"}))
        """
        Retrieve owner's manual, knowledge base, and online context relevant to the user message and tree using SearchEngineService.
        Also use valid links in car_guide_links for online context.
        """
        # Get car info for owner manual search
        car = await self.car_crud.get_car_by_id(self.car_id)
        make = car.get("make") if car else None
        model = car.get("model") if car else None
        year = car.get("year") if car else None
        # Use vector_search for knowledge base and owner manual context
        kb_chunks = await self.search_engine_service.vector_search(user_message, query_type="validation")
        manual_chunks = await self.search_engine_service.vector_search(user_message, query_type="generation", make=make, model=model, year=year)
        # Use web_search for online context (top 3 URLs, then scrape)
        web_links = await self.search_engine_service.web_search(user_message, num_results=3)
        # Also use car_guide_links from car row
        car_guide_links = car.get("car_guide_links") if car else []
        valid_prefixes = ("http://", "https://", "file://", "raw:")
        valid_guide_links = [link for link in car_guide_links if isinstance(link, str) and link.startswith(valid_prefixes)]
        # Combine and deduplicate links
        all_links = list(dict.fromkeys(web_links + valid_guide_links))
        online_context = []
        if all_links:
            try:
                scraped = await self.scraper_service.perform_action(all_links, limit=len(all_links))
                online_context = [item.get("text", "") for item in scraped if item.get("text")]
            except Exception:
                online_context = []
        # Compose context strings
        manual_context = "\n".join([c["content"] for c in manual_chunks[:3]]) if manual_chunks else ""
        kb_context = "\n".join([c["content"] for c in kb_chunks[:3]]) if kb_chunks else ""
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

    async def diagnose(self, user_message: str) -> dict:
        await manager.broadcast(json.dumps({"type": "stage", "stage": "Starting diagnosis"}))
        """
        Main entry: generate diagnosis using tree, user message, and multi-source context.
        Improved error handling.
        """
        try:
            context = await self.retrieve_context(user_message)
            await manager.broadcast(json.dumps({"type": "stage", "stage": "Context retrieved"}))
            tree_summary = self.summarize_tree()
            prompt_vars = {
                "user_message": user_message,
                "tree_summary": tree_summary,
                "manual_context": context["manual_context"],
                "kb_context": context["kb_context"],
                "online_context": "\n".join(context["online_context"])
            }
            await manager.broadcast(json.dumps({"type": "stage", "stage": "Invoking LLM for diagnosis"}))
            llm = self.llm_service.get_llm()
            prompt = self.prompt.format(**prompt_vars)
            response = await llm.ainvoke(prompt) if hasattr(llm, "ainvoke") else llm.invoke(prompt)
            await manager.broadcast(json.dumps({"type": "stage", "stage": "LLM response received"}))
            # --- Symptom extraction trigger logic ---
            # If the LLM response or tree indicates more symptoms are needed, set the flag
            need_symptom_extraction = False
            if isinstance(response, str) and ("need more symptom" in response.lower() or "provide more symptoms" in response.lower()):
                need_symptom_extraction = True
            # You can add more advanced logic here based on the tree or structured response
            return {
                "diagnosis": response,
                "success": True,
                "need_symptom_extraction": need_symptom_extraction
            }
        except Exception as e:
            tb = traceback.format_exc()
            await manager.broadcast(json.dumps({"type": "stage", "stage": f"Error occurred - {type(e).__name__}"}))
            await self.logger.error(f"DiagnosisAgent error: {e}\n{tb}")
            user_friendly = "An internal error occurred while generating the diagnosis. Please try again later."
            return {
                "diagnosis": None,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "user_message": user_friendly
            }

    async def process(self, user_message: str) -> Dict[str, Any]:
        """
        Accepts the user message, runs the diagnosis, and returns the result and success status.
        Improved error handling.
        """
        try:
            result = await self.diagnose(user_message)
            return {
                "success": result.get("success", False),
                "result": result.get("diagnosis"),
                "error": result.get("error") if not result.get("success", False) else None,
                "error_type": result.get("error_type") if not result.get("success", False) else None,
                "user_message": result.get("user_message") if not result.get("success", False) else None
            }
        except Exception as e:
            tb = traceback.format_exc()
            await self.logger.error(f"Process error: {e}\n{tb}")
            user_friendly = "An unexpected error occurred while processing your request. Please try again later."
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "error_type": type(e).__name__,
                "user_message": user_friendly
            }

    async def generate_diagnosis(self, user_message: str, tree_summary: str, manual_context: str, kb_context: str = "", online_context: str = "") -> str:
        """
        Generate diagnosis using the LLM service.
        """
        prompt = self.prompt.format(user_message=user_message, tree_summary=tree_summary, manual_context=manual_context, kb_context=kb_context, online_context=online_context)
        response = await self.llm_service.generate_response(prompt)
        return response

    def get_langchain_llm(self):
        """
        For advanced LangChain integrations (e.g., chains), use this accessor.
        """
        return self.llm_service.get_llm()
