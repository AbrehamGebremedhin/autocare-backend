from typing import Any, Dict, List, Optional
from app.utils.logger import Logger
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.services.scraper_service import ScraperService
from app.CRUD.car_crud import CarCRUD
from app.utils.diagnosis_tree import DiagnosisTreeNode
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_ollama import OllamaLLM
import asyncio
import numpy as np
import traceback

class DiagnosisAgent:
    """
    Agentic RAG for generating a diagnosis using the diagnosis tree, user message, and multi-source context.
    """
    def __init__(self, car_id: str, diagnosis_tree: DiagnosisTreeNode, **kwargs):
        self.car_id = car_id
        self.diagnosis_tree = diagnosis_tree
        self.logger = Logger("DiagnosisAgent")
        self.llm = OllamaLLM(model="gemma3:12b")
        self.car_crud = CarCRUD()
        self.embedding_service = EmbeddingService()
        self.scraper_service = ScraperService(headless=True)
        self.prompt = PromptTemplate.from_template(
            """
            You are an expert automotive diagnostician. Using the provided diagnosis tree (structured symptom and issue data), the user's message, and all available context (owner's manual, knowledge base, and online sources), generate a comprehensive diagnosis for the user's vehicle.

            - Reference the diagnosis tree to identify the most likely root causes.
            - Use the owner's manual and knowledge base for technical accuracy.
            - Use online sources for up-to-date or rare issues.
            - Clearly explain the reasoning, referencing evidence from each source.
            - Provide actionable recommendations and next steps.

            Input:
            - User message: {user_message}
            - Diagnosis tree: {tree_summary}
            - Owner's manual/context: {manual_context}
            - Knowledge base: {kb_context}
            - Online context: {online_context}

            Output a structured JSON object with:
            - diagnosis_summary: Main diagnosis and reasoning
            - supporting_evidence: List of evidence from each source
            - recommendations: List of next steps for the user
            """
        )

    async def retrieve_context(self, user_message: str) -> Dict[str, Any]:
        """
        Retrieve owner's manual, knowledge base, and online context relevant to the user message and tree.
        """
        car = await self.car_crud.get_car_by_id(self.car_id)
        manual_context = car.get("vector", "") if car else ""
        guide_links: List[str] = car.get("car_guide_links") or [] if car else []
        valid_prefixes = ("http://", "https://", "file://", "raw:")
        guide_links = [link for link in guide_links if isinstance(link, str) and link.startswith(valid_prefixes)]
        # Use embeddings to select top guide links
        input_vec, link_vecs = await asyncio.gather(
            self.embedding_service.embed_text(user_message),
            self.embedding_service.embed_texts(guide_links) if guide_links else asyncio.sleep(0, result=[])
        )
        def cosine_sim(a, b):
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        scored_links = [
            (link, cosine_sim(input_vec, link_vec))
            for link, link_vec in zip(guide_links, link_vecs)
        ] if guide_links else []
        scored_links.sort(key=lambda x: x[1], reverse=True)
        top_links = [link for link, score in scored_links[:3] if score > 0.3]
        # Scrape online context
        online_context = []
        if top_links:
            try:
                scraped = await self.scraper_service.perform_action(top_links, limit=len(top_links))
                online_context = [item.get("text", "") for item in scraped if item.get("text")]
            except Exception:
                online_context = []
        # Knowledge base context (could be expanded to vector search)
        kb_context = manual_context  # For now, use manual as KB
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

    async def diagnose(self, user_message: str) -> Dict[str, Any]:
        """
        Main entry: generate diagnosis using tree, user message, and multi-source context.
        Improved error handling.
        """
        try:
            context = await self.retrieve_context(user_message)
            tree_summary = self.summarize_tree()
            prompt_vars = {
                "user_message": user_message,
                "tree_summary": tree_summary,
                "manual_context": context["manual_context"],
                "kb_context": context["kb_context"],
                "online_context": "\n".join(context["online_context"])
            }
            chain = self.llm | self.prompt
            response = await chain.ainvoke(prompt_vars) if hasattr(chain, "ainvoke") else chain.invoke(prompt_vars)
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
