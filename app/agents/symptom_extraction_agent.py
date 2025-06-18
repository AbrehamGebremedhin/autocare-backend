from app.CRUD.car_crud import CarCRUD
from typing import Any, Dict, List, Optional
from app.core.config import get_settings
from langchain.prompts import PromptTemplate
from app.services.llm_service import LLMService
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
import asyncio
from functools import partial
from app.services.embedding_service import EmbeddingService
from app.services.scraper_service import ScraperService
import numpy as np
import time
from app.utils.logger import Logger
from app.utils.diagnosis_tree import DiagnosisTreeNode
from app.agents.tree_manager_agent import TreeManagerAgent

class SymptomExtractorAgent():
    """
    Agent for extracting symptoms from input text.
    """

    @staticmethod
    def get_prompt_template() -> PromptTemplate:
        return PromptTemplate.from_template("""
            You are an expert automotive mechanic and diagnostic specialist with extensive experience in automotive systems and failure diagnostics. 
            Using the user's reported symptoms and any provided context (such as previous diagnostics, vehicle data, or sensor readings), along with your comprehensive automotive knowledge, identify all plausible underlying issues that could cause the described symptoms.
            Carefully analyze the symptoms and context using your diagnostic expertise and automotive knowledge base. Include:
            - Common causes that match these symptoms
            - Less likely but critical failures that should not be overlooked
            - Any issue that could contribute indirectly or as a downstream effect

            Think broadly and reason through how multiple issues may be connected. Output a complete, well-structured JSON array.
            Context:
            {context}

            For each possible issue, output a detailed JSON object with the following fields:
            - **issue_name**: A concise name for the issue (e.g., "Engine Knock")
            - **likelihood**: Your estimated likelihood (0–100) that this issue is causing the symptom
            - **issue_type**: The general type of the issue (e.g., "mechanical", "electrical", "software")
            - **issue_category**: The broad category of the issue (e.g., "engine", "transmission", "fuel system")
            - **issue_subcategory**: A specific subcategory if applicable (e.g., "ignition system", "fuel injection")
            - **issue_description**: A clear, detailed explanation of the issue and how it relates to the symptom
            - **severity**: The severity level of the issue, one of ["low", "medium", "high"]
            - **additional_info**: Any other relevant details (common causes, conditions, diagnostic tips, etc.)

            Important instructions:
            - Return **ONLY** a valid JSON array of issue objects. Do **NOT** include any extra text or explanations.
            - Ensure the JSON is well-formed and parseable.
            - Use appropriate data types: strings for text fields, numbers for likelihood, arrays for lists.
            - If a field is unknown or not applicable, use `null` or an empty list.
            - If no possible issues are found, return an empty array.

            Input text:
            \"\"\"
            {input_text}
            \"\"\"

            Carefully analyze the symptoms and context, leveraging your expertise, and list all plausible issues as described above.
        """)


    def __init__(self, car_id: str, diagnosis_tree: DiagnosisTreeNode = None, **kwargs: Any):
        """
        Initialize the SymptomExtractorAgent with a language model and car_id.
        Args:
            car_id (str): The unique identifier for the car.
            diagnosis_tree (DiagnosisTreeNode): The diagnosis tree instance for the session.
        """
        super().__init__()
        self.llm_service = LLMService()
        self.prompt = self.get_prompt_template()
        self.output_parser = JsonOutputParser()
        self.car_id = car_id
        self.car_crud = CarCRUD()
        self.logger = Logger("SymptomExtractorAgent")
        self.diagnosis_tree = diagnosis_tree
        self.tree_manager_agent = None
        if self.diagnosis_tree is not None:
            self.tree_manager_agent = TreeManagerAgent(self.diagnosis_tree, llm_service=self.llm_service)

    async def pre_process(self, task: str) -> Dict[str, Any]:
        """
        Pre-process the input task by fetching car data and relevant context.
        Args:
            task (str): The input text describing symptoms/issues.
        Returns:
            Dict[str, Any]: Context dictionary including manuals and scraped data.
        """
        context: Dict[str, Any] = {}
        timings = {}
        t0 = time.perf_counter()
        car = await self.car_crud.get_car_by_id(self.car_id)
        timings['car_fetch'] = time.perf_counter() - t0
        if not car:
            await self.logger.error(f"Car with id {self.car_id} not found.")
            raise ValueError(f"Car with id {self.car_id} not found.")

        input_text = task
        if not input_text:
            raise ValueError("No input text provided for symptom extraction.")

        guide_links: List[str] = car.get("car_guide_links") or []
        # Filter only valid URLs for crawling
        valid_prefixes = ("http://", "https://", "file://", "raw:")
        guide_links = [link for link in guide_links if isinstance(link, str) and link.startswith(valid_prefixes)]
        if not guide_links:
            # No guide links available, return empty context
            return {}

        embedding_service = EmbeddingService()

        # Start embedding and scraping in parallel (pipeline parallelism)
        async def get_embeddings():
            input_vec, link_vecs = await asyncio.gather(
                embedding_service.embed_text(input_text),
                embedding_service.embed_texts(guide_links)
            )
            return input_vec, link_vecs

        async def get_scraped_text(top_links):
            scraper = ScraperService(headless=True)
            try:
                scraped = await scraper.perform_action(top_links, limit=len(top_links))
                return [item.get("text", "") for item in scraped if item.get("text")]
            except Exception:
                return []

        # Get embeddings first to determine top_links
        t1 = time.perf_counter()
        input_vec, link_vecs = await get_embeddings()
        timings['embedding'] = time.perf_counter() - t1
        def cosine_sim(a: List[float], b: List[float]) -> float:
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        scored_links = [
            (link, cosine_sim(input_vec, link_vec))
            for link, link_vec in zip(guide_links, link_vecs)
        ]
        scored_links.sort(key=lambda x: x[1], reverse=True)
        top_links = [link for link, score in scored_links[:3] if score > 0.3]
        context["owner_manual"] = car.get("vector", "")

        # Start scraping in parallel with other work (if any)
        t2 = time.perf_counter()
        if top_links:
            guide_links_text_task = asyncio.create_task(get_scraped_text(top_links))
            guide_links_text = await guide_links_text_task
        else:
            guide_links_text = []
        timings['scraping'] = time.perf_counter() - t2
        context["guide_links_text"] = guide_links_text
        context["timings"] = timings
        await self.logger.info(f"pre_process timings: {timings}")
        return context

    async def handle(self, task: str) -> Any:
        """
        Main handler that runs the symptom extraction chain with lazy context loading and pipeline parallelism.
        Uses only the LLM with minimal context first. If the result is ambiguous (not good enough),
        retries with owner manual and online data as additional context using pre_process.
        If still ambiguous, returns a response indicating more info is needed from the user.
        Args:
            task (str): The input text describing symptoms/issues.
        Returns:
            Any: Parsed JSON array of extracted issues, or a dict requesting more info if ambiguous.
        """
        def is_ambiguous(result):
            # Consider ambiguous if result is empty or all likelihoods are below a threshold
            if not result or not isinstance(result, list):
                return True
            if all((item.get('likelihood', 0) < 30) for item in result if isinstance(item, dict)):
                return True
            return False

        timings = {}
        t_llm_min = time.perf_counter()
        # 1. Try LLM with minimal context (just the input text)
        minimal_documents = [Document(page_content=task, metadata={"type": "input_text_only"})]
        chain = create_stuff_documents_chain(llm=self.llm_service, prompt=self.prompt)
        if hasattr(chain, "ainvoke"):
            response = await chain.ainvoke({"input_text": task, "context": minimal_documents})
        else:
            response = chain.invoke({"input_text": task, "context": minimal_documents})
        timings['llm_minimal'] = time.perf_counter() - t_llm_min
        t_parse_min = time.perf_counter()
        try:
            parsed_result = self.output_parser.parse(response)
        except Exception:
            parsed_result = []
        timings['parse_minimal'] = time.perf_counter() - t_parse_min

        # 2. If ambiguous, fetch context (owner manual and online data) and retry using pre_process
        if is_ambiguous(parsed_result):
            t_context = time.perf_counter()
            context = await self.pre_process(task)
            timings['context'] = time.perf_counter() - t_context
            documents: List[Document] = []
            if context.get("owner_manual"):
                documents.append(Document(page_content=str(context["owner_manual"]), metadata={"type": "owner_manual"}))
            for idx, text in enumerate(context.get("guide_links_text", [])):
                documents.append(Document(page_content=text, metadata={"type": "guide_link", "index": idx}))
            t_llm_full = time.perf_counter()
            if hasattr(chain, "ainvoke"):
                response = await chain.ainvoke({"input_text": task, "context": documents})
            else:
                response = chain.invoke({"input_text": task, "context": documents})
            timings['llm_full'] = time.perf_counter() - t_llm_full
            t_parse_full = time.perf_counter()
            try:
                parsed_result = self.output_parser.parse(response)
            except Exception:
                parsed_result = []
            timings['parse_full'] = time.perf_counter() - t_parse_full
            # Merge in context timings
            if 'timings' in context:
                timings.update({f'context_{k}': v for k, v in context['timings'].items()})

        await self.logger.info(f"handle timings: {timings}")

        # If still ambiguous after all attempts, ask user for more info
        if is_ambiguous(parsed_result):
            return {
                'need_more_info': True,
                'info_type': 'symptom description',
                'response': 'Could you please provide more details about the symptoms or describe the issue more clearly?'
            }

        # Add results to the diagnosis_tree if it exists
        if self.tree_manager_agent is not None and isinstance(parsed_result, list):
            for issue in parsed_result:
                issue_name = issue.get('issue_name', 'Unknown Issue')
                likelihood = issue.get('likelihood', 0) / 100.0  # Convert to 0-1 float
                self.tree_manager_agent.add_symptom(
                    symptom=issue_name,
                    likelyhood=likelihood,
                    data=issue
                )
            # Optionally prune and sort after adding
            self.tree_manager_agent.prune_tree()
            self.tree_manager_agent.sort_tree()

        return parsed_result

    async def process(self, task: str) -> Dict[str, Any]:
        """
        Accepts the incoming request, handles the extraction, and returns the processed result.
        Args:
            task (str): The input text describing symptoms/issues.
        Returns:
            Dict[str, Any]: Contains the result, tree instance, and success status.
        """
        try:
            result = await self.handle(task)
            success = True
        except Exception as e:
            result = None
            success = False
            await self.logger.error(f"Error in process: {e}")

        return {
            "result": result,
            "tree": self.diagnosis_tree,
            "success": success
        }

    async def extract_symptoms(self, input_text: str, context: str = "") -> List[Dict[str, Any]]:
        """
        Extract symptoms from the input text using the LLM service.
        Args:
            input_text (str): The input text describing symptoms/issues.
            context (str): Additional context to include (optional).
        Returns:
            List[Dict[str, Any]]: Parsed JSON array of extracted symptoms.
        """
        prompt = self.prompt.format(input_text=input_text, context=context)
        response = await self.llm_service.generate_response(prompt)
        try:
            return self.output_parser.parse(response)
        except Exception as e:
            await self.logger.error(f"SymptomExtractorAgent parsing error: {e}")
            return []

    def get_langchain_llm(self):
        """
        For advanced LangChain integrations (e.g., chains), use this accessor.
        """
        return self.llm_service.get_llm()
