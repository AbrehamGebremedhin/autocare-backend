from app.CRUD.car_crud import CarCRUD
from .base import AgentBase
from typing import Any, Dict, List, Optional
from app.core.config import get_settings
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
import asyncio
from functools import partial

class SymptomExtractorAgent(AgentBase):
    """
    Agent for extracting symptoms from input text.
    """

    @staticmethod
    def get_prompt_template() -> PromptTemplate:
        return PromptTemplate.from_template("""
            You are an expert automotive mechanic and diagnostic specialist. Your task is to carefully analyze the input text describing car symptoms or problems and extract **all possible underlying issues** that could cause these symptoms.

            Context information (may help you reason better):
            {context}

            For each possible issue, provide a detailed and structured JSON object with the following fields:

            - **issue_name**: The concise name of the issue (e.g., "Engine Knock")
            - **likelihood**: Your estimated likelihood of this issue causing the symptom, as a percentage between 0 and 100 (e.g., 70)
            - **issue_type**: The general type of issue (e.g., "mechanical", "electrical", "software")
            - **issue_category**: The broad category of the issue (e.g., "engine", "transmission", "fuel system")
            - **issue_subcategory**: A more specific subcategory if applicable (e.g., "ignition system", "fuel injection")
            - **issue_description**: A clear, detailed explanation of the issue and how it relates to the symptom
            - **severity**: The severity level of the issue, one of ["low", "medium", "high"]
            - **related_symptoms**: A list of symptoms or signs associated with this issue (e.g., ["engine knocking noise", "loss of power"])
            - **additional_info**: Any other relevant details, such as common causes, conditions, or diagnostic tips

            **Important instructions:**
            - Return ONLY a valid JSON array of issue objects—do NOT include any explanations, apologies, or extra text.
            - Ensure the JSON is well-formed and can be parsed without errors.
            - Use appropriate data types: strings for text fields, numbers for likelihood, and arrays for lists.
            - If any field is unknown or not applicable, use `null` or an empty list as appropriate.
            - If no possible issues are found, return an empty array.

            Input text:
            \"\"\"
            {input_text}
            \"\"\"

            Analyze the symptoms described and extract all plausible issues that could cause them, with detailed information as specified.
        """)


    def __init__(self, car_id: str, **kwargs: Any):
        """
        Initialize the SymptomExtractorAgent with a language model and car_id.
        Args:
            car_id (str): The unique identifier for the car.
        """
        super().__init__()
        settings = get_settings()
        self.gemini_api_key = settings.GEMINI_KEY
        self.gemini_model = settings.GEMINI_MODEL_1
        self.llm = ChatGoogleGenerativeAI(
            model=self.gemini_model,
            google_api_key=self.gemini_api_key
        )

        self.prompt = self.get_prompt_template()
        self.output_parser = JsonOutputParser()
        self.car_id = car_id
        self.car_crud = CarCRUD()

    async def pre_process(self, task: str) -> Dict[str, Any]:
        """
        Pre-process the input task by fetching car data and relevant context.
        Args:
            task (str): The input text describing symptoms/issues.
        Returns:
            Dict[str, Any]: Context dictionary including manuals and scraped data.
        """
        context: Dict[str, Any] = {}
        car = await self.car_crud.get_car_by_id(self.car_id)
        if not car:
            raise ValueError(f"Car with id {self.car_id} not found.")

        input_text = task
        if not input_text:
            raise ValueError("No input text provided for symptom extraction.")

        guide_links: List[str] = car.get("car_guide_links") or []
        if not guide_links:
            # No guide links available, return empty context
            return {}

        from app.services.embedding_service import EmbeddingService
        import numpy as np

        embedding_service = EmbeddingService()

        # Embed input text and guide links
        input_vec = await embedding_service.embed_text(input_text)
        link_vecs = await embedding_service.embed_texts(guide_links)

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

        if top_links:
            from app.services.scraper_service import ScraperService
            scraper = ScraperService(headless=True)
            try:
                scraped = await scraper.perform_action(top_links, limit=len(top_links))
                guide_links_text = [item.get("text", "") for item in scraped if item.get("text")]
            except Exception as e:
                guide_links_text = []
        else:
            guide_links_text = []

        context["guide_links_text"] = guide_links_text
        return context

    async def handle(self, task: str) -> Any:
        """
        Main handler that runs the symptom extraction chain.
        Args:
            task (str): The input text describing symptoms/issues.
        Returns:
            Any: Parsed JSON array of extracted issues.
        """
        context = await self.pre_process(task)
        documents: List[Document] = []

        if context.get("owner_manual"):
            documents.append(Document(page_content=str(context["owner_manual"]), metadata={"type": "owner_manual"}))
        for idx, text in enumerate(context.get("guide_links_text", [])):
            documents.append(Document(page_content=text, metadata={"type": "guide_link", "index": idx}))

        chain = create_stuff_documents_chain(llm=self.llm, prompt=self.prompt)

        # Use ainvoke/invoke for modern LangChain
        if hasattr(chain, "ainvoke"):
            response = await chain.ainvoke({"input_text": task, "context": documents})
        else:
            response = chain.invoke({"input_text": task, "context": documents})

        # Parse JSON output
        try:
            parsed_result = self.output_parser.parse(response)
        except Exception as e:
            # Handle JSON parse errors gracefully
            parsed_result = []

        return parsed_result

    async def post_process(self, result: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Post-process the extracted symptom data if needed.
        Args:
            result (Any): Parsed symptom extraction result.
            context (Optional[Dict[str, Any]]): Optional context dictionary.
        Returns:
            Any: Final processed result.
        """
        return await super().post_process(result, context)
import time

# Example usage
if __name__ == "__main__":
    import asyncio
    from pprint import pprint

    async def main():
        car_id = "toyota-echo-2001"  # Replace with actual car ID
        agent = SymptomExtractorAgent(car_id=car_id)
        task = "The engine is making a strange noise and the check engine light is on."
        start_time = time.perf_counter()
        result = await agent.handle(task)
        end_time = time.perf_counter()
        pprint(result)
        print(f"Processing time: {end_time - start_time:.2f} seconds")

    asyncio.run(main())
