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
from app.services.search_engine_service import SearchEngineService
from app.services.scraper_service import ScraperService
import numpy as np
import time
import json
from datetime import datetime
from app.utils.logger import Logger
from app.utils.diagnosis_tree import DiagnosisTreeNode
from app.agents.tree_manager_agent import TreeManagerAgent
from app.utils.websocket import manager  # WebSocket manager for broadcasting stages
import json
from app.agents.base_agent import BaseAgent
from app.core.interfaces import IWebSocketManager
from app.utils.message_types import MessageSource
from app.utils.monitoring import monitor_and_handle
from app.agents.session_context_agent import SessionContextAgent

class SymptomExtractorAgent(BaseAgent):
    """
    Agent for extracting symptoms from input text.
    """

    @staticmethod
    def get_prompt_template() -> PromptTemplate:
        return PromptTemplate.from_template(
            """
            You are an expert automotive mechanic and diagnostic specialist with extensive experience in automotive systems and failure diagnostics.
            Your goal is to help the user diagnose and, if possible, resolve the issue themselves. Provide clear, step-by-step instructions for safe DIY troubleshooting and minor repairs. Only recommend seeing a mechanic if the issue is dangerous, requires specialized tools, or cannot be safely addressed by a typical car owner.

            Always include safety warnings before any potentially hazardous steps. Use simple language and explain technical terms. Do NOT recommend visiting a mechanic unless absolutely necessary. Try to empower the user to understand and address the problem first.

            IMPORTANT:
            - You have access to the car's make, model, and year: Make: {car_make}, Model: {car_model}, Year: {car_year}.
            - Do NOT ask the user for car make, model, or year; you already have this information.
            - Do NOT tell the user to refer to the owner's manual; use the information from the manual directly in your response.

            Using up to the last 5 user messages (provided below, most recent last) and any provided context (such as previous diagnostics, vehicle data, or sensor readings), along with your comprehensive automotive knowledge, identify all plausible underlying issues that could cause the described symptoms.
            Carefully analyze the conversation context, symptoms, provided context, and the current diagnosis tree (if available) using your diagnostic expertise and automotive knowledge base. Include:
            - Common causes that match these symptoms
            - Less likely but critical failures that should not be overlooked
            - Any issue that could contribute indirectly or as a downstream effect
            - If the diagnosis tree exists, mention other possible causes from the tree that may be relevant, as "Other Possible Causes" in your output.

            For each possible issue, provide a detailed explanation of why it is plausible, and include step-by-step diagnostic or repair suggestions for the user (if applicable). If any information is missing or ambiguous, clearly state what is needed and ask the user for it in a friendly way.

            Think broadly and reason through how multiple issues may be connected. Output a complete, well-structured JSON array.
            Car make: {car_make}
            Car model: {car_model}
            Car year: {car_year}
            Context:
            {context}

            For each possible issue, output a detailed JSON object with the following fields:
            - **issue_name**: A concise name for the issue (e.g., "Engine Knock")
            - **likelihood**: Your estimated likelihood (0–100) that this issue is causing the symptom
            - **issue_type**: The general type of the issue (e.g., "mechanical", "electrical", "software")
            - **issue_category**: The broad category of the issue (e.g., "engine", "transmission", "fuel system")
            - **issue_subcategory**: A specific subcategory if applicable (e.g., "ignition system", "fuel injection")
            - **issue_description**: A clear, detailed explanation of the issue, how it relates to the symptom, and step-by-step diagnostic or repair suggestions for the user (if applicable)
            - **severity**: The severity level of the issue, one of ["low", "medium", "high"]
            - **additional_info**: Any other relevant details (common causes, conditions, diagnostic tips, etc.)
            - **other_possible_causes**: If the diagnosis tree suggests other plausible causes or related issues, list them here as additional suggestions (otherwise null or empty list).
            - **missing_info_request**: If any information is missing or ambiguous, specify what to ask the user for, otherwise null or empty string.

            Important instructions:
            - Return **ONLY** a valid JSON array of issue objects. Do **NOT** include any extra text or explanations.
            - Ensure the JSON is well-formed and parseable.
            - Use appropriate data types: strings for text fields, numbers for likelihood, arrays for lists.
            - If a field is unknown or not applicable, use `null` or an empty list.
            - If no possible issues are found, return an empty array.
            - Analyze up to the last 5 user messages as a conversation (most recent last) to extract all relevant symptoms and context.

            Input:
            - Car make: {car_make}
            - Car model: {car_model}
            - Car year: {car_year}
            - User messages (up to last 5, most recent last):
            """
            """{input_text}"""
            """

            Carefully analyze the symptoms and context, leveraging your expertise, and list all plausible issues as described above.
            Ensure the JSON is valid and well-formed. Do NOT include any additional text, markdown, or explanations outside the JSON object.
            Return ONLY a valid JSON object as specified above. Do NOT include any extra text, markdown, explanations outside the JSON or surrounded by backticks.
            """
        )


    def __init__(
        self,
        car_id: str,
        diagnosis_tree: Optional[DiagnosisTreeNode] = None,
        car_make: Optional[str] = None,
        car_model: Optional[str] = None,
        car_year: Optional[str] = None,
        websocket_manager: Optional[IWebSocketManager] = None,
        llm_service: Optional[LLMService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        search_engine_service: Optional[SearchEngineService] = None,
        scraper_service: Optional[ScraperService] = None,
        tree_manager_agent: Optional[TreeManagerAgent] = None,
        session_context_manager: Optional[SessionContextAgent] = None,
        **kwargs: Any
    ):
        """
        Initialize the SymptomExtractorAgent with all dependencies injected for testability.
        """
        super().__init__(car_crud=CarCRUD(), car_id=car_id, car_make=car_make, car_model=car_model, car_year=car_year, logger_name="SymptomExtractorAgent", websocket_manager=websocket_manager)
        self.llm_service = llm_service or LLMService()
        self.prompt = self.get_prompt_template()
        self.output_parser = JsonOutputParser()
        self.embedding_service = embedding_service or EmbeddingService()
        self.search_engine_service = search_engine_service or SearchEngineService()
        self.scraper_service = scraper_service or ScraperService(headless=True)
        self.session_context_manager = session_context_manager or SessionContextAgent()
        
        
        # Initialize diagnosis tree if not provided
        if diagnosis_tree is None:
            from app.utils.diagnosis_tree_factory import get_diagnosis_tree
            self.diagnosis_tree = get_diagnosis_tree(issue_name='root', likelyhood=1.0)
        else:
            self.diagnosis_tree = diagnosis_tree
            
        if tree_manager_agent is not None:
            self.tree_manager_agent = tree_manager_agent
            # Ensure the tree manager is using the same tree reference
            if hasattr(tree_manager_agent, 'root') and tree_manager_agent.root != self.diagnosis_tree:
                # Note: Cannot use async logger in __init__, will log on first method call
                self._init_warning = f"TreeManagerAgent root differs from provided diagnosis_tree. TreeManager root id: {id(tree_manager_agent.root)}, Provided tree id: {id(self.diagnosis_tree)}"
                # Force the tree manager to use our tree
                tree_manager_agent.root = self.diagnosis_tree
        elif self.diagnosis_tree is not None:
            self.tree_manager_agent = TreeManagerAgent(self.diagnosis_tree, llm_service=self.llm_service)
            # Verify the tree manager is using the correct tree
            if self.tree_manager_agent.root != self.diagnosis_tree:
                self._init_error = f"TreeManagerAgent created with wrong tree reference! TreeManager root id: {id(self.tree_manager_agent.root)}, Expected tree id: {id(self.diagnosis_tree)}"
        else:
            self.tree_manager_agent = None
        
        # Initialize warning/error flags for logging in async methods
        self._init_warning = getattr(self, '_init_warning', None)
        self._init_error = getattr(self, '_init_error', None)

    async def pre_process(self, task: Any) -> Dict[str, Any]:
        """
        PERFORMANCE OPTIMIZED: Pre-process with minimal context fetching and parallel operations.
        """
        await self._log_entry("pre_process")
        
        await self.broadcast_stage(json.dumps({"type": "stage", "stage": "Symptom extraction - Pre-processing context"}))
        context: Dict[str, Any] = {}
        timings = {}
        t0 = time.perf_counter()
        await self._ensure_car_info()
        if self.car_make and self.car_model and self.car_year:
            car = {'make': self.car_make, 'model': self.car_model, 'year': self.car_year, 'id': self.car_id}
        else:
            car = None
        timings['car_fetch'] = time.perf_counter() - t0
        if not car:
            await self.logger.error(f"Car with id {self.car_id} not found.")
            raise ValueError(f"Car with id {self.car_id} not found.")

        input_text = self._concat_user_messages(task)
        if not input_text:
            raise ValueError("No input text provided for symptom extraction.")

        # PERFORMANCE OPTIMIZATION: Skip guide links processing for now to reduce latency
        # This eliminates the embedding and scraping overhead that was adding significant time
        
        # Fetch owner manual text from DB and perform vector search on it - REDUCED scope
        user_message_concat = self._concat_user_messages(task)
        if self.search_engine_service and self.car_id:
            # Use the optimized embed_and_vector_search with reduced scope
            t2 = time.perf_counter()
            manual_chunks = await self.search_engine_service.embed_and_vector_search(
                car_id=self.car_id, query=user_message_concat, top_k=1  # Keep minimal for symptom extraction
            )
            owner_manual_text = manual_chunks[0]["chunk"][:1000] if manual_chunks else ""  # Limit to 1000 chars
            timings['manual_search'] = time.perf_counter() - t2
        else:
            owner_manual_text = ""
        
        context["owner_manual"] = owner_manual_text
        context["guide_links_text"] = []  # Skip guide links processing for performance
        context["timings"] = timings
        await self.logger.info(f"OPTIMIZED pre_process timings: {timings}")
        
        await self._log_exit("pre_process", success=True, 
                           guide_links_count=0,  # No guide links processed for performance
                           has_manual=bool(owner_manual_text))
        return context

    async def handle(self, task: Any) -> Any:
        """
        PERFORMANCE OPTIMIZED: Main handler with minimal context loading and faster processing.
        """
        await self._log_entry("handle")
        
        def is_ambiguous(result):
            # Consider ambiguous if result is empty or all likelihoods are below a threshold
            if not result or not isinstance(result, list):
                return True
            if all((item.get('likelihood', 0) < 30) for item in result if isinstance(item, dict)):
                return True
            return False

        timings = {}
        user_message_concat = self._concat_user_messages(task)
        
        # PERFORMANCE OPTIMIZATION: Try LLM with basic manual context first
        t_context = time.perf_counter()
        context = {}
        if self.search_engine_service and self.car_id:
            # Get minimal manual context for better symptom extraction
            manual_chunks = await self.search_engine_service.embed_and_vector_search(
                car_id=self.car_id, query=user_message_concat, top_k=1
            )
            manual_text = manual_chunks[0]["chunk"][:800] if manual_chunks else ""  # Limit to 800 chars
            context["owner_manual"] = manual_text
        else:
            context["owner_manual"] = ""
        timings['context_basic'] = time.perf_counter() - t_context
        
        # Create documents for LLM
        documents = [Document(page_content=user_message_concat, metadata={"type": "input_text"})]
        if context.get("owner_manual"):
            documents.append(Document(page_content=context["owner_manual"], metadata={"type": "owner_manual"}))
        
        # Run LLM with basic context
        chain = create_stuff_documents_chain(llm=self.llm_service.get_llm(), prompt=self.prompt)
        prompt_vars = {
            "input_text": user_message_concat,
            "context": documents,
            "car_make": self.car_make or "",
            "car_model": self.car_model or "",
            "car_year": self.car_year or ""
        }
        
        t_llm = time.perf_counter()
        if hasattr(chain, "ainvoke"):
            response = await chain.ainvoke(prompt_vars)
        else:
            response = chain.invoke(prompt_vars)
        timings['llm_call'] = time.perf_counter() - t_llm
        
        t_parse = time.perf_counter()
        try:
            parsed_result = self.output_parser.parse(response)
        except Exception:
            parsed_result = []
        timings['parse'] = time.perf_counter() - t_parse

        # If result is still ambiguous, ask for more info instead of extended processing
        if is_ambiguous(parsed_result):
            await self.logger.info("Result is ambiguous, requesting more info from user")
            await self._log_exit("handle", success=True, result_type="more_info_needed")
            return {
                'need_more_info': True,
                'info_type': 'symptom description',
                'response': 'Could you please provide more details about the symptoms or describe the issue more clearly?'
            }

        await self.broadcast_stage(json.dumps({"type": "stage", "stage": "Symptom extraction - Completed"}))
        

        await self.logger.info(f"OPTIMIZED handle timings: {timings}")

        # Log parsed result information
        await self.logger.info(f"Parsed result type: {type(parsed_result)}, items count: {len(parsed_result) if isinstance(parsed_result, list) else 'N/A'}")

        # Add results to the diagnosis_tree if it exists
        if self.tree_manager_agent is not None and isinstance(parsed_result, list):
            await self.logger.info(f"Adding {len(parsed_result)} symptoms to diagnosis tree")
            tree_before_count = len(self.diagnosis_tree.children) if self.diagnosis_tree else 0
            
            # For faster processing, add symptoms sequentially but with optimized tree building
            for issue in parsed_result:
                issue_name = issue.get('issue_name', 'Unknown Issue')
                likelihood = issue.get('likelihood', 0) / 100.0  # Convert to 0-1 float
                await self.logger.info(f"Adding symptom: {issue_name} with likelihood {likelihood}")
                
                await self.tree_manager_agent.add_symptom(
                    symptom=issue_name,
                    likelyhood=likelihood,
                    data=issue
                )
            
            # Optionally prune and sort after adding with a lower threshold to preserve more symptoms
            await self.logger.info("Pruning tree with threshold 0.2 (20%)")
            self.tree_manager_agent.prune_tree(threshold=0.2)
            
            # Log pruning stats if available
            if hasattr(self.tree_manager_agent, '_last_prune_stats'):
                stats = self.tree_manager_agent._last_prune_stats
                await self.logger.info(f"Pruning completed: {stats['initial_count']} -> {stats['final_count']} children (threshold: {stats['threshold']})")
                
            self.tree_manager_agent.sort_tree()
            
            tree_after_count = len(self.diagnosis_tree.children) if self.diagnosis_tree else 0
            await self.logger.info(f"Tree children count: before={tree_before_count}, after={tree_after_count}")
            
            # Verify the tree has been updated
            if tree_after_count > tree_before_count:
                await self.logger.info("Symptoms successfully added to diagnosis tree")
            else:
                await self.logger.warning("No symptoms were added to the diagnosis tree")
        else:
            if self.tree_manager_agent is None:
                await self.logger.warning("TreeManagerAgent is None, cannot add symptoms to tree")
            if not isinstance(parsed_result, list):
                await self.logger.warning(f"Parsed result is not a list: {type(parsed_result)}")

        processed_result = [self._sanitize_output(json.dumps(issue)) if isinstance(issue, dict) else self._sanitize_output(str(issue)) for issue in parsed_result] if isinstance(parsed_result, list) else parsed_result
        
        # Return both the result and the updated diagnosis tree
        # Ensure we return the actual tree that was modified
        actual_tree = self.tree_manager_agent.root if self.tree_manager_agent else self.diagnosis_tree
        
        # Log tree state
        if actual_tree:
            await self.logger.info(f"Returning tree with {len(actual_tree.children)} children")
        
        # Send tree data immediately via WebSocket if available
        if hasattr(self, 'websocket') and self.websocket:
            try:
                tree_data = actual_tree.to_dict() if actual_tree else None
                if tree_data:
                    tree_message = {
                        "type": "tree_data",
                        "source": "symptom_extraction",
                        "content": "Tree data updated with extracted symptoms",
                        "timestamp": datetime.now().isoformat() + "Z",
                        "data": {
                            "tree_data": tree_data,
                            "stage": "symptom_extraction_complete",
                            "symptoms_count": len(processed_result) if isinstance(processed_result, list) else 0,
                            "tree_children_count": len(actual_tree.children) if actual_tree else 0
                        }
                    }
                    await self.websocket.send_text(json.dumps(tree_message))
                    await self.logger.info(f"Sent tree data via WebSocket: {len(actual_tree.children)} children")
            except Exception as e:
                await self.logger.error(f"Failed to send tree data via WebSocket: {e}")
        
        await self._log_exit("handle", success=True, 
                           symptoms_count=len(processed_result) if isinstance(processed_result, list) else 0,
                           tree_children=len(actual_tree.children) if actual_tree else 0)
        
        return {
            'symptoms': processed_result,
            'diagnosis_tree': actual_tree
        }

    @monitor_and_handle("SymptomExtractorAgent")
    async def process(self, user_message: str, websocket=None, session_id=None) -> Any:
        """
        Main entry point for symptom extraction. Handles websocket communication and error reporting.
        Session-aware: integrates with session context management to maintain focus.
        Args:
            user_message (str): The user's message describing symptoms.
            websocket: Optional websocket connection for real-time updates.
            session_id: Optional session identifier for context management.
        Returns:
            Dict[str, Any]: Extracted symptoms and updated diagnosis tree.
        """
        await self._log_entry("process", message_length=len(user_message), session_id=session_id)
        
        # Store websocket reference for use in handle method
        self.websocket = websocket
        
        # Log any init warnings/errors
        if self._init_warning:
            await self.logger.warning(self._init_warning)
            self._init_warning = None
        if self._init_error:
            await self.logger.error(self._init_error)
            self._init_error = None
        
        try:
            # Stage 1: Symptom extraction started
            if websocket:
                await self.send_ws_stage(websocket, "Symptom extraction started", MessageSource.SYMPTOM_EXTRACTION, session_id=session_id)
            
            # Stage 2: Pre-processing
            if websocket:
                await self.send_ws_stage(websocket, "Pre-processing user input", MessageSource.SYMPTOM_EXTRACTION, session_id=session_id)
            
            # SESSION CONTEXT INTEGRATION
            session_context = None
            context_reminder = ""
            
            if session_id:
                # Get existing session context if available
                session_context = self.session_context_manager.get_original_context(session_id)
                
                if session_context:
                    # Get context reminder for focused extraction - symptom extraction is our job, not the session agent's
                    context_reminder = self.session_context_manager.get_context_reminder(session_id)
                    await self.logger.info(f"Session {session_id}: Using context - {session_context.issue_category} issue with {len(session_context.symptoms)} symptoms")
                else:
                    # First message in session - extract original issue context
                    session_context = await self.session_context_manager.extract_original_issue(user_message, session_id)
                    await self.logger.info(f"Session {session_id}: Extracted original issue - {session_context.primary_issue} (category: {session_context.issue_category})")
            
            # Enhanced message processing with session context
            enhanced_message = user_message
            if context_reminder:
                enhanced_message = f"{context_reminder}\n\nUser message: {user_message}"
            
            # Stage 3: Analyzing symptoms
            if websocket:
                await self.send_ws_stage(websocket, "Analyzing symptoms and extracting issues", MessageSource.SYMPTOM_EXTRACTION, session_id=session_id)
            
            result = await self.handle(enhanced_message)
            
            # Stage 4: Building diagnosis tree
            if websocket:
                tree_data = None
                if isinstance(result, dict) and 'diagnosis_tree' in result and result['diagnosis_tree']:
                    tree = result['diagnosis_tree']
                    tree_data = {
                        "total_nodes": len(tree.children) if tree else 0,
                        "root_issue": tree.issue_name if tree else "Unknown",
                        "children": [
                            {
                                "issue_name": child.issue_name,
                                "likelihood": child.likelyhood,
                                "type": child.data.get("issue_type") if child.data else "Unknown",
                                "category": child.data.get("issue_category") if child.data else "Unknown"
                            }
                            for child in (tree.children if tree else [])
                        ][:5]  # Send first 5 children to avoid large payloads
                    }

                await self.send_ws_stage(
                    websocket, 
                    "Building diagnosis tree with extracted symptoms", 
                    MessageSource.SYMPTOM_EXTRACTION, 
                    session_id=session_id, 
                    details={"tree_data": tree_data}
                )

            # Update session context with extracted symptoms if available
            if session_id and session_context and isinstance(result, dict) and 'symptoms' in result:
                # Note: We don't update diagnosis tree context here - that's the DiagnosticAgent's job
                # The session context agent only manages session focus, not diagnosis updates
                await self.logger.info(f"Session {session_id}: Extracted {len(result.get('symptoms', []))} symptoms")
            
            # Stage 5: Symptom extraction complete
            if websocket:
                extraction_summary = {
                    "symptoms_count": len(result.get('symptoms', [])) if isinstance(result, dict) else 0,
                    "has_tree": isinstance(result, dict) and 'diagnosis_tree' in result and result['diagnosis_tree'] is not None,
                    "session_context": session_context.to_dict() if session_context else None
                }
                await self.send_ws_result(websocket, "Symptom extraction complete", MessageSource.SYMPTOM_EXTRACTION, session_id=session_id, details=extraction_summary)
            
            # Return both symptoms and updated diagnosis tree with session info
            if isinstance(result, dict) and 'symptoms' in result:
                process_result = {
                    "result": result['symptoms'],
                    "diagnosis_tree": result['diagnosis_tree'],
                    "session_context": session_context.to_dict() if session_context else None
                }
            else:
                # Fallback for backwards compatibility
                process_result = {
                    "result": result, 
                    "diagnosis_tree": self.diagnosis_tree,
                    "session_context": session_context.to_dict() if session_context else None
                }
            
            await self._log_exit("process", success=True, 
                               has_symptoms='symptoms' in result if isinstance(result, dict) else False,
                               session_context_available=session_context is not None)
            return process_result
            
        except Exception as e:
            if websocket:
                await self.send_ws_error(websocket, "Error during symptom extraction", MessageSource.SYMPTOM_EXTRACTION, session_id=session_id, details={"error": str(e)})
            await self._log_exit("process", success=False, error=str(e))
            raise

    async def extract_symptoms(self, input_text: Any, context: str = "") -> List[Dict[str, Any]]:
        """
        Extract symptoms from the input text using the LLM service.
        Args:
            input_text (Any): The input text or list of user messages describing symptoms/issues.
            context (str): Additional context to include (optional).
        Returns:
            List[Dict[str, Any]]: Parsed JSON array of extracted symptoms.
        """
        await self._ensure_car_info()
        user_message_concat = self._concat_user_messages(input_text)
        prompt = self.prompt.format(input_text=user_message_concat, context=context)
        response = await self.llm_service.generate_response(prompt)
        response = self._sanitize_output(response)
        try:
            return self.output_parser.parse(response)
        except Exception as e:
            await self.logger.error(f"SymptomExtractorAgent parsing error: {e}")
            return []

    def _concat_user_messages(self, messages: Any) -> str:
        """
        Utility to join up to the last 5 user messages (most recent last).
        Accepts a string or a list of strings.
        """
        if isinstance(messages, list):
            return "\n".join(messages[-5:])
        elif isinstance(messages, str):
            return messages
        return ""

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
