from typing import Any, Dict, List, Optional
import asyncio
from app.utils.logger import Logger
from .symptom_extractor_agent import DiagnosisTreeAgent
from .base import AgentBase

class DiagnosisAgent(AgentBase):
    """
    Agent that uses an LLM to make a diagnosis and suggest fixes for a specific car (make, model, year).
    It takes extracted symptoms, builds/updates a diagnosis tree, and adds diagnosis and suggested fixes to the tree data.
    Inherits from AgentBase for caching, logging, and performance tracking.
    """
    def __init__(self, services: Dict[str, Any], config: Dict[str, Any] = None, logger: Optional[Logger] = None):
        super().__init__(services, config, logger)
        self.llm = self.services.get('llm')
        self.prompt_template = self.services.get('diagnosis_prompt')
        self.diagnosis_tree_agent = self.services.get('diagnosis_tree_agent')
        if not self.diagnosis_tree_agent:
            # Fallback: create a new tree agent if not provided
            self.diagnosis_tree_agent = DiagnosisTreeAgent(
                llm=self.llm,
                prompt=self.prompt_template,
                logger=self.logger
            )

    async def can_handle(self, task: str, context: Dict[str, Any]) -> bool:
        keywords = ['diagnose', 'diagnosis', 'fix', 'suggest', 'repair']
        return any(k in task.lower() for k in keywords)

    async def pre_process(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.logger.info(f"Pre-processing diagnosis task: {task}")
        # Optionally validate car_info and symptoms
        return context

    async def handle(self, task: str, context: Dict[str, Any]) -> Any:
        car_info = context.get('car_info', {})
        symptoms = context.get('symptoms', [])
        make = car_info.get('make', '')
        model = car_info.get('model', '')
        year = car_info.get('year', '')
        symptom_texts = [s.get('text', '') for s in symptoms]
        joined_symptoms = '\n'.join(symptom_texts)

        # Build LLM prompt
        prompt = f"""
        You are an expert automotive diagnostician. Given the following car and symptoms, provide:
        1. A likely diagnosis (with a short explanation)
        2. Suggested fixes (step-by-step, specific to the car make/model/year)
        3. The automotive system involved
        4. Severity (low/medium/high)
        5. Confidence (0.0-1.0)
        Car: {year} {make} {model}
        Symptoms:\n{joined_symptoms}
        Respond in JSON with keys: diagnosis, explanation, fixes (list), system, severity, confidence.
        """

        chat_service = self.services.get('chat_service')
        if chat_service:
            response = await chat_service.process_message(message=prompt, context=context)
            content = response.get('content', '{}') if response else '{}'
        else:
            # Fallback: use LLM directly if available
            if hasattr(self.llm, 'ainvoke'):
                content = await self.llm.ainvoke(prompt)
            else:
                content = '{}'

        # Parse LLM response
        import json
        try:
            diagnosis_data = json.loads(content)
        except Exception as e:
            await self.logger.error(f"Failed to parse LLM diagnosis: {e}")
            diagnosis_data = {
                'diagnosis': 'Unknown',
                'explanation': '',
                'fixes': [],
                'system': 'unknown',
                'severity': 'unknown',
                'confidence': 0.0
            }

        # Add to diagnosis tree
        node_name = diagnosis_data.get('diagnosis', 'Unknown')
        data = {
            'diagnosis': diagnosis_data.get('diagnosis'),
            'explanation': diagnosis_data.get('explanation'),
            'fixes': diagnosis_data.get('fixes', []),
            'system': diagnosis_data.get('system'),
            'severity': diagnosis_data.get('severity'),
            'confidence': diagnosis_data.get('confidence'),
            'car_info': car_info,
            'symptoms': symptoms
        }
        # Insert or update node in the tree
        await self.diagnosis_tree_agent.update_issue(node_name, data)

        return {
            'success': True,
            'diagnosis': diagnosis_data,
            'tree_node': node_name
        }

    async def post_process(self, result: Any, context: Dict[str, Any]) -> Any:
        await self.logger.info("Diagnosis post-processing completed")
        return result
