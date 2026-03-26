# ============================================
# LLM-epiparam generator agent
# ============================================

from typing import Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from formats.data_formats import EpiParameters, Episode, GraphState

class LLMEpiParamGenerator:
    """
    LLM agent that generates epidemiological parameters for SIRD model
    """
    
    def __init__(self, llm, output_class=EpiParameters):
        """
        Initialize the generator agent
        
        Args:
            llm: LangChain LLM instance (ChatHuggingFace)
            output_class: Pydantic class for output validation
        """
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=output_class)
        self.prompt = self._create_prompt()
        self.history: List[Episode] = []
        self.task_config: Dict = {}
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for parameter generation"""
        
        prompt_template = ChatPromptTemplate(messages=[
            ("system", """You are an expert epidemiologist specialized in SIRD model parameter optimization. Your task is to select optimal parameters (β, γ, μ) to achieve target epidemic scenarios.

**Your role:**
- Analyze the current epidemic state and history of previous attempts
- Consider expert feedback to improve parameter selection
- Apply epidemiological knowledge about parameter relationships
- Balance exploration and exploitation in the parameter space

**Key epidemiological relationships:**
- Higher β → earlier and higher infection peak
- Lower β → later and lower infection peak
- Higher γ → earlier and lower peak (faster recovery)
- Lower γ → later and higher peak (slower recovery)
- Higher μ → lower peak, more deaths
- Lower μ → higher peak, fewer deaths

**Optimization strategy:**
1. Start with reasonable parameter ranges
2. Adjust based on expert feedback
3. Learn from successful and failed attempts
4. Consider trade-offs between peak timing, peak height, and mortality

{format_instructions}

Return ONLY valid JSON in the exact format specified. Do not include any additional text, explanations, or markdown formatting."""),

            ("human", """Please generate new epidemiological parameters for the SIRD model based on the following information:

## 1. Task Configuration
{task_config}

## 2. Current Episode
{current_episode}

## 3. History of Previous Episodes
{history}

## 4. Expert Comment
{expert_comment}

## 5. Statistical Summary
{stats_summary}

## 6. Target Metrics
{target_metrics}

Generate new parameters that will bring us closer to the target epidemic scenario.
Return only the valid JSON object without any additional text.""")
        ],
            partial_variables={"format_instructions": self.parser.get_format_instructions()})
        
        return prompt_template
    
    def _format_history(self, history: List[Episode], max_episodes: int = 5) -> str:
        """Format history episodes for prompt"""
        if not history:
            return "No previous episodes available."
        
        # Take only last N episodes to keep prompt manageable
        recent_history = history[-max_episodes:]
        
        formatted = ""
        for episode in recent_history:
            formatted += episode.to_prompt_format() + "\n"
        
        return formatted
    
    def _format_stats_summary(self, history: List[Episode]) -> str:
        """Generate statistical summary from history"""
        if not history:
            return "No statistics available yet."
        
        betas = [e.beta for e in history]
        gammas = [e.gamma for e in history]
        mus = [e.mu for e in history]
        peaks = [e.peak_position for e in history if e.peak_position]
        deaths = [e.total_deaths for e in history if e.total_deaths]
        
        # Find best episode based on target
        target_peak = self.task_config.get('target_peak', 30)
        best_episode = min(history, key=lambda x: abs(x.peak_position - target_peak)) if peaks else None
        
        summary = f"""
**Parameter ranges explored:**
- β: {min(betas):.4f} – {max(betas):.4f}
- γ: {min(gammas):.4f} – {max(gammas):.4f}
- μ: {min(mus):.5f} – {max(mus):.5f}

**Results achieved:**
- Peak position: {min(peaks):.1f} – {max(peaks):.1f} days (target: {target_peak})
- Total deaths: {min(deaths):.0f} – {max(deaths):.0f}

**Best attempt so far:**
"""
        if best_episode:
            summary += f"""
- β={best_episode.beta:.4f}, γ={best_episode.gamma:.4f}, μ={best_episode.mu:.5f}
- Peak at day {best_episode.peak_position:.1f} with {best_episode.peak_height:.0f} infected
- Total deaths: {best_episode.total_deaths:.0f}
"""
        
        return summary
    
    def _format_current_episode(self, episode: Episode) -> str:
        """Format current episode for prompt"""
        if not episode:
            return "No current episode available."
        
        return f"""
- β = {episode.beta:.4f}
- γ = {episode.gamma:.4f}
- μ = {episode.mu:.5f}
- Peak position: {episode.peak_position:.1f} days
- Peak height: {episode.peak_height:.0f} infected
- Total deaths: {episode.total_deaths:.0f}
"""
    
    def _format_task_config(self) -> str:
        """Format task configuration for prompt"""
        if not self.task_config:
            return "No task configuration provided."
        
        config_str = f"""
- Description: {self.task_config.get('description', 'Not specified')}
- Population: {self.task_config.get('population', '1,000,000'):,}
- Initial infected: {self.task_config.get('I0', 100)}
- Target peak: {self.task_config.get('target_peak', 'Not specified')} days
"""
        return config_str
    
    def _format_target_metrics(self) -> str:
        """Format target metrics for prompt"""
        if not self.task_config:
            return "No target metrics specified."
        
        metrics = f"""
- Desired peak position: {self.task_config.get('target_peak', 'Not specified')} days
"""
        if self.task_config.get('target_height'):
            metrics += f"- Desired peak height: {self.task_config.get('target_height'):,.0f} infected\n"
        if self.task_config.get('target_deaths'):
            metrics += f"- Desired total deaths: {self.task_config.get('target_deaths'):,.0f}\n"
        
        return metrics
    
    def set_task_config(self, config: Dict):
        """Set the task configuration"""
        self.task_config = config
        print(f"✅ Task configured: {config.get('description', 'No description')}")
    
    def add_to_history(self, episode: Episode):
        """Add episode to history"""
        episode.iteration = len(self.history) + 1
        self.history.append(episode)
        print(f"📝 Added episode {episode.iteration} to history")
    
    def generate(self, state: GraphState) -> GraphState:
        """
        Generate new parameters based on current state
        
        Args:
            state: GraphState containing:
                - current_episode: Episode object
                - expert_comment: Optional expert feedback
                
        Returns:
            Updated GraphState with generated parameters
        """
        print("=" * 60)
        print("🎯 LLM-EPIPARAM GENERATOR AGENT")
        print("=" * 60)
        
        # Extract data from state
        current_episode = state.get('current_episode')
        expert_comment = state.get('expert_comment', "No expert comment provided.")
        
        if not current_episode:
            print("❌ No current episode in state")
            return state
        
        # Prepare prompt inputs
        prompt_inputs = {
            "task_config": self._format_task_config(),
            "current_episode": self._format_current_episode(current_episode),
            "history": self._format_history(self.history),
            "expert_comment": expert_comment,
            "stats_summary": self._format_stats_summary(self.history),
            "target_metrics": self._format_target_metrics()
        }
        
        # Create chain with retry mechanism
        chain = self.prompt | RunnableParallel(
            response=self.llm, 
            prompt=RunnablePassthrough()
        )
        
        try:
            print("🚀 Generating new parameters...")
            result = chain.invoke(prompt_inputs)
            
            # Parse the response
            raw_response = result.get('response', {}).content if hasattr(result.get('response', {}), 'content') else str(result)
            
            # Use parser to validate
            parsed_output = self.parser.parse(raw_response)
            
            print(f"✅ Generated: β={parsed_output.beta:.4f}, γ={parsed_output.gamma:.4f}, μ={parsed_output.mu:.5f}")
            print(f"💭 Reasoning: {parsed_output.reasoning[:100]}...")
            print(f"🎯 Expected peak: day {parsed_output.expected_peak_position:.1f}")
            
            # Store generated parameters in state
            state.set('generated_params', {
                'reasoning': parsed_output.reasoning,
                'beta': parsed_output.beta,
                'gamma': parsed_output.gamma,
                'mu': parsed_output.mu,
                'expected_peak_position': parsed_output.expected_peak_position,
                'expected_peak_height': parsed_output.expected_peak_height,
                'expected_total_deaths': parsed_output.expected_total_deaths,
                'confidence': parsed_output.confidence
            })
            
            return state
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            state.set('generated_params', None)
            state.set('generation_error', str(e))
            return state
    
    def __call__(self, state: GraphState) -> GraphState:
        """Call the agent"""
        return self.generate(state)
