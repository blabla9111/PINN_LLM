# ============================================
# LLM-epiparam generator agent
# ============================================

from typing import Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from formats.data_formats import EpiParameters, Episode, PipelineState

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
    
    def _format_history(self, history: List, max_episodes: int = 5) -> str:
        """Format history episodes for prompt - handles both dict and Episode objects"""
        if not history:
            return "No previous episodes available."
        
        # Take only last N episodes to keep prompt manageable
        recent_history = history[-max_episodes:]
        
        formatted = ""
        for episode in recent_history:
            if isinstance(episode, dict):
                formatted += f"""
    Episode {episode.get('iteration', 'N/A')}:
    - Parameters: β={episode.get('beta', 0):.4f}, γ={episode.get('gamma', 0):.4f}, μ={episode.get('mu', 0):.5f}
    - Results: Peak at day {episode.get('peak_position', 0):.1f}, Deaths: {episode.get('total_deaths', 0):.0f}
    - Accepted: {episode.get('accepted', False)}
    - Reasoning: {episode.get('reasoning', 'No reasoning')[:100]}...
    """
            elif hasattr(episode, 'to_prompt_format'):
                formatted += episode.to_prompt_format() + "\n"
            else:
                formatted += f"""
    Episode {getattr(episode, 'iteration', 'N/A')}:
    - Parameters: β={getattr(episode, 'beta', 0):.4f}, γ={getattr(episode, 'gamma', 0):.4f}, μ={getattr(episode, 'mu', 0):.5f}
    - Results: Peak at day {getattr(episode, 'peak_position', 0):.1f}, Deaths: {getattr(episode, 'total_deaths', 0):.0f}
    - Accepted: {getattr(episode, 'accepted', False)}
    - Reasoning: {getattr(episode, 'reasoning', 'No reasoning')[:100]}...
    """
        
        return formatted
    
    def _format_stats_summary(self, history: List) -> str:
        """Generate statistical summary from history - handles both dict and Episode objects"""
        if not history:
            return "No statistics available yet."
        
        # Extract values handling both dict and object
        betas = []
        gammas = []
        mus = []
        peaks = []
        deaths = []
        
        for item in history:
            if isinstance(item, dict):
                if item.get('beta') is not None:
                    betas.append(item['beta'])
                    gammas.append(item['gamma'])
                    mus.append(item['mu'])
                if item.get('peak_position'):
                    peaks.append(item['peak_position'])
                if item.get('total_deaths'):
                    deaths.append(item['total_deaths'])
            elif hasattr(item, 'beta'):
                betas.append(item.beta)
                gammas.append(item.gamma)
                mus.append(item.mu)
                if item.peak_position:
                    peaks.append(item.peak_position)
                if item.total_deaths:
                    deaths.append(item.total_deaths)
        
        if not betas:
            return "No valid parameter data in history."
        
        target_peak = self.task_config.get('target_peak', 30)
        
        # Find best episode based on target
        best_episode = None
        best_error = float('inf')
        
        for item in history:
            peak = item.get('peak_position') if isinstance(item, dict) else (item.peak_position if hasattr(item, 'peak_position') else None)
            if peak:
                error = abs(peak - target_peak)
                if error < best_error:
                    best_error = error
                    best_episode = item
        
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
            if isinstance(best_episode, dict):
                summary += f"""
    - β={best_episode.get('beta', 0):.4f}, γ={best_episode.get('gamma', 0):.4f}, μ={best_episode.get('mu', 0):.5f}
    - Peak at day {best_episode.get('peak_position', 0):.1f} with {best_episode.get('peak_height', 0):.0f} infected
    - Total deaths: {best_episode.get('total_deaths', 0):.0f}
    """
            else:
                summary += f"""
    - β={best_episode.beta:.4f}, γ={best_episode.gamma:.4f}, μ={best_episode.mu:.5f}
    - Peak at day {best_episode.peak_position:.1f} with {best_episode.peak_height:.0f} infected
    - Total deaths: {best_episode.total_deaths:.0f}
    """
        
        return summary
    
    def _format_current_episode(self, episode) -> str:
        """Format current episode for prompt - handles both dict and Episode object"""
        if not episode:
            return "No current episode available."
        
        # Handle dictionary
        if isinstance(episode, dict):
            return f"""
    - β = {episode.get('beta', 0):.4f}
    - γ = {episode.get('gamma', 0):.4f}
    - μ = {episode.get('mu', 0):.5f}
    - Peak position: {episode.get('peak_position', 0):.1f} days
    - Peak height: {episode.get('peak_height', 0):.0f} infected
    - Total deaths: {episode.get('total_deaths', 0):.0f}
    """
        # Handle Episode object
        elif hasattr(episode, 'beta'):
            return f"""
    - β = {episode.beta:.4f}
    - γ = {episode.gamma:.4f}
    - μ = {episode.mu:.5f}
    - Peak position: {episode.peak_position:.1f} days
    - Peak height: {episode.peak_height:.0f} infected
    - Total deaths: {episode.total_deaths:.0f}
    """
        else:
            return f"Unknown episode format: {type(episode)}"
    
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
    
    def add_to_history(self, episode):
        """Add episode to history - handles both dict and Episode objects"""
        if isinstance(episode, dict):
            # Convert dict to Episode object if you want to maintain consistency
            from formats.data_formats import Episode
            episode_obj = Episode(
                beta=episode.get('beta', 0),
                gamma=episode.get('gamma', 0),
                mu=episode.get('mu', 0),
                peak_position=episode.get('peak_position', 0),
                peak_height=episode.get('peak_height', 0),
                total_deaths=episode.get('total_deaths', 0),
                expert_comment=episode.get('expert_comment'),
                accepted=episode.get('accepted', False),
                iteration=len(self.history) + 1,
                reasoning=episode.get('reasoning'),
                timestamp=episode.get('timestamp')
            )
            self.history.append(episode_obj)
        else:
            # It's already an Episode object
            episode.iteration = len(self.history) + 1
            self.history.append(episode)
        
        print(f"📝 Added episode {len(self.history)} to history")
    
    def generate(self, state: PipelineState) -> PipelineState:
        """
        Generate new parameters based on current state
        """
        print("=" * 60)
        print("🎯 LLM-EPIPARAM GENERATOR AGENT")
        print("=" * 60)
        
        # Debug: print current iteration
        current_iteration = state.get('iteration', 0)
        print(f"📍 Current iteration from state: {current_iteration}")
        
        # Extract data from state
        current_episode = state.get('current_episode')
        
        # Debug: print what current_episode looks like
        if current_episode:
            if isinstance(current_episode, dict):
                print(f"📋 Current episode: dict with beta={current_episode.get('beta', 'N/A')}")
            elif hasattr(current_episode, 'beta'):
                print(f"📋 Current episode: Episode object with beta={current_episode.beta}")
            else:
                print(f"📋 Current episode: {type(current_episode)}")
        
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
            state['generated_params'] = {
                'reasoning': parsed_output.reasoning,
                'beta': parsed_output.beta,
                'gamma': parsed_output.gamma,
                'mu': parsed_output.mu,
                'expected_peak_position': parsed_output.expected_peak_position,
                'expected_peak_height': parsed_output.expected_peak_height,
                'expected_total_deaths': parsed_output.expected_total_deaths,
                'confidence': parsed_output.confidence
            }

            state['iteration'] = state['iteration'] + 1
            # print(f'iter = {state['iteration']}')
            
            return state
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            state['generated_params'] = None
            state['generation_error'] = str(e)
            return state
    
    def __call__(self, state: PipelineState) -> PipelineState:
        """Call the agent"""
        return self.generate(state)
