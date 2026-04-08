from formats.data_formats import PipelineState
from agents.EpiParamGeneratorAgent import LLMEpiParamGenerator
# from agents.ParameterCriticAgent import ParameterCriticAgent
from agents.ReActCriticAgent import DeterministicCriticAgent

from typing import Dict
from agents.BaseLLMClient import BaseLLMClient  
from typing import Union 

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import config

class SurrogateNode:
    """Node for surrogate model evaluation"""
    
    def __init__(self, surrogate_agent):
        self.surrogate_agent = surrogate_agent
    
    def __call__(self, state: PipelineState) -> PipelineState:
        print("=" * 60)
        print("📊 SURROGATE MODEL")
        print("=" * 60)
        
        # Получаем сгенерированные параметры
        params = state.get('generated_params', {})
        if not params:
            print("❌ No parameters to evaluate")
            return state
        
        # Добавляем начальные условия в state, если их нет
        if 'initial_conditions' not in state:
            # Используем значения по умолчанию из config или из task_config
            task_config = state.get('task_config', {})
            state['initial_conditions'] = {
                'population': task_config.get('population', 10_000),
                'S0': task_config.get('S0', 9_999),
                'I0': task_config.get('I0', 1),
                'R0': task_config.get('R0', 0),
                'D0': task_config.get('D0', 0)
            }
        
        # Вызываем суррогатного агента
        state = self.surrogate_agent(state)
        
        # Результаты уже в state['surrogate_results']
        results = state.get('surrogate_results', {})
        
        if results.get('success', False):
            print(f"✅ Peak: {results['peak_position']:.1f} days")
            print(f"✅ Deaths: {results['total_deaths']:.0f}")
        else:
            print(f"❌ Simulation failed: {results.get('error', 'Unknown error')}")
        
        return state
    
class HistoryNode:
    """Node for managing history"""
    
    def __init__(self, generator=None, critic=None):
        """
        Initialize history node
        
        Args:
            generator: Generator agent to update its history
            critic: Critic agent to get history from
        """
        self.generator = generator
        self.critic = critic
    
    def __call__(self, state: PipelineState) -> PipelineState:
        print("=" * 60)
        print("📝 HISTORY MANAGER")
        print("=" * 60)
        
        # Get current iteration
        current_iteration = state.get('iteration', 0)
        
        # Get generated parameters and results
        generated_params = state.get('generated_params', {})
        surrogate_results = state.get('surrogate_results', {})
        critic_decision = state.get('critic_decision', 'reject')
        critic_reasoning = state.get('critic_reasoning', '')
        
        # Create Episode object with all available information
        from formats.data_formats import Episode
        
        episode = Episode(
            beta=generated_params.get('beta', 0.0),
            gamma=generated_params.get('gamma', 0.0),
            mu=generated_params.get('mu', 0.0),
            # Optional fields from surrogate results
            peak_position=surrogate_results.get('peak_position', 0.0),
            peak_height=surrogate_results.get('peak_height', 0.0),
            total_deaths=surrogate_results.get('total_deaths', 0.0),
            # Additional metadata
            iteration=current_iteration,
            expert_comment=state.get('expert_comment'),
            accepted=(critic_decision == 'accept'),
            reasoning=critic_reasoning
        )
        
        # Add to history in state
        history = state.get('history', [])
        history.append(episode)
        state['history'] = history
        
        # ✅ Update critic's history (critic stores episodes with evaluation results)
        if self.critic:
            # Проверяем, нет ли уже такого эпизода
            if not any(e.iteration == episode.iteration for e in self.critic.history):
                self.critic.add_to_history(episode)
        
        # ✅ Update generator's history (for context in next generations)
        if self.generator:
            # Generator needs history of all previous episodes for context
            # But doesn't need to add episodes itself - just reference critic's history
            # Synchronize generator's history with critic's history
            self.generator.history = self.critic.history
            print(f"✅ Synchronized generator history ({len(self.critic.history)} episodes)")
        
        # Always update current_episode for next iteration
        state['current_episode'] = episode
        
        if critic_decision == 'accept':
            print(f"✅ Episode {current_iteration} ACCEPTED and added to history")
        else:
            print(f"❌ Episode {current_iteration} REJECTED and added to history")
        
        print(f"📊 Total history: {len(history)} episodes")
        print(f"   - Accepted: {len([ep for ep in history if ep.accepted])}")
        print(f"   - Rejected: {len([ep for ep in history if not ep.accepted])}")
        
        return state


class DecisionNode:
    """Decision node for controlling the loop"""
    
    def __call__(self, state: PipelineState) -> str:
        print("=" * 60)
        print("⚖️ DECISION NODE")
        print("=" * 60)
        
        decision = state.get('critic_decision', 'reject')
        iteration = state.get('iteration', 0)
        max_iterations = state.get('max_iterations', 10)
        
        print(f"Iteration: {iteration}/{max_iterations}")
        print(f"Decision: {decision}")
        
        # Continue conditions
        if decision == 'accept':
            print("✅ Parameters accepted - ending loop")
            return "end"
        elif iteration >= max_iterations:
            print(f"⚠️ Max iterations ({max_iterations}) reached - ending loop")
            return "end"
        else:
            # 🔧 FIX: Increment iteration for the next loop
            state['iteration'] = iteration + 1
            print(f"🔄 Continuing loop with new generation (iteration {iteration + 1})")
            return "continue"
        

# ============================================
# LangGraph Pipeline
# ============================================

class OptimizationPipeline:
    """
    Complete optimization pipeline using LangGraph
    """
    
    def __init__(self, llm: Union[BaseLLMClient, object], surrogate_agent=None, initial_conditions=None):
        self.llm = llm
        self.is_base_client = isinstance(llm, BaseLLMClient)  
        
        # Создаем суррогатного агента
        if surrogate_agent is None:
            from agents.SurrogateModel import SurrogateAgent
            self.surrogate_agent = SurrogateAgent(verbose=True)
        else:
            self.surrogate_agent = surrogate_agent
        
        # Сохраняем начальные условия для использования в pipeline
        self.initial_conditions = initial_conditions or {
            'population': 10_000,
            'S0': 9_999,
            'I0': 1,
            'R0': 0,
            'D0': 0
        }
        
        # Create agents

        llm_for_agents = self._get_llm_for_agents()

        self.generator = LLMEpiParamGenerator(llm_for_agents, enable_logging=True, log_format="json")
        # self.critic = ParameterCriticAgent(llm_for_agents, enable_logging=True, log_format="json")
        self.critic = DeterministicCriticAgent(
                                        llm_for_agents, 
                                        enable_logging=True, 
                                        log_format="json",
                                        max_retries=3
                                    )
        
        # ✅ Pass both generator and critic to history node
        self.history_node = HistoryNode(generator=self.generator, critic=self.critic)
        self.surrogate_node = SurrogateNode(self.surrogate_agent)
        self.decision_node = DecisionNode()
        self.memory = MemorySaver()
        
        # Build graph
        self.graph = self._build_graph()

    def _get_llm_for_agents(self):
        """Get LLM object compatible with agents (langchain format)"""
        if self.is_base_client:
            return self._create_langchain_compatible_llm()
        return self.llm
    
    def _create_langchain_compatible_llm(self):
        """Create wrapper for BaseLLMClient"""
        from langchain_core.language_models.llms import LLM
        from typing import Any, List, Mapping, Optional
        
        class LLMClientWrapper(LLM):
            client: BaseLLMClient
            
            @property
            def _llm_type(self) -> str:
                return "base_llm_client_wrapper"
            
            def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
                response = self.client.invoke(prompt)
                return response.content
            
            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                return {"model": self.client.model_name}
            
            @property
            def temperature(self):
                return self.client.temperature
            
            @temperature.setter
            def temperature(self, value):
                self.client.temperature = value
        
        return LLMClientWrapper(client=self.llm)
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        # Create graph with state
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("generate", self.generator.generate)
        workflow.add_node("surrogate", self.surrogate_node)
        workflow.add_node("critic", self.critic)
        workflow.add_node("history", self.history_node)
        
        # Add edges
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "surrogate")
        workflow.add_edge("surrogate", "critic")
        workflow.add_edge("critic", "history")
        
        # Add conditional edge
        workflow.add_conditional_edges(
            "history",
            self.decision_node,
            {
                "continue": "generate",
                "end": END
            }
        )
        
        return workflow.compile(checkpointer=self.memory)
    
    def run(
        self,
        task_config: Dict,
        initial_params: Dict,
        expert_comment: str = None,
        max_iterations: int = 10
    ) -> Dict:
        """
        Run the optimization pipeline
        """
        print("\n" + "=" * 60)
        print("🚀 STARTING OPTIMIZATION PIPELINE")
        print("=" * 60)
        
        # Convert initial_params dict to Episode object
        from formats.data_formats import Episode
        
        initial_episode = Episode(
            reasoning=initial_params.get('reasoning', None),
            beta=initial_params.get('beta', 0.0),
            gamma=initial_params.get('gamma', 0.0),
            mu=initial_params.get('mu', 0.0),
            peak_position=initial_params.get('peak_position', 0.0),
            peak_height=initial_params.get('peak_height', 0.0),
            total_deaths=initial_params.get('total_deaths', 0.0),
            iteration=initial_params.get('iteration', 0),
            expert_comment=expert_comment
        )
        
        # ✅ Initialize critic history with initial episode
        # This ensures the generator has context from the start
        if initial_episode:
            # Очищаем перед инициализацией
            self.critic.history = []
            # Присваиваем напрямую, без add_to_history
            self.critic.history = [initial_episode]
            self.generator.history = [initial_episode]
            print(f"📚 Initialized history with 1 episode (initial parameters)")
        
        # Initial state with proper structure
        initial_state: PipelineState = {
            'task_config': task_config,
            'current_episode': initial_episode,
            'expert_comment': expert_comment,
            'history': [initial_episode],  # Will be populated by history node
            'generated_params': {
                'beta': initial_params.get('beta', 0.0),
                'gamma': initial_params.get('gamma', 0.0),
                'mu': initial_params.get('mu', 0.0)
            },
            'surrogate_results': {
                'peak_position': initial_params.get('peak_position', 0.0),
                'peak_height': initial_params.get('peak_height', 0.0),
                'total_deaths': initial_params.get('total_deaths', 0.0)
            },
            'critic_decision': None,
            'critic_reasoning': None,
            # 'suggested_params': None,
            'final_episode': None,
            'iteration': 0,
            'max_iterations': max_iterations,
            'should_continue': True,
            'initial_conditions': {
                'population': task_config.get('population', self.initial_conditions['population']),
                'S0': task_config.get('S0', self.initial_conditions['S0']),
                'I0': task_config.get('I0', self.initial_conditions['I0']),
                'R0': task_config.get('R0', self.initial_conditions['R0']),
                'D0': task_config.get('D0', self.initial_conditions['D0'])
            },
            'simulation_params': {
                't_max': task_config.get('t_max', 200),
                'num_points': task_config.get('num_points', 1000)
            }
        }
        
        # Configure thread for checkpointing
        config = {"configurable": {"thread_id": "optimization_1"}}
        
        self.critic.set_task_config(task_config)
        
        # Run the graph
        final_state = self.graph.invoke(initial_state, config)
        
        print("\n" + "=" * 60)
        print("🏁 PIPELINE COMPLETE")
        print("=" * 60)
        
        # Get history from final state
        history = final_state.get('history', [])
        print(f"Total iterations: {len(history)}")
        
        # Filter accepted episodes
        accepted_episodes = [ep for ep in history if ep.accepted]
        
        # Display best parameters if any accepted episodes
        if accepted_episodes:
            best = accepted_episodes[-1]  # Last accepted episode
            print(f"\n✅ Best accepted episode:")
            print(f"   Parameters: β={best.beta:.4f}, γ={best.gamma:.4f}, μ={best.mu:.5f}")
            print(f"   Peak at day {best.peak_position:.1f}")
            print(f"   Total deaths: {best.total_deaths:.0f}")
            print(f"   Reasoning: {best.reasoning if best.reasoning else 'No reasoning provided'}")
        else:
            print("\n⚠️ No episodes were accepted during the run")
            
            # Show the last episode even if not accepted
            if history:
                last = history[-1]
                print(f"\n📊 Last attempted episode:")
                print(f"   Parameters: β={last.beta:.4f}, γ={last.gamma:.4f}, μ={last.mu:.5f}")
                print(f"   Peak at day {last.peak_position:.1f}")
                print(f"   Status: {'Accepted' if last.accepted else 'Rejected'}")
        
        # Show summary of all episodes
        print("\n📊 History Summary:")
        for i, ep in enumerate(history):
            status = "✅ ACCEPTED" if ep.accepted else "❌ REJECTED"
            print(f"   Episode {ep.iteration}: {status} | β={ep.beta:.4f}, γ={ep.gamma:.4f}, μ={ep.mu:.5f} | Peak={ep.peak_position:.1f} days, Height={ep.peak_height:.0f}")
        
        return final_state


# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    from agents.LLMFactory import LLMFactory
    
    # Initialize LLM
    llm_client = LLMFactory.from_config(config.LLM_CONFIG)

    print(f"🚀 Using LLM: {llm_client.get_model_info()}")
    
    # Create pipeline с BaseLLMClient
    pipeline = OptimizationPipeline(llm=llm_client)
    try:
        # Get graph with xray to show all details
        graph_png = pipeline.graph.get_graph(xray=True)
        
        # Draw as PNG bytes
        png_bytes = graph_png.draw_mermaid_png()
        
        # Save to file
        with open("SciResearch_agent.png", "wb") as f:
            f.write(png_bytes)
        
        print("✅ Pipeline visualization saved to SciResearch_agent.png")
        
    except Exception as e:
        print(f"❌ Error generating visualization: {e}")
        print("Make sure you have installed: pip install matplotlib pillow")
    
    # Configure task with initial conditions
    task_config = {
        'description': 'Optimization of SIRD model parameters',
        # 'target_peak': 30,
        'peak_tolerance': 5.0,
        # Начальные условия для симуляции
        'population': 1000,
        'S0': 999,
        'I0': 1,
        'R0': 0,
        'D0': 0,
        't_max': 400,
        'num_points': 1000
    }
    
    # Initial parameters
    initial_params = {
        'beta': 0.095,
        'gamma': 0.0436,
        'mu': 0.00242,
        'peak_position': 140,
        'peak_height': 165,
        'total_deaths': 8500,
        'iteration': 0
    }
    
    # Run pipeline
    result = pipeline.run(
        task_config=task_config,
        initial_params=initial_params,
        expert_comment="Need later andlower peak",
        max_iterations=10
    )