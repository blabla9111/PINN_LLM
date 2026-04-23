from formats.data_formats import PipelineState
from agents.EpiParamGeneratorAgent import LLMEpiParamGenerator
from agents.DeterministicCriticAgent import DeterministicCriticAgent

from agents.PINN_const import EINN_PINN
from agents.PINNAgent import PINNAgent

from typing import Dict, Optional
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
        

class PINNNode:
    """Node wrapper for PINN Agent in LangGraph"""
    
    def __init__(self, pinn_agent: PINNAgent):
        self.pinn_agent = pinn_agent
    
    def __call__(self, state: PipelineState) -> PipelineState:
        """Execute PINN agent"""
        print("=" * 60)
        print("🧠 PINN NODE")
        print("=" * 60)
        
        # Проверяем, нужно ли запускать PINN
        # Запускаем только если параметры были приняты критиком
        decision = state.get('critic_decision', 'reject')
        
        if decision == 'accept':
            print("✅ Parameters accepted, running PINN...")
            state = self.pinn_agent(state)
        else:
            print(f"⏭️ Parameters rejected, skipping PINN")
            state['pinn_results'] = {
                'success': False,
                'skipped': True,
                'reason': f'Parameters rejected by critic (decision: {decision})'
            }
        
        return state

# ============================================
# LangGraph Pipeline
# ============================================

class OptimizationPipeline:
    """
    Complete optimization pipeline using LangGraph
    """
    
    def __init__(
        self, 
        llm: Union[BaseLLMClient, object], 
        surrogate_agent=None, 
        initial_conditions=None,
        pinn_agent: Optional[PINNAgent] = None,  # Новый параметр
        use_pinn: bool = True  # Флаг для включения/отключения PINN
    ):
        self.llm = llm
        self.is_base_client = isinstance(llm, BaseLLMClient)
        self.use_pinn = use_pinn
        
        # Создаем суррогатного агента
        if surrogate_agent is None:
            from agents.SurrogateModel import SurrogateAgent
            self.surrogate_agent = SurrogateAgent(verbose=True)
        else:
            self.surrogate_agent = surrogate_agent
        
        # Создаем или используем переданный PINN агент
        if use_pinn:
            if pinn_agent is None:
                # Создаем PINN агент с параметрами по умолчанию
                self.pinn_agent = PINNAgent(
                    pinn_class=EINN_PINN,
                    n_epoch=10_000,
                    lambda_data=0.01,
                    lambda_ode=1.0,
                    results_dir="PINN_agent_results",
                    verbose=True
                )
            else:
                self.pinn_agent = pinn_agent
            self.pinn_node = PINNNode(self.pinn_agent)
        else:
            self.pinn_agent = None
            self.pinn_node = None
        
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
        self.critic = DeterministicCriticAgent(
                                        llm_for_agents, 
                                        enable_logging=True, 
                                        log_format="json",
                                        max_retries=3
                                    )
        
        # Pass both generator and critic to history node
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
        """Build LangGraph workflow - PINN runs after history, then ends"""
        
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("generate", self.generator.generate)
        workflow.add_node("surrogate", self.surrogate_node)
        workflow.add_node("critic", self.critic)
        workflow.add_node("history", self.history_node)
        
        if self.use_pinn and self.pinn_node:
            workflow.add_node("pinn", self.pinn_node)
        
        # Add edges
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "surrogate")
        workflow.add_edge("surrogate", "critic")
        workflow.add_edge("critic", "history")
        
        # После history: либо PINN (если accept), либо decision (если reject)
        if self.use_pinn and self.pinn_node:
            def route_after_history(state: PipelineState) -> str:
                decision = state.get('critic_decision', 'reject')
                iteration = state.get('iteration', 0)
                max_iterations = state.get('max_iterations', 10)
                
                print(f"\n🔄 ROUTING after history (iter {iteration}):")
                print(f"   Decision: {decision}")
                
                if decision == 'accept':
                    print(f"   ➡️  ACCEPTED → Running PINN validation, then END")
                    return "pinn"
                elif iteration >= max_iterations:
                    print(f"   ⏹️  Max iterations reached → END")
                    return "end"
                else:
                    print(f"   ➡️  REJECTED → Continue to next iteration")
                    state['iteration'] = iteration + 1
                    return "continue"
            
            workflow.add_conditional_edges(
                "history",
                route_after_history,
                {
                    "pinn": "pinn",
                    "continue": "generate",
                    "end": END
                }
            )
            
            # После PINN всегда завершаем работу
            workflow.add_edge("pinn", END)
            
        else:
            # Без PINN: стандартная логика
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
    beta: float,
    gamma: float,
    mu: float,
    expert_comment: str = None,
    max_iterations: int = 10,
    population: int = 10000,
    S0: int = 9999,
    I0: int = 1,
    R0: int = 0,
    D0: int = 0,
    t_max: int = 200,
    num_points: int = 1000,
    pinn_data: Dict = None,  # опционально для PINN
) -> Dict:
        """
        Run the optimization pipeline starting from baseline parameters
        
        Args:
            beta: Infection rate
            gamma: Recovery rate  
            mu: Mortality rate
            expert_comment: Expert guidance for optimization
            max_iterations: Maximum optimization iterations
            population, S0, I0, R0, D0: Initial conditions
            t_max, num_points: Simulation parameters
            pinn_data: Optional real data for PINN validation
        
        Returns:
            Final pipeline state
        """
        print("\n" + "=" * 60)
        print("🚀 STARTING OPTIMIZATION PIPELINE")
        print("=" * 60)
        print(f"📊 Baseline parameters: β={beta:.4f}, γ={gamma:.4f}, μ={mu:.5f}")
        if expert_comment:
            print(f"💬 Expert comment: {expert_comment}")
        
        # ============================================================
        # ШАГ 1: Получаем baseline через суррогат
        # ============================================================
        print("\n" + "=" * 60)
        print("📏 STEP 1: Computing baseline via surrogate")
        print("=" * 60)
        
        # Создаем временный state для baseline
        baseline_state = {
            'task_config': {
                'population': population,
                'S0': S0,
                'I0': I0,
                'R0': R0,
                'D0': D0,
                't_max': t_max,
                'num_points': num_points,
            },
            'generated_params': {'beta': beta, 'gamma': gamma, 'mu': mu},
            'initial_conditions': {
                'population': population,
                'S0': S0,
                'I0': I0,
                'R0': R0,
                'D0': D0
            }
        }
        
        # Запускаем суррогат
        baseline_state = self.surrogate_node(baseline_state)
        surrogate_results = baseline_state.get('surrogate_results', {})
        
        if not surrogate_results.get('success', False):
            raise RuntimeError(f"Failed to compute baseline: {surrogate_results.get('error')}")
        
        baseline_peak_position = surrogate_results['peak_position']
        baseline_peak_height = surrogate_results['peak_height']
        baseline_total_deaths = surrogate_results['total_deaths']
        
        print(f"\n✅ Baseline computed:")
        print(f"   Peak position: {baseline_peak_position:.1f} days")
        print(f"   Peak height: {baseline_peak_height:.0f} infected")
        print(f"   Total deaths: {baseline_total_deaths:.0f}")
        
        # ============================================================
        # ШАГ 2: Создаем начальный эпизод с baseline
        # ============================================================
        from formats.data_formats import Episode
        
        baseline_episode = Episode(
            beta=beta,
            gamma=gamma,
            mu=mu,
            peak_position=baseline_peak_position,
            peak_height=baseline_peak_height,
            total_deaths=baseline_total_deaths,
            iteration=0,
            expert_comment=f"BASELINE: {expert_comment if expert_comment else 'Initial parameters'}",
            accepted=True,  # Baseline всегда accepted
            reasoning="Baseline parameters from initial input"
        )
        
        # ============================================================
        # ШАГ 3: Инициализируем агентов с baseline
        # ============================================================
        self.critic.history = [baseline_episode]
        self.generator.history = [baseline_episode]
        print(f"\n📚 Initialized agents with baseline episode")
        
        # ============================================================
        # ШАГ 4: Формируем task_config для оптимизации
        # ============================================================
        task_config = {
            'description': f'Optimization from baseline: β={beta}, γ={gamma}, μ={mu}',
            'baseline_peak': baseline_peak_position,
            'baseline_height': baseline_peak_height,
            'baseline_deaths': baseline_total_deaths,
            'peak_tolerance': 5.0,
            'population': population,
            'S0': S0,
            'I0': I0,
            'R0': R0,
            'D0': D0,
            't_max': t_max,
            'num_points': num_points,
        }
        
        # Добавляем PINN данные если есть
        if pinn_data:
            task_config['pinn_data'] = pinn_data
        
        # ============================================================
        # ШАГ 5: Запускаем оптимизацию
        # ============================================================
        print("\n" + "=" * 60)
        print("🎯 STEP 2: Running optimization")
        print("=" * 60)
        
        self.critic.set_task_config(task_config)
        
        # Начальное состояние для оптимизации
        initial_state: PipelineState = {
            'task_config': task_config,
            'current_episode': baseline_episode,
            'expert_comment': expert_comment,
            'history': [baseline_episode],
            'generated_params': {'beta': beta, 'gamma': gamma, 'mu': mu},
            'surrogate_results': surrogate_results,
            'critic_decision': None,
            'critic_reasoning': None,
            'final_episode': None,
            'iteration': 0,
            'max_iterations': max_iterations,
            'should_continue': True,
            'initial_conditions': {
                'population': population,
                'S0': S0,
                'I0': I0,
                'R0': R0,
                'D0': D0
            },
            'simulation_params': {
                't_max': t_max,
                'num_points': num_points
            }
        }
        
        # Запускаем граф
        config = {"configurable": {"thread_id": "optimization_1"}}
        final_state = self.graph.invoke(initial_state, config)
        
        # ============================================================
        # ШАГ 6: Вывод результатов
        # ============================================================
        print("\n" + "=" * 60)
        print("🏁 PIPELINE COMPLETE")
        print("=" * 60)
        
        history = final_state.get('history', [])
        
        # Сравнение с baseline
        print(f"\n📊 Baseline (iteration 0):")
        print(f"   β={baseline_episode.beta:.4f}, γ={baseline_episode.gamma:.4f}, μ={baseline_episode.mu:.5f}")
        print(f"   Peak: {baseline_episode.peak_position:.1f} days, {baseline_episode.peak_height:.0f} infected")
        print(f"   Deaths: {baseline_episode.total_deaths:.0f}")
        
        # Лучший принятый эпизод
        accepted_episodes = [ep for ep in history if ep.accepted and ep.iteration > 0]
        if accepted_episodes:
            best = accepted_episodes[-1]
            print(f"\n✅ Best optimized (iteration {best.iteration}):")
            print(f"   β={best.beta:.4f} (Δ={best.beta - baseline_episode.beta:+.4f})")
            print(f"   γ={best.gamma:.4f} (Δ={best.gamma - baseline_episode.gamma:+.4f})")
            print(f"   μ={best.mu:.5f} (Δ={best.mu - baseline_episode.mu:+.5f})")
            print(f"   Peak: {best.peak_position:.1f} days (Δ={best.peak_position - baseline_episode.peak_position:+.1f})")
            print(f"   Deaths: {best.total_deaths:.0f} (Δ={best.total_deaths - baseline_episode.total_deaths:+.0f})")
            if best.reasoning:
                print(f"   Reasoning: {best.reasoning}")
        
        # ============================================================
        # ШАГ 6: Вывод результатов
        # ============================================================
        print("\n" + "=" * 60)
        print("🏁 PIPELINE COMPLETE")
        print("=" * 60)
        
        history = final_state.get('history', [])
        
        # ============================================================
        # ДЕТАЛЬНЫЙ ВЫВОД BASELINE
        # ============================================================
        print(f"\n📊 BASELINE (Initial Parameters):")
        print(f"   β={baseline_episode.beta:.4f}")
        print(f"   γ={baseline_episode.gamma:.4f}")
        print(f"   μ={baseline_episode.mu:.5f}")
        print(f"   📈 Peak position: {baseline_episode.peak_position:.1f} days")
        print(f"   📊 Peak height: {baseline_episode.peak_height:.0f} infected")
        print(f"   💀 Total deaths: {baseline_episode.total_deaths:.0f}")
        
        # ============================================================
        # ЛУЧШИЙ ОПТИМИЗИРОВАННЫЙ РЕЗУЛЬТАТ
        # ============================================================
        accepted_episodes = [ep for ep in history if ep.accepted and ep.iteration > 0]
        if accepted_episodes:
            best = accepted_episodes[-1]
            print(f"\n✅ BEST OPTIMIZED (Iteration {best.iteration}):")
            print(f"   β={best.beta:.4f} (Δ={best.beta - baseline_episode.beta:+.4f})")
            print(f"   γ={best.gamma:.4f} (Δ={best.gamma - baseline_episode.gamma:+.4f})")
            print(f"   μ={best.mu:.5f} (Δ={best.mu - baseline_episode.mu:+.5f})")
            print(f"   📈 Peak position: {best.peak_position:.1f} days (Δ={best.peak_position - baseline_episode.peak_position:+.1f})")
            print(f"   📊 Peak height: {best.peak_height:.0f} infected (Δ={best.peak_height - baseline_episode.peak_height:+.0f})")
            print(f"   💀 Total deaths: {best.total_deaths:.0f} (Δ={best.total_deaths - baseline_episode.total_deaths:+.0f})")
            if best.reasoning:
                print(f"   💭 Reasoning: {best.reasoning}")
        else:
            print(f"\n⚠️ No episodes were accepted during optimization")
        
        # ============================================================
        # ПОЛНАЯ ИСТОРИЯ ВСЕХ ИТЕРАЦИЙ
        # ============================================================
        print(f"\n📈 COMPLETE HISTORY ({len(history)} episodes):")
        print("-" * 100)
        print(f"{'Status':<6} {'Iter':<5} {'β':<8} {'γ':<8} {'μ':<10} {'Peak day':<10} {'Peak height':<12} {'Deaths':<10}")
        print("-" * 100)
        
        for ep in history:
            status = "✅" if ep.accepted else "❌"
            baseline_marker = " (BASELINE)" if ep.iteration == 0 else ""
            
            print(f"{status:<6} {ep.iteration:<5} "
                f"{ep.beta:<8.4f} {ep.gamma:<8.4f} {ep.mu:<10.5f} "
                f"{ep.peak_position:<10.1f} {ep.peak_height:<12.0f} {ep.total_deaths:<10.0f}"
                f"{baseline_marker}")
        
        print("-" * 100)
        
        # ============================================================
        # СТАТИСТИКА ПО ИТЕРАЦИЯМ
        # ============================================================
        if len(history) > 1:
            print(f"\n📊 STATISTICS:")
            print(f"   Total iterations: {len(history)}")
            print(f"   Accepted: {len([ep for ep in history if ep.accepted])}")
            print(f"   Rejected: {len([ep for ep in history if not ep.accepted])}")
            
            # Анализ тренда
            peaks = [ep.peak_position for ep in history if ep.iteration > 0]
            heights = [ep.peak_height for ep in history if ep.iteration > 0]
            deaths = [ep.total_deaths for ep in history if ep.iteration > 0]
            
            if peaks:
                print(f"\n📈 TRENDS (from baseline):")
                print(f"   Peak position: {baseline_episode.peak_position:.1f} → {peaks[-1]:.1f} days (Δ={peaks[-1] - baseline_episode.peak_position:+.1f})")
                print(f"   Peak height: {baseline_episode.peak_height:.0f} → {heights[-1]:.0f} infected (Δ={heights[-1] - baseline_episode.peak_height:+.0f})")
                print(f"   Total deaths: {baseline_episode.total_deaths:.0f} → {deaths[-1]:.0f} (Δ={deaths[-1] - baseline_episode.total_deaths:+.0f})")
        
        # ============================================================
        # РЕЗУЛЬТАТЫ PINN (если есть)
        # ============================================================
        pinn_results = final_state.get('pinn_results')
        if pinn_results and pinn_results.get('success'):
            print(f"\n🧠 PINN VALIDATION RESULTS:")
            fp = pinn_results.get('final_params', {})
            print(f"   Estimated parameters:")
            print(f"      β={fp.get('beta', 0):.4f}")
            print(f"      γ={fp.get('gamma', 0):.4f}")
            print(f"      μ={fp.get('mu', 0):.5f}")
            
            # Сравнение с оптимизированными параметрами
            if accepted_episodes:
                best = accepted_episodes[-1]
                print(f"\n   Difference from optimized:")
                print(f"      Δβ={fp.get('beta', 0) - best.beta:+.4f}")
                print(f"      Δγ={fp.get('gamma', 0) - best.gamma:+.4f}")
                print(f"      Δμ={fp.get('mu', 0) - best.mu:+.5f}")
            
            if pinn_results.get('plot_paths'):
                print(f"\n   📁 Plots saved: {len(pinn_results['plot_paths'])} files")
        
        return final_state


# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    from agents.LLMFactory import LLMFactory
    import pandas as pd
    
    # Initialize LLM
    llm_client = LLMFactory.from_config(config.LLM_CONFIG)
    print(f"🚀 Using LLM: {llm_client.get_model_info()}")
    
    # Загружаем данные для PINN (опционально)
    covid_cases = pd.read_csv('../../NEW_PINN/real_datasets/covid-19_Kouprianov.csv')
    
    pinn_data = {
        'S': covid_cases['S'].tolist(),
        'I': covid_cases['I'].tolist(),
        'R': covid_cases['R'].tolist(),
        'D': covid_cases['D'].tolist(),
        'train_size': 150,
    }
    
    # Создаем PINN агент (опционально)
    pinn_agent = PINNAgent(
        pinn_class=EINN_PINN,
        n_epoch=5_000,
        lambda_data=1.0,
        lambda_ode=0.1,
        results_dir="PINN_agent_results",
        verbose=True
    )
    
    # Создаем пайплайн
    pipeline = OptimizationPipeline(
        llm=llm_client,
        pinn_agent=pinn_agent,
        use_pinn=True
    )
    
    # ============================================================
    # ПРОСТОЙ ЗАПУСК - только параметры и комментарий
    # ============================================================
    result = pipeline.run(
        beta=0.1,
        gamma=0.042,
        mu=0.0025,
        expert_comment="Need higher peak",
        max_iterations=100,
        # Опциональные параметры (если нужны нестандартные)
        population=1000,
        S0=999,
        I0=1,
        t_max=400,
        pinn_data=pinn_data  # если нужен PINN
    )