"""
main_test.py - Тестирование оптимизационного пайплайна с PINN валидацией
Сравнивает baseline и оптимизированные параметры через PINN
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import torch

# Добавляем пути для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.LLMFactory import LLMFactory
from agents.PINN_const import EINN_PINN
from agents.PINNAgent import PINNAgent
import config
from formats.data_formats import PipelineState
from agents.EpiParamGeneratorAgent import LLMEpiParamGenerator
from agents.DeterministicCriticAgent import DeterministicCriticAgent

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



def visualize_pipeline(pipeline, filename="pipeline_graph.png"):
    """Визуализация графа пайплайна"""
    try:
        graph_png = pipeline.graph.get_graph(xray=True)
        png_bytes = graph_png.draw_mermaid_png()
        
        with open(filename, "wb") as f:
            f.write(png_bytes)
        
        print(f"✅ Pipeline visualization saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error generating visualization: {e}")
        print("   Make sure you have installed: pip install matplotlib pillow")
        return False


def run_pinn_comparison(
    baseline_params: dict,
    optimized_params: dict,
    pinn_data: dict,
    pipeline_pinn_results: dict = None,
    results_dir: str = "PINN_comparison_results",
    n_epoch: int = 5_000,
    lambda_data: float = 1.0,
    lambda_ode: float = 1.0,
    lambda_ic: float = 0.1,
    lambda_bc: float = 0.1
):
    """
    Сравнивает baseline и оптимизированные параметры через PINN.
    Обучает PINN ровно 2 раза: baseline и (если нет в pipeline) optimized.
    """
    import matplotlib.pyplot as plt
    import json
    import torch
    
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 80)
    print("🧠 PINN COMPARISON: Baseline vs Optimized Predictions")
    print("=" * 80)
    
    # Подготовка данных
    S = np.array(pinn_data['S'])
    I = np.array(pinn_data['I'])
    R = np.array(pinn_data['R'])
    D = np.array(pinn_data['D'])
    t = np.arange(len(S), dtype=float)
    population = float(S[0] + I[0] + R[0] + D[0])
    train_size = pinn_data.get('train_size', int(len(S) * 0.7))
    
    from agents.PINN_const import EINN_PINN
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ============================================================
    # 1. ОБУЧЕНИЕ BASELINE (1-й раз)
    # ============================================================
    print("\n" + "-" * 60)
    print("📊 STEP 1: Training PINN for BASELINE parameters")
    print("-" * 60)
    print(f"   β={baseline_params['beta']:.4f}, γ={baseline_params['gamma']:.4f}, μ={baseline_params['mu']:.5f}")
    
    baseline_model = EINN_PINN(
        t=t, S_data=S, I_data=I, R_data=R, D_data=D,
        population=population, train_size=train_size,
        device=device, init_params=baseline_params
    )
    baseline_model.train_model(
        n_epoch=n_epoch,
        lambda_data=lambda_data,
        lambda_ode=lambda_ode,
        lambda_ic=lambda_ic,
        lambda_bc=lambda_bc
    )
    S_pred_bl, I_pred_bl, R_pred_bl, D_pred_bl = baseline_model.predict(t)
    I_pred_bl = I_pred_bl.numpy()
    S_pred_bl_np = S_pred_bl.numpy()
    R_pred_bl_np = R_pred_bl.numpy()
    D_pred_bl_np = D_pred_bl.numpy()
    baseline_est = baseline_model.params.get_params_dict()
    baseline_losses = baseline_model.losses.copy()
    
    print(f"   ✅ Baseline PINN complete")
    
    # ============================================================
    # 2. ПОЛУЧЕНИЕ OPTIMIZED (из pipeline или обучение - 2-й раз)
    # ============================================================
    print("\n" + "-" * 60)
    print("📊 STEP 2: Getting OPTIMIZED predictions")
    print("-" * 60)
    
    if pipeline_pinn_results and pipeline_pinn_results.get('success'):
        preds = pipeline_pinn_results.get('predictions')
        if preds is not None:
            print("   ✅ Using predictions from pipeline")
            I_pred_opt = np.array(preds['I'])
            S_pred_opt_np = np.array(preds['S'])
            R_pred_opt_np = np.array(preds['R'])
            D_pred_opt_np = np.array(preds['D'])
            optimized_est = pipeline_pinn_results.get('estimated_params', optimized_params)
            optimized_losses = pipeline_pinn_results.get('losses', [])
        else:
            print("   ⚠️ No predictions in pipeline, training...")
            optimized_model = EINN_PINN(
                t=t, S_data=S, I_data=I, R_data=R, D_data=D,
                population=population, train_size=train_size,
                device=device, init_params=optimized_params
            )
            optimized_model.train_model(
                n_epoch=n_epoch,
                lambda_data=lambda_data,
                lambda_ode=lambda_ode,
                lambda_ic=lambda_ic,
                lambda_bc=lambda_bc
            )
            S_pred_opt, I_pred_opt, R_pred_opt, D_pred_opt = optimized_model.predict(t)
            I_pred_opt = I_pred_opt.numpy()
            S_pred_opt_np = S_pred_opt.numpy()
            R_pred_opt_np = R_pred_opt.numpy()
            D_pred_opt_np = D_pred_opt.numpy()
            optimized_est = optimized_model.params.get_params_dict()
            optimized_losses = optimized_model.losses.copy()
            print(f"   ✅ Optimized model trained")
    else:
        print(f"   🔧 Training optimized model...")
        print(f"   β={optimized_params['beta']:.4f}, γ={optimized_params['gamma']:.4f}, μ={optimized_params['mu']:.5f}")
        optimized_model = EINN_PINN(
            t=t, S_data=S, I_data=I, R_data=R, D_data=D,
            population=population, train_size=train_size,
            device=device, init_params=optimized_params
        )
        optimized_model.train_model(
            n_epoch=n_epoch,
            lambda_data=lambda_data,
            lambda_ode=lambda_ode,
            lambda_ic=lambda_ic,
            lambda_bc=lambda_bc
        )
        S_pred_opt, I_pred_opt, R_pred_opt, D_pred_opt = optimized_model.predict(t)
        I_pred_opt = I_pred_opt.numpy()
        S_pred_opt_np = S_pred_opt.numpy()
        R_pred_opt_np = R_pred_opt.numpy()
        D_pred_opt_np = D_pred_opt.numpy()
        optimized_est = optimized_model.params.get_params_dict()
        optimized_losses = optimized_model.losses.copy()
        print(f"   ✅ Optimized model trained")
    
    # ============================================================
    # 3. Анализ пиков в I
    # ============================================================
    print("\n" + "-" * 60)
    print("📊 STEP 3: Analyzing I(t) peaks")
    print("-" * 60)
    
    baseline_peak_idx = np.argmax(I_pred_bl)
    optimized_peak_idx = np.argmax(I_pred_opt)
    data_peak_idx = np.argmax(I)
    
    baseline_peak_day = t[baseline_peak_idx]
    baseline_peak_height = I_pred_bl[baseline_peak_idx]
    optimized_peak_day = t[optimized_peak_idx]
    optimized_peak_height = I_pred_opt[optimized_peak_idx]
    data_peak_day = t[data_peak_idx]
    data_peak_height = I[data_peak_idx]
    
    peak_day_change = optimized_peak_day - baseline_peak_day
    peak_height_change = optimized_peak_height - baseline_peak_height
    peak_day_error_change = abs(optimized_peak_day - data_peak_day) - abs(baseline_peak_day - data_peak_day)
    
    print(f"\n   PEAK ANALYSIS:")
    print(f"   {'':<20} {'Day':<10} {'Height':<12}")
    print(f"   {'-'*42}")
    print(f"   {'Real data':<20} {data_peak_day:<10.1f} {data_peak_height:<12.0f}")
    print(f"   {'Baseline PINN':<20} {baseline_peak_day:<10.1f} {baseline_peak_height:<12.0f}")
    print(f"   {'Optimized PINN':<20} {optimized_peak_day:<10.1f} {optimized_peak_height:<12.0f}")
    
    print(f"\n   CHANGES (Optimized - Baseline):")
    print(f"   Peak day: {peak_day_change:+.1f} days")
    print(f"   Peak height: {peak_height_change:+.0f} infected")
    print(f"   Error to real data (day): {peak_day_error_change:+.1f} days")
    
    # ============================================================
    # 4. Сравнение параметров
    # ============================================================
    print("\n" + "-" * 60)
    print("📊 STEP 4: Parameter comparison")
    print("-" * 60)
    
    print(f"\n   {'Parameter':<10} {'Baseline Input':<15} {'Baseline Est':<15} {'Optimized Input':<16} {'Optimized Est':<15}")
    print(f"   {'-'*75}")
    for param, name in [('beta', 'β'), ('gamma', 'γ'), ('mu', 'μ')]:
        print(f"   {name:<10} {baseline_params[param]:<15.4f} {baseline_est[param]:<15.4f} "
              f"{optimized_params[param]:<16.4f} {optimized_est[param]:<15.4f}")
    
    # ============================================================
    # 5. Графики
    # ============================================================
    print("\n" + "-" * 60)
    print("📊 STEP 5: Creating comparison plots")
    print("-" * 60)
    
    # FIGURE 1: Сравнение I(t)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # I(t)
    ax1 = axes[0, 0]
    ax1.plot(t, I, 'o', ms=2, color='gray', alpha=0.5, label='Real data')
    ax1.plot(t, I_pred_bl, '-', linewidth=2, color='blue', alpha=0.8, label='Baseline PINN')
    ax1.plot(t, I_pred_opt, '-', linewidth=2, color='red', alpha=0.8, label='Optimized PINN')
    ax1.axvline(train_size, color='gray', linestyle='--', alpha=0.5, label='Train/test split')
    ax1.scatter(baseline_peak_day, baseline_peak_height, color='blue', s=100, marker='v', edgecolors='black', zorder=5)
    ax1.scatter(optimized_peak_day, optimized_peak_height, color='red', s=100, marker='^', edgecolors='black', zorder=5)
    ax1.scatter(data_peak_day, data_peak_height, color='gray', s=100, marker='s', edgecolors='black', zorder=5)
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Infected (I)')
    ax1.set_title(f'I(t) Comparison\nPeak change: {peak_day_change:+.0f} days, {peak_height_change:+.0f} infected')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # I(t) log
    ax2 = axes[0, 1]
    ax2.semilogy(t, I + 1, 'o', ms=2, color='gray', alpha=0.5, label='Real data')
    ax2.semilogy(t, I_pred_bl + 1, '-', linewidth=2, color='blue', alpha=0.8, label='Baseline PINN')
    ax2.semilogy(t, I_pred_opt + 1, '-', linewidth=2, color='red', alpha=0.8, label='Optimized PINN')
    ax2.axvline(train_size, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Infected (I) - log scale')
    ax2.set_title('I(t) Comparison (Log Scale)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # S(t)
    ax3 = axes[1, 0]
    ax3.plot(t, S, 'o', ms=2, color='gray', alpha=0.5, label='Real data')
    ax3.plot(t, S_pred_bl_np, '-', linewidth=2, color='blue', alpha=0.8, label='Baseline PINN')
    ax3.plot(t, S_pred_opt_np, '-', linewidth=2, color='red', alpha=0.8, label='Optimized PINN')
    ax3.axvline(train_size, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Susceptible (S)')
    ax3.set_title('S(t) Comparison')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # D(t)
    ax4 = axes[1, 1]
    ax4.plot(t, D, 'o', ms=2, color='gray', alpha=0.5, label='Real data')
    ax4.plot(t, D_pred_bl_np, '-', linewidth=2, color='blue', alpha=0.8, label='Baseline PINN')
    ax4.plot(t, D_pred_opt_np, '-', linewidth=2, color='red', alpha=0.8, label='Optimized PINN')
    ax4.axvline(train_size, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Deceased (D)')
    ax4.set_title('D(t) Comparison')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(results_dir, f"comparison_{timestamp}.png")
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Comparison plot saved: {comparison_plot_path}")
    
    # FIGURE 2: Детальный анализ I(t)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, I, 'o', ms=2, color='gray', alpha=0.4, label='Real data')
    ax.plot(t, I_pred_bl, '-', linewidth=2.5, color='blue', alpha=0.7, label='Baseline PINN')
    ax.plot(t, I_pred_opt, '-', linewidth=2.5, color='red', alpha=0.7, label='Optimized PINN')
    ax.axvline(train_size, color='gray', linestyle='--', alpha=0.5)
    ax.scatter(baseline_peak_day, baseline_peak_height, color='blue', s=150, marker='v', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(optimized_peak_day, optimized_peak_height, color='red', s=150, marker='^', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(data_peak_day, data_peak_height, color='gray', s=150, marker='s', edgecolors='black', linewidth=2, zorder=5)
    # ax.annotate('', xy=(optimized_peak_day, optimized_peak_height), xytext=(baseline_peak_day, baseline_peak_height),
    #             arrowprops=dict(arrowstyle='->', color='green', lw=2))
    mid_x = (baseline_peak_day + optimized_peak_day) / 2
    mid_y = (baseline_peak_height + optimized_peak_height) / 2
    ax.text(0.98, 0.98, 
        f'Δ day: {peak_day_change:+.0f} days\nΔ height: {peak_height_change:+.0f} infected', 
        fontsize=11, ha='right', va='top', 
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', alpha=0.9))
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Infected (I)', fontsize=12)
    ax.set_title(f'Peak Analysis: Baseline → Optimized', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    peak_analysis_path = os.path.join(results_dir, f"peak_analysis_{timestamp}.png")
    plt.savefig(peak_analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Peak analysis plot saved: {peak_analysis_path}")
    
     # ============================================================
    # 6. Сохранение результатов
    # ============================================================
    results = {
        'timestamp': timestamp,
        'baseline_params_input': baseline_params,
        'optimized_params_input': optimized_params,
        'baseline_pinn_estimated': baseline_est,
        'optimized_pinn_estimated': optimized_est,
        'baseline_pinn': {'success': True, 'losses': baseline_losses},
        'optimized_pinn': {'success': True, 'losses': optimized_losses},
        'peak_analysis': {
            'real_data': {'day': float(data_peak_day), 'height': float(data_peak_height)},
            'baseline': {'day': float(baseline_peak_day), 'height': float(baseline_peak_height)},
            'optimized': {'day': float(optimized_peak_day), 'height': float(optimized_peak_height)},
            'changes': {
                'peak_day': float(peak_day_change),
                'peak_height': float(peak_height_change),
                'error_to_real_day': float(peak_day_error_change)
            }
        },
        'plots': [comparison_plot_path, peak_analysis_path]
    }
    
    json_path = os.path.join(results_dir, f"comparison_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✅ Results saved: {json_path}")
    
    # ============================================================
    # 7. Итоговый вывод
    # ============================================================
    print("\n" + "=" * 80)
    print("📋 PINN VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n   Baseline peak:  day {baseline_peak_day:.0f}, height {baseline_peak_height:.0f}")
    print(f"   Optimized peak: day {optimized_peak_day:.0f}, height {optimized_peak_height:.0f}")
    print(f"   Real data peak: day {data_peak_day:.0f}, height {data_peak_height:.0f}")
    print(f"\n   Peak day change: {peak_day_change:+.0f} days")
    print(f"   Peak height change: {peak_height_change:+.0f} infected")

    results['t_data'] = t.tolist()
    results['I_real'] = I.tolist()
    results['i_predictions'] = {
        'baseline': I_pred_bl.tolist(),
        'optimized': I_pred_opt.tolist()
    }
    results['train_size'] = train_size
    
    return results


def main():
    """Основная функция тестирования"""
    
    print("=" * 80)
    print("🚀 OPTIMIZATION PIPELINE TEST WITH PINN VALIDATION")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ============================================================
    # 1. Инициализация LLM
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 1: Initializing LLM")
    print("-" * 60)
    
    llm_client = LLMFactory.from_config(config.LLM_CONFIG)
    print(f"   ✅ LLM initialized: {llm_client.get_model_info()}")
    
    # ============================================================
    # 2. Загрузка данных для PINN
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 2: Loading PINN data")
    print("-" * 60)
    
    covid_cases = pd.read_csv('../../NEW_PINN/real_datasets/covid-19_Kouprianov.csv')
    print(f"   ✅ Loaded {len(covid_cases)} data points")
    
    pinn_data = {
        'S': covid_cases['S'].tolist(),
        'I': covid_cases['I'].tolist(),
        'R': covid_cases['R'].tolist(),
        'D': covid_cases['D'].tolist(),
        'train_size': 180,
    }
    
    # ============================================================
    # 3. Создание PINN агента
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 3: Creating PINN agent")
    print("-" * 60)

    # ЕДИНЫЕ ПАРАМЕТРЫ ОБУЧЕНИЯ
    n_epoch = 10_000
    lambda_data = 1.0
    lambda_ode = 0.1
    lambda_ic = 0.1
    lambda_bc = 0.1

    pinn_agent = PINNAgent(
        pinn_class=EINN_PINN,
        n_epoch=n_epoch,
        lambda_data=lambda_data,
        lambda_ode=lambda_ode,
        lambda_ic=lambda_ic,
        lambda_bc=lambda_bc,
        results_dir="PINN_agent_results",
        verbose=True
    )
    print(f"   ✅ PINN agent created (device: {pinn_agent.device})")
    print(f"   Training config: epochs={n_epoch}, λ_data={lambda_data}, λ_ode={lambda_ode}")
    
    # ============================================================
    # 4. Создание пайплайна
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 4: Creating optimization pipeline")
    print("-" * 60)
    
    pipeline = OptimizationPipeline(
        llm=llm_client,
        pinn_agent=pinn_agent,
        use_pinn=True
    )
    print("   ✅ Pipeline created")
    
    # ============================================================
    # 5. Визуализация пайплайна
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 5: Visualizing pipeline")
    print("-" * 60)
    
    visualize_pipeline(pipeline, "pipeline_graph.png")
    
    # ============================================================
    # 6. Параметры для тестирования
    # ============================================================
    baseline_beta = 0.045727
    baseline_gamma = 0.034483
    baseline_mu = 0.005296
    
    expert_comment = "Need higher peak"

    print("\n" + "-" * 60)
    print("📌 STEP 6: Test parameters")
    print("-" * 60)
    print(f"   Baseline: β={baseline_beta:.4f}, γ={baseline_gamma:.4f}, μ={baseline_mu:.5f}")
    print(f"   Expert comment: {expert_comment}")
    print(f"   Max iterations: 5")
    
    # ============================================================
    # 7. Запуск оптимизации
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 7: Running optimization pipeline")
    print("-" * 60)
    
    result = pipeline.run(
        beta=baseline_beta,
        gamma=baseline_gamma,
        mu=baseline_mu,
        expert_comment=expert_comment,
        max_iterations=5,
        population=1000,
        S0=999,
        I0=1,
        t_max=400,
        pinn_data=pinn_data
    )
    
    # ============================================================
    # 8. Извлечение результатов
    # ============================================================
    history = result.get('history', [])
    baseline_episode = next((ep for ep in history if ep.iteration == 0), None)
    
    accepted_episodes = [ep for ep in history if ep.accepted and ep.iteration > 0]
    optimized_episode = accepted_episodes[-1] if accepted_episodes else None
    
    if not baseline_episode:
        print("❌ Baseline episode not found!")
        return
    
    if not optimized_episode:
        print("⚠️ No optimized episode accepted, using last episode")
        optimized_episode = history[-1]
    
    # ============================================================
    # 9. Сравнение через PINN
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 8: Running PINN comparison")
    print("-" * 60)
    
    baseline_params = {
        'beta': baseline_episode.beta,
        'gamma': baseline_episode.gamma,
        'mu': baseline_episode.mu
    }
    
    optimized_params = {
        'beta': optimized_episode.beta,
        'gamma': optimized_episode.gamma,
        'mu': optimized_episode.mu
    }
    
    comparison_results = run_pinn_comparison(
        baseline_params=baseline_params,
        optimized_params=optimized_params,
        pinn_data=pinn_data,
        pipeline_pinn_results=result.get('pinn_results'),
        results_dir="PINN_comparison_results",
        # Передаем параметры обучения
        n_epoch=n_epoch,
        lambda_data=lambda_data,
        lambda_ode=lambda_ode,
        lambda_ic=lambda_ic,
        lambda_bc=lambda_bc
    )

    # ============================================================
    # 9.5. Создание сводного отчета
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 9: Creating summary report")
    print("-" * 60)
    
    import torch  # Добавьте в начало файла если еще нет
    
    report_path = create_summary_report(
        comparison_results=comparison_results,
        history=history,
        baseline_episode=baseline_episode,
        optimized_episode=optimized_episode,
        expert_comment=expert_comment,
        n_epoch=n_epoch,
        lambda_data=lambda_data,
        lambda_ode=lambda_ode,
        lambda_ic=lambda_ic,
        lambda_bc=lambda_bc,
        output_path=None
    )
    
    # ============================================================
    # 10. Итоговый отчет
    # ============================================================
    print("\n" + "=" * 80)
    print("📋 FINAL TEST REPORT")
    print("=" * 80)
    
    print(f"\n📊 OPTIMIZATION SUMMARY:")
    print(f"   Baseline: β={baseline_params['beta']:.4f}, γ={baseline_params['gamma']:.4f}, μ={baseline_params['mu']:.5f}")
    print(f"   Optimized: β={optimized_params['beta']:.4f}, γ={optimized_params['gamma']:.4f}, μ={optimized_params['mu']:.5f}")
    print(f"   Iterations: {len(history)} total, {len(accepted_episodes)} accepted")
    
    print(f"\n📈 SURROGATE RESULTS:")
    print(f"   Baseline peak: {baseline_episode.peak_position:.1f} days, height: {baseline_episode.peak_height:.0f}")
    print(f"   Optimized peak: {optimized_episode.peak_position:.1f} days, height: {optimized_episode.peak_height:.0f}")
    
    print(f"\n🧠 PINN VALIDATION:")
    if comparison_results.get('baseline_pinn', {}).get('success'):
        print(f"   Baseline PINN: ✓ complete")
    if comparison_results.get('optimized_pinn', {}).get('success'):
        print(f"   Optimized PINN: ✓ complete")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   - Pipeline graph: pipeline_graph.png")
    print(f"   - PINN agent results: PINN_agent_results/")
    print(f"   - Comparison results: PINN_comparison_results/")
    print(f"   - Summary report: {report_path}")  # ← Добавить
    
    print("\n" + "=" * 80)
    print(f"✅ TEST COMPLETE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return result, comparison_results

def create_summary_report(
    comparison_results: dict,
    history: list,
    baseline_episode,
    optimized_episode,
    expert_comment: str,
    n_epoch: int,
    lambda_data: float,
    lambda_ode: float,
    lambda_ic: float,
    lambda_bc: float,
    output_path: str = None
):
    """
    Создает сводный отчет в виде одного изображения
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.gridspec as gridspec
    
    # Создаем фигуру
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(5, 2, height_ratios=[0.5, 1.5, 1, 1.5, 1], hspace=0.4, wspace=0.3)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"PINN_comparison_results/summary_report_{timestamp}.png"
    
    # Цвета
    header_color = '#2C3E50'
    text_color = '#2C3E50'
    baseline_color = '#3498DB'
    optimized_color = '#E74C3C'
    accept_color = '#27AE60'
    reject_color = '#E74C3C'
    
    # ============================================================
    # 1. ЗАГОЛОВОК
    # ============================================================
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    title_text = f"PINN VALIDATION REPORT\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ax_title.text(0.5, 0.7, title_text, fontsize=20, fontweight='bold', 
                  ha='center', va='center', color=header_color)
    ax_title.text(0.5, 0.3, f"Expert comment: '{expert_comment}'", 
                  fontsize=14, ha='center', va='center', 
                  color=text_color, style='italic',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', alpha=0.8))
    
    # ============================================================
    # 2. ПАРАМЕТРЫ ОБУЧЕНИЯ
    # ============================================================
    ax_params = fig.add_subplot(gs[1, 0])
    ax_params.axis('off')
    ax_params.set_title('Training Parameters', fontsize=14, fontweight='bold', pad=15)
    
    params_text = f"""
    ┌─────────────────────────────────┐
    │ Number of epochs:     {n_epoch:>8}       │
    │ λ_data (data loss):   {lambda_data:>8.3f}       │
    │ λ_ode (ODE loss):     {lambda_ode:>8.3f}       │
    │ λ_ic (initial cond):  {lambda_ic:>8.3f}       │
    │ λ_bc (boundary cond): {lambda_bc:>8.3f}       │
    │ Device:               {'CUDA' if torch.cuda.is_available() else 'CPU':>8}       │
    └─────────────────────────────────┘
    """
    ax_params.text(0.1, 0.5, params_text, fontsize=11, family='monospace',
                   va='center', transform=ax_params.transAxes)
    
    # ============================================================
    # 3. СРАВНЕНИЕ ПАРАМЕТРОВ
    # ============================================================
    ax_param_comp = fig.add_subplot(gs[1, 1])
    ax_param_comp.axis('off')
    ax_param_comp.set_title('Parameter Comparison', fontsize=14, fontweight='bold', pad=15)
    
    baseline = comparison_results['baseline_params_input']
    optimized = comparison_results['optimized_params_input']
    baseline_est = comparison_results['baseline_pinn_estimated']
    optimized_est = comparison_results['optimized_pinn_estimated']
    
    param_text = f"""
    ╔═════════╦══════════════╦══════════════╦═══════════════╗
    ║ Param   ║ Baseline     ║ Optimized    ║ Change        ║
    ╠═════════╬══════════════╬══════════════╬═══════════════╣
    ║ β       ║ {baseline['beta']:.4f}        ║ {optimized['beta']:.4f}        ║ {optimized['beta'] - baseline['beta']:+.4f}          ║
    ║ γ       ║ {baseline['gamma']:.4f}        ║ {optimized['gamma']:.4f}        ║ {optimized['gamma'] - baseline['gamma']:+.4f}          ║
    ║ μ       ║ {baseline['mu']:.5f}       ║ {optimized['mu']:.5f}       ║ {optimized['mu'] - baseline['mu']:+.5f}         ║
    ╠═════════╬══════════════╬══════════════╬═══════════════╣
    ║ PINN β  ║ {baseline_est['beta']:.4f}        ║ {optimized_est['beta']:.4f}        ║ {optimized_est['beta'] - baseline_est['beta']:+.4f}          ║
    ║ PINN γ  ║ {baseline_est['gamma']:.4f}        ║ {optimized_est['gamma']:.4f}        ║ {optimized_est['gamma'] - baseline_est['gamma']:+.4f}          ║
    ║ PINN μ  ║ {baseline_est['mu']:.5f}       ║ {optimized_est['mu']:.5f}       ║ {optimized_est['mu'] - baseline_est['mu']:+.5f}         ║
    ╚═════════╩══════════════╩══════════════╩═══════════════╝
    """
    ax_param_comp.text(0.05, 0.5, param_text, fontsize=10, family='monospace',
                       va='center', transform=ax_param_comp.transAxes)
    
    # ============================================================
    # 4. COMPLETE HISTORY (таблица)
    # ============================================================
    ax_history = fig.add_subplot(gs[2, :])
    ax_history.axis('off')
    ax_history.set_title('Optimization History', fontsize=14, fontweight='bold', pad=15)
    
    # Создаем таблицу
    table_data = [['Status', 'Iter', 'β', 'γ', 'μ', 'Peak day', 'Height', 'Deaths']]
    for ep in history:
        status = '✓' if ep.accepted else '✗'
        marker = ' (BL)' if ep.iteration == 0 else ''
        table_data.append([
            status,
            f"{ep.iteration}{marker}",
            f"{ep.beta:.4f}",
            f"{ep.gamma:.4f}",
            f"{ep.mu:.5f}",
            f"{ep.peak_position:.1f}",
            f"{ep.peak_height:.0f}",
            f"{ep.total_deaths:.0f}"
        ])
    
    table = ax_history.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Цвета для строк
    for i, ep in enumerate(history, start=1):
        color = accept_color if ep.accepted else reject_color
        if ep.iteration == 0:
            color = baseline_color
        for j in range(8):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.15)
            table[(i, j)].set_text_props(weight='bold' if ep.accepted else 'normal')
    
    # Заголовок таблицы
    for j in range(8):
        table[(0, j)].set_facecolor(header_color)
        table[(0, j)].set_text_props(color='white', weight='bold')
    
    # ============================================================
    # 5. ГРАФИК I(t)
    # ============================================================
    ax_i = fig.add_subplot(gs[3, :])

    # Берем данные из comparison_results
    peak_data = comparison_results['peak_analysis']
    t_data = comparison_results.get('t_data', [])
    I_real = comparison_results.get('I_real', [])
    I_pred_bl = comparison_results.get('i_predictions', {}).get('baseline', [])
    I_pred_opt = comparison_results.get('i_predictions', {}).get('optimized', [])
    train_size = comparison_results.get('train_size', 150)

    baseline_peak_day = peak_data['baseline']['day']
    baseline_peak_height = peak_data['baseline']['height']
    optimized_peak_day = peak_data['optimized']['day']
    optimized_peak_height = peak_data['optimized']['height']
    data_peak_day = peak_data['real_data']['day']
    data_peak_height = peak_data['real_data']['height']

    # Строим кривые
    if len(t_data) > 0:
        ax_i.plot(t_data, I_real, 'o', ms=2, color='gray', alpha=0.4, label='Real data')
        ax_i.plot(t_data, I_pred_bl, '-', linewidth=2, color=baseline_color, alpha=0.8, label='Baseline PINN')
        ax_i.plot(t_data, I_pred_opt, '-', linewidth=2, color=optimized_color, alpha=0.8, label='Optimized PINN')
        ax_i.axvline(train_size, color='gray', linestyle='--', alpha=0.5, label='Train/test split')

    # Отмечаем пики
    ax_i.scatter(baseline_peak_day, baseline_peak_height, color=baseline_color, s=120, 
                marker='v', edgecolors='black', linewidth=1.5, zorder=5, label=f'Baseline peak')
    ax_i.scatter(optimized_peak_day, optimized_peak_height, color=optimized_color, s=120, 
                marker='^', edgecolors='black', linewidth=1.5, zorder=5, label=f'Optimized peak')
    ax_i.scatter(data_peak_day, data_peak_height, color='gray', s=120, 
                marker='s', edgecolors='black', linewidth=1.5, zorder=5, label=f'Real peak')

    # Текст с изменениями
    changes = peak_data['changes']
    ax_i.text(0.98, 0.98, 
            f"Δ day: {changes['peak_day']:+.0f} days\nΔ height: {changes['peak_height']:+.0f} infected",
            fontsize=11, ha='right', va='top', transform=ax_i.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    ax_i.set_xlabel('Days', fontsize=11)
    ax_i.set_ylabel('Infected (I)', fontsize=11)
    ax_i.set_title('I(t) Predictions Comparison', fontsize=14, fontweight='bold', pad=15)
    ax_i.legend(loc='upper left', fontsize=9)
    ax_i.grid(True, alpha=0.3)
    
    # ============================================================
    # 6. ИТОГОВЫЙ ВЫВОД
    # ============================================================
    ax_summary = fig.add_subplot(gs[4, :])
    ax_summary.axis('off')
    
    changes = peak_data['changes']
    day_change = changes['peak_day']
    height_change = changes['peak_height']
    
    # Определяем успешность
    if abs(day_change) > 0 or abs(height_change) > 0:
        success = "✅ Optimization successfully changed peak characteristics"
    else:
        success = "⚠️ No significant changes in peak characteristics"
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                          SUMMARY                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                              ║
    ║   {success:<90}║
    ║                                                                                              ║
    ║   Peak Day Change:     {day_change:+>8.1f} days  │  Peak Height Change:  {height_change:+>8.0f} infected    ║
    ║                                                                                              ║
    ║   Baseline → Optimized:                                                                      ║
    ║   • Peak day:     {peak_data['baseline']['day']:.0f} → {peak_data['optimized']['day']:.0f} days                                       ║
    ║   • Peak height:  {peak_data['baseline']['height']:.0f} → {peak_data['optimized']['height']:.0f} infected                                  ║
    ║                                                                                              ║
    ║   Real Data Reference:                                                                       ║
    ║   • Peak day:     {peak_data['real_data']['day']:.0f} days                                                              ║
    ║   • Peak height:  {peak_data['real_data']['height']:.0f} infected                                                         ║
    ║                                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax_summary.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
                    va='center', transform=ax_summary.transAxes)
    
    # ============================================================
    # Сохранение
    # ============================================================
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n📄 Summary report saved: {output_path}")
    return output_path

if __name__ == "__main__":
    result, comparison = main()