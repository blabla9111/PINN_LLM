"""
LLM-epiparam generator agent for SIRD model parameter optimization
"""

import json


# Import your config and data structures
import config
from formats.data_formats import Episode, GraphState
from agents.EpiParamGeneratorAgent import LLMEpiParamGenerator







# ============================================
# Usage example
# ============================================

if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    
    # Initialize LLM
    llm = HuggingFaceEndpoint(
        repo_id=config.DEFAULT_MODEL_NAME,
        huggingfacehub_api_token=config.HUGGINGFACE_TOKEN,
        temperature=float(config.DEFAULT_TEMPERATURE),
        max_new_tokens=int(config.DEFAULT_MAX_TOKENS)
    )
    chat = ChatHuggingFace(llm=llm)
    
    # Create generator agent
    generator = LLMEpiParamGenerator(llm=chat)
    
    # Configure task
    generator.set_task_config({
        'description': 'Find parameters to achieve infection peak at day 30',
        'target_peak': 30,
        'population': 1_000_000,
        'I0': 100
    })
    
    # Create initial episode
    current_episode = Episode(
        beta=0.8,
        gamma=0.2,
        mu=0.02,
        peak_position=45.0,
        peak_height=15000,
        total_deaths=8500,
        expert_comment="Peak too late, need earlier peak",
        accepted=False
    )
    
    # Add to history
    generator.add_to_history(current_episode)
    
    # Create state
    state = GraphState({
        'current_episode': current_episode,
        'expert_comment': "Peak should be around day 30, try increasing beta"
    })
    
    # Generate new parameters
    result_state = generator(state)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 GENERATION RESULTS")
    print("=" * 60)
    generated = result_state.get('generated_params')
    if generated:
        print(json.dumps(generated, indent=2, ensure_ascii=False))