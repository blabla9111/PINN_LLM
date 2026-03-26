import re
from datetime import datetime
from typing import Dict
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field, validator

# ============================================
# Pydantic models for output parsing
# ============================================

class EpiParameters(BaseModel):
    """Pydantic model for LLM output validation"""
    reasoning: str = Field(description="Detailed justification for parameter choices")
    beta: float = Field(description="Infection rate (0.1 - 1.0)")
    gamma: float = Field(description="Recovery rate (0.05 - 1.0)")
    mu: float = Field(description="Mortality rate (0.001 - 0.1)")
    expected_peak_position: float = Field(description="Expected peak position in days")
    expected_peak_height: float = Field(description="Expected peak height (number infected)")
    expected_total_deaths: float = Field(description="Expected total deaths")
    confidence: str = Field(description="Confidence level: high/medium/low")
    
    @validator('beta')
    def beta_validator(cls, v):
        if not (0.1 <= v <= 3.0):
            raise ValueError(f'beta must be between 0.1 and 3.0, got {v}')
        return v
    
    @validator('gamma')
    def gamma_validator(cls, v):
        if not (0.05 <= v <= 1.0):
            raise ValueError(f'gamma must be between 0.05 and 1.0, got {v}')
        return v
    
    @validator('mu')
    def mu_validator(cls, v):
        if not (0.001 <= v <= 0.1):
            raise ValueError(f'mu must be between 0.001 and 0.1, got {v}')
        return v
    
    @validator('confidence')
    def confidence_validator(cls, v):
        if v not in ['high', 'medium', 'low']:
            raise ValueError(f'confidence must be high/medium/low, got {v}')
        return v


@dataclass
class Episode:
    """Data class for storing parameter episodes"""
    beta: float
    gamma: float
    mu: float
    peak_position: float = None
    peak_height: float = None
    total_deaths: float = None
    timestamp: str = None
    expert_comment: str = None
    accepted: bool = False
    iteration: int = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_prompt_format(self) -> str:
        """Format episode for prompt display"""
        status = "✓ ACCEPTED" if self.accepted else "✗ REJECTED"
        return (
            f"**Iteration {self.iteration}:**\n"
            f"- Parameters: β={self.beta:.4f}, γ={self.gamma:.4f}, μ={self.mu:.5f}\n"
            f"- Results: peak at day {self.peak_position:.1f}, height {self.peak_height:.0f}, deaths {self.total_deaths:.0f}\n"
            f"- Status: {status}\n"
            f"- Expert comment: {self.expert_comment if self.expert_comment else 'None'}\n"
        )
    

class GraphState:
    """State for LangGraph workflow"""
    def __init__(self, data: Dict = None):
        self.data = data or {}
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def set(self, key, value):
        self.data[key] = value
    
    def to_dict(self):
        return self.data