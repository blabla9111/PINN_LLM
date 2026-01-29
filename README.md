# Expert-Guided PINN Framework

## 🧠 About the Project  
Expert-Guided PINN is a system for interactive epidemiological forecasting that allows epidemiologists to adjust predictions using simple text comments, without requiring deep understanding of the underlying mathematical model.

**🌐 Live Demo:** [pinnllm.streamlit.app](https://pinnllm.streamlit.app)

📌 Status: MVP (Minimum Viable Product) – Research Prototype

![GUI](./Images/figure_gui.png)

## 🎯 Goal  
To bridge the gap between qualitative expert knowledge (expressed in natural language) and formal mathematical models. The system automatically converts expert text comments into modifications of the loss function of a Physics-Informed Neural Network (PINN), making the forecast calibration process accessible and dynamic.

## 🔧 Key Features  
📝 **Text-Based Feedback** — Experts provide comments in natural language  
🧩 **Automatic Classification** — A BERT-based model determines the comment's class and subclass  
🤖 **LLM Code Generation** — A large language model translates comments into loss function modifications  
📊 **Dynamic PINN Adaptation** — The model retrains based on expert feedback  
🔄 **Interactive Loop** — Experts see results and can refine comments iteratively  

Application workflow:  
![app_scheme](./Images/figure_1.png)

## 🐍 Prerequisites

- **Python 3.11+** (required for compatibility with dependencies)
- Stable internet connection (for Hugging Face and Supabase APIs)

## ⚙️ Setup & Configuration

### 1. Environment Configuration
The system uses external services that require API keys:

1. **Copy the example configuration file:**
   ```bash
   cp secrets_example.toml secrets.toml
   ```

2. **Edit `secrets.toml` with your credentials:**
   ```toml
   # Hugging Face Token (for LLM and classifier access)
   HUGGINGFACE_HUB_TOKEN = "hf_***"
   
   # Supabase Database (SaaS PostgreSQL)
   SUPABASE_URL = "https://***.supabase.co"
   SUPABASE_KEY = "***"
   
   # LLM Model Configuration
   LLM_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
   LLM_TEMPERATURE = "1.0"
   
   # App Configuration
   APP_MODE = "DEV"  # DEV or PROD
   DEBUG = "true"    # true or false
   
   # TensorBoard Configuration
   ENABLE_TENSORBOARD = "true"  # true or false
   ```

### 2. Required Services

#### **Hugging Face** 🤗
- Provides access to:
  - **LLM models** (Llama-3.3-70B-Instruct, DeepSeek-V3.1, etc.)
  - **Hierarchical text classifier** (BERT-based model for comment classification)
- Get your token at: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

#### **Supabase** 🗄️
- SaaS PostgreSQL database used for:
  - Storing user sessions
  - Saving forecast results
  - Logging expert interactions
- Create a free project at: [supabase.com](https://supabase.com)

All necessary database configuration files, trained models and loss functions used are located in the `supabase/` folder.

## 📦 Installation

1. **Create and activate virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Linux
   venv\Scripts\activate # on Windows
   ```

2. **Install dependencies:**
   For CPU version:
   ```bash
   pip install -r requirements.txt
   ```
   For GPU support use requirements_cpu_or_gpu.txt

## 🚀 Running the Application
```bash
streamlit run streamlit_main.py
```


---

**Note:** The project is research-oriented and currently serves as a proof-of-concept for integrating LLMs into expert-guided epidemiological modeling. Ensure you comply with the terms of service for Hugging Face and Supabase when deploying in production environments.