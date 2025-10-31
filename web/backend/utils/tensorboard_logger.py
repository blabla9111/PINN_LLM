import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import numpy as np

class TensorBoardExperimentLogger:
    def __init__(self, log_dir=None, experiment_name=None):
        if experiment_name is None:
            experiment_name = f"dinn_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if log_dir is None:
            log_dir = os.path.join("./runs/", experiment_name)
        else:
            log_dir = os.path.join("./runs/"+log_dir, experiment_name)
            # log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = log_dir
        print(f"✅ TensorBoard эксперимент: {experiment_name}")
        print(f"📊 Для просмотра: tensorboard --logdir=./runs/")

    def save_model(self, pytorch_model):
        """Сохраняет информацию о модели и создает чекпоинт в папке эксперимента"""
        try:
            # Сохраняем модель в папке эксперимента
            model_checkpoint_path = os.path.join(self.experiment_dir, "model_checkpoint.pth")
            
            checkpoint = {
                'model_state_dict': pytorch_model.state_dict(),
                'beta_tilda': pytorch_model.beta_tilda,
                'gamma_tilda': pytorch_model.gamma_tilda,
                'S_max': pytorch_model.S_max,
                'I_max': pytorch_model.I_max,
                'D_max': pytorch_model.D_max,
                'R_max': pytorch_model.R_max,
                'S_min': pytorch_model.S_min,
                'I_min': pytorch_model.I_min,
                'D_min': pytorch_model.D_min,
                'R_min': pytorch_model.R_min,
                'train_size': pytorch_model.train_size,
                'device': pytorch_model.device
            }
            
            torch.save(checkpoint, model_checkpoint_path)
            
            # Логируем информацию о модели
            model_info = f"""
            Модель DINN:
            - Эксперимент: {self.experiment_name}
            - Всего параметров: {sum(p.numel() for p in pytorch_model.parameters())}
            - Обучаемые параметры: {sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)}
            - Устройство: {pytorch_model.device}
            - Размер обучающей выборки: {pytorch_model.train_size}
            - N (население): {pytorch_model.N}
            - Модель сохранена: {model_checkpoint_path}
            """
            self.writer.add_text("Model/Architecture", model_info)
            
            # Логируем параметры модели
            self.writer.add_text("Model/Parameters", 
                               f"Beta: {pytorch_model.beta.item():.4f}\nGamma: {pytorch_model.gamma.item():.4f}\nR0: {pytorch_model.beta.item() / pytorch_model.gamma.item():.4f}")
            
            print(f"✅ Чекпоинт модели сохранен: {model_checkpoint_path}")
            
        except Exception as e:
            print(f"⚠️ Не удалось сохранить информацию о модели: {e}")

    def save_train_process(self, loss_num, beta_num, gamma_num, epoch):
        """Сохраняет процесс обучения: loss и параметры"""
        self.writer.add_scalar("Training/Loss", loss_num, epoch)
        self.writer.add_scalar("Parameters/Beta", beta_num, epoch)
        self.writer.add_scalar("Parameters/Gamma", gamma_num, epoch)

    def save_metrics(self, metrics_S, metrics_I, metrics_R, metrics_D):
        """Сохраняет метрики как таблицу в TensorBoard один раз в конце обучения"""
        
        # Создаем общую таблицу в Markdown формате
        table = "### 📊 Финальные метрики качества модели\n\n"
        table += "| Метрика | S (Susceptible) | I (Infected) | R (Recovered) | D (Dead) |\n"
        table += "|---------|-----------------|--------------|---------------|----------|\n"
        
        # Проходим по всем метрикам (MAE, MSE, RMSE, R2)
        for metric_name in ['MAE', 'MSE', 'RMSE', 'R2']:
            s_val = metrics_S.get(metric_name, 0)
            i_val = metrics_I.get(metric_name, 0)
            r_val = metrics_R.get(metric_name, 0)
            d_val = metrics_D.get(metric_name, 0)
            
            table += f"| **{metric_name}** | {s_val:.4f} | {i_val:.4f} | {r_val:.4f} | {d_val:.4f} |\n"
        
        # Сохраняем таблицу в TensorBoard (используем epoch=0 для однократного сохранения)
        self.writer.add_text("Metrics/Final_Table", table, 0)
        
        
        
        print("✅ Финальные метрики сохранены в TensorBoard")

    def save_comparing_graph(self, fig_S, fig_I, fig_R, fig_D):
        """Сохраняет 4 графика сравнения моделей в TensorBoard"""
        
        try:
            # Конвертируем Plotly figures в изображения
            import plotly.io as pio
            
            # Сохраняем каждый график как изображение
            fig_dict = {
                "Susceptible (S)": fig_S,
                "Infected (I)": fig_I, 
                "Recovered (R)": fig_R,
                "Dead (D)": fig_D
            }
            
            for name, fig in fig_dict.items():
                if fig is not None:
                    # Конвертируем Plotly figure в image tensor
                    img_bytes = pio.to_image(fig, format='png', width=1000, height=600)
                    
                    # Конвертируем bytes в numpy array
                    import io
                    from PIL import Image
                    img = Image.open(io.BytesIO(img_bytes))
                    img_array = np.array(img)
                    
                    # Конвертируем в tensor (C, H, W)
                    if img_array.ndim == 3:
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                    else:
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
                    
                    # Сохраняем в TensorBoard
                    self.writer.add_image(f"Comparison/{name}", img_tensor, 0)
            
            print("✅ Графики сравнения сохранены в TensorBoard")
            
            # # Также сохраняем информацию о графиках как текст
            # comparison_info = """
            # ### 📊 Графики сравнения моделей
            
            # Сохранены графики для всех компонентов:
            # - **Susceptible (S)** - Восприимчивые
            # - **Infected (I)** - Зараженные  
            # - **Recovered (R)** - Выздоровевшие
            # - **Dead (D)** - Умершие
            
            # На графиках:
            # - 🔵 Синие точки - реальные данные (обучающая часть)
            # - ⚪ Белые точки - реальные данные (тестовая часть)
            # - ⚫ Черная линия - предсказания старой модели
            # - 🔴 Красная линия - предсказания новой модели
            # """
            
            # self.writer.add_text("Comparison/Info", comparison_info, 0)
            
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении графиков сравнения: {e}")

    def save_expert_comment(self, expert_comment):
        """Сохраняет экспертный комментарий к эксперименту"""
        if expert_comment:
            comment_text = f"""
            ### 💬 Экспертный комментарий:
            {expert_comment}
            
            *Эксперимент: {self.experiment_name}*
            *Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
            """
            self.writer.add_text("Expert/Comment", comment_text)

    def save_loss_func(self, loss_func_before, loss_func_after):
        """Сохраняет информацию о функции потерь в виде текста"""
        loss_info = "### 📝 Функция потерь\n\n"
        
        if loss_func_before:
            loss_info += "**Исходная функция потерь:**\n"
            loss_info += f"```python\n{loss_func_before}\n```\n\n"
        
        if loss_func_after:
            loss_info += "**Модифицированная функция потерь:**\n"
            loss_info += f"```python\n{loss_func_after}\n```\n\n"
        
        if loss_func_before and loss_func_after:
            loss_info += "**Изменения:** Сравнение исходной и модифицированной версий функции потерь"
        
        # Сохраняем в TensorBoard
        self.writer.add_text("Training/Loss_Function", loss_info, 0)
        
        print("✅ Информация о функции потерь сохранена в TensorBoard")

    def save_loss_function_error_counter(self, error_counter, is_fixed):
        self.writer.add_scalar(f'loss_function_error_counter', error_counter, 0)
        self.writer.add_text('loss_function_errors/network_info', 
                        f'Fixed: {is_fixed}, Errors: {error_counter}',0)
        
    def save_LLM_model_name(self, model_name, model_temperature):
        self.writer.add_text("Training/LLM_model_name", model_name, 0)
        self.writer.add_text("Training/LLM_model_temperature", str(model_temperature), 0)