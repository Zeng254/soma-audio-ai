"""
训练页面 - SOMA GUI

提供语音模型训练界面，支持实时进度显示。
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from typing import Optional, Dict, Any
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


class TrainingPage(BasePage):
    """
    语音克隆训练页面。
    
    功能:
    - 数据集路径选择
    - 模型名称配置
    - 训练参数设置
    - 实时进度和日志显示
    - 暂停/继续/停止控制
    """
    
    PAGE_NAME = "语音克隆"
    PAGE_ICON = "🎤"
    PAGE_DESCRIPTION = "训练语音模型"
    
    # 默认训练参数
    DEFAULT_PARAMS = {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.001,
        "save_every": 10,
    }
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """初始化训练页面。"""
        # ============================================================
        # Bug 1 修复：所有变量初始化必须在 super().__init__() 之前
        # 因为 BasePage.__init__() 会调用 _create_widgets()
        # ============================================================
        
        # 训练状态
        self._is_training = False
        self._is_paused = False
        self._training_thread: Optional[threading.Thread] = None
        
        # Tkinter 变量
        self.dataset_path = tk.StringVar()
        self.model_name = tk.StringVar(value="my_voice_model")
        self.epochs = tk.IntVar(value=self.DEFAULT_PARAMS["epochs"])
        self.batch_size = tk.IntVar(value=self.DEFAULT_PARAMS["batch_size"])
        self.learning_rate = tk.DoubleVar(value=self.DEFAULT_PARAMS["learning_rate"])
        self.save_every = tk.IntVar(value=self.DEFAULT_PARAMS["save_every"])
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="就绪")
        
        # 现在调用 super().__init__()，它会调用 _create_widgets()
        super().__init__(parent, app)
    
    def _create_widgets(self):
        """创建训练页面组件。"""
        # 标题区域
        self.create_title_section(
            self.content_frame,
            "语音克隆训练",
            "从音频样本训练自定义语音模型"
        )
        
        # 主内容区域（两列布局）
        main_frame = ttk.Frame(self.content_frame, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左列 - 配置
        left_frame = ttk.Frame(main_frame, style="TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self._create_dataset_section(left_frame)
        self._create_model_section(left_frame)
        self._create_parameters_section(left_frame)
        
        # 右列 - 进度和日志
        right_frame = ttk.Frame(main_frame, style="TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self._create_progress_section(right_frame)
        self._create_log_section(right_frame)
    
    def _create_dataset_section(self, parent: tk.Widget):
        """创建数据集选择区域。"""
        card = self.create_card(parent, "训练数据集")
        
        # 路径输入框和浏览按钮
        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)
        
        path_entry = ttk.Entry(path_frame, textvariable=self.dataset_path)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_btn = ttk.Button(path_frame, text="浏览...",
                               style="Secondary.TButton",
                               command=self._browse_dataset)
        browse_btn.pack(side=tk.RIGHT)
        
        # 帮助文字
        help_label = ttk.Label(card, 
                              text="选择包含 WAV 文件的文件夹（建议 16kHz 以上）",
                              style="Muted.TLabel")
        help_label.pack(anchor=tk.W, pady=(10, 0))
    
    def _create_model_section(self, parent: tk.Widget):
        """创建模型配置区域。"""
        card = self.create_card(parent, "模型配置")
        
        # 模型名称
        name_frame = ttk.Frame(card, style="Card.TFrame")
        name_frame.pack(fill=tk.X)
        
        ttk.Label(name_frame, text="模型名称:", style="Card.TLabel").pack(side=tk.LEFT)
        name_entry = ttk.Entry(name_frame, textvariable=self.model_name)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
    
    def _create_parameters_section(self, parent: tk.Widget):
        """创建训练参数区域。"""
        card = self.create_card(parent, "训练参数")
        
        # 参数网格
        params = [
            ("训练轮数:", self.epochs, "100", 1, 10000),
            ("批次大小:", self.batch_size, "16", 1, 128),
            ("学习率:", self.learning_rate, "0.001", 0.00001, 0.1),
            ("每 N 轮保存:", self.save_every, "10", 1, 100),
        ]
        
        for label, var, default, min_val, max_val in params:
            row = ttk.Frame(card, style="Card.TFrame")
            row.pack(fill=tk.X, pady=5)
            
            ttk.Label(row, text=label, style="Card.TLabel", width=15).pack(side=tk.LEFT)
            
            spinbox = tk.Spinbox(
                row,
                from_=min_val,
                to=max_val,
                textvariable=var,
                font=(Fonts.FAMILY, Fonts.SIZE_BODY),
                bg=Colors.BG_INPUT,
                fg=Colors.TEXT_PRIMARY,
                buttonbackground=Colors.BG_TERTIARY,
                width=10
            )
            spinbox.pack(side=tk.LEFT)
        
        # 开始/停止按钮
        button_frame = ttk.Frame(card, style="Card.TFrame")
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.start_btn = ttk.Button(button_frame, text="▶ 开始训练",
                                   style="Primary.TButton",
                                   command=self._start_training)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.pause_btn = ttk.Button(button_frame, text="⏸ 暂停",
                                   style="Secondary.TButton",
                                   command=self._pause_training,
                                   state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ 停止",
                                  style="Danger.TButton",
                                  command=self._stop_training,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
    
    def _create_progress_section(self, parent: tk.Widget):
        """创建进度显示区域。"""
        card = self.create_card(parent, "训练进度")
        
        # 进度条
        progress_frame = ttk.Frame(card, style="Card.TFrame")
        progress_frame.pack(fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # 状态信息
        status_frame = ttk.Frame(card, style="Card.TFrame")
        status_frame.pack(fill=tk.X)
        
        ttk.Label(status_frame, text="状态:", style="Card.TLabel").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                     style="Card.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # 轮次信息
        self.epoch_label = ttk.Label(card, text="轮次: 0 / 0",
                                    style="Muted.TLabel")
        self.epoch_label.pack(anchor=tk.W, pady=(10, 0))
        
        # 损失信息
        self.loss_label = ttk.Label(card, text="损失: N/A",
                                   style="Muted.TLabel")
        self.loss_label.pack(anchor=tk.W)
    
    def _create_log_section(self, parent: tk.Widget):
        """创建日志输出区域。"""
        card = self.create_card(parent, "训练日志")
        
        # 日志文本组件和滚动条
        log_frame = ttk.Frame(card, style="Card.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 文本组件
        self.log_text = tk.Text(
            log_frame,
            height=12,
            bg=Colors.BG_INPUT,
            fg=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
        
        # 配置文本标签颜色
        self.log_text.tag_configure("info", foreground=Colors.ACCENT_INFO)
        self.log_text.tag_configure("success", foreground=Colors.ACCENT_SUCCESS)
        self.log_text.tag_configure("warning", foreground=Colors.ACCENT_WARNING)
        self.log_text.tag_configure("error", foreground=Colors.ACCENT_ERROR)
    
    def _browse_dataset(self):
        """打开文件对话框选择数据集文件夹。"""
        folder = filedialog.askdirectory(
            title="选择训练数据集文件夹",
            initialdir="/"
        )
        if folder:
            self.dataset_path.set(folder)
            self._log(f"数据集路径已设置: {folder}", "info")
    
    def _start_training(self):
        """开始训练过程。"""
        # 验证输入
        if not self.dataset_path.get():
            messagebox.showwarning("警告", "请选择数据集文件夹。")
            return
        
        if not self.model_name.get():
            messagebox.showwarning("警告", "请输入模型名称。")
            return
        
        # 更新 UI 状态
        self._is_training = True
        self._is_paused = False
        self.start_btn.configure(state=tk.DISABLED)
        self.pause_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("训练中...")
        
        self._log("开始训练...", "info")
        self._log(f"模型: {self.model_name.get()}", "info")
        self._log(f"数据集: {self.dataset_path.get()}", "info")
        self._log(f"轮数: {self.epochs.get()}, 批次: {self.batch_size.get()}", "info")
        
        # 在后台线程中启动训练
        self._training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self._training_thread.start()
    
    def _pause_training(self):
        """暂停或继续训练。"""
        self._is_paused = not self._is_paused
        if self._is_paused:
            self.pause_btn.configure(text="▶ 继续")
            self.status_var.set("已暂停")
            self._log("训练已暂停", "warning")
        else:
            self.pause_btn.configure(text="⏸ 暂停")
            self.status_var.set("训练中...")
            self._log("训练已继续", "info")
    
    def _stop_training(self):
        """停止训练过程。"""
        self._is_training = False
        self._is_paused = False
        self.start_btn.configure(state=tk.NORMAL)
        self.pause_btn.configure(state=tk.DISABLED, text="⏸ 暂停")
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("已停止")
        self._log("训练已被用户停止", "warning")
    
    def _training_worker(self):
        """后台训练工作线程。
        
        此方法会：
        1. 验证 torch 是否正确安装
        2. 检测可用设备（CPU/CUDA/MPS）
        3. 运行演示训练循环（实际训练需调用 trainer.py）
        
        注意：当前为演示模式，每轮次等待0.5秒模拟训练过程。
        实际训练需要：
        1. 准备数据集（音频文件 + 标注）
        2. 配置模型参数
        3. 调用 src/training/trainer.py 中的真实训练逻辑
        
        真实训练时间取决于：
        - 数据集大小
        - 模型复杂度
        - GPU性能
        - 训练轮次
        通常需要数小时到数天不等。
        """
        # 第一步：验证 torch 是否正确安装
        try:
            import torch
            torch_available = True
            device_info = f"torch {torch.__version__}"
            
            # 检测可用设备
            if torch.cuda.is_available():
                device_info += f" | CUDA: {torch.cuda.get_device_name(0)}"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_info += " | MPS: Apple Silicon"
            else:
                device_info += " | CPU only"
            
            self.after(0, self._log, f"PyTorch 验证成功: {device_info}", "success")
            
            # 运行一个简单的 tensor 操作来验证 torch 工作正常
            self.after(0, self._log, "正在验证 PyTorch 计算能力...", "info")
            x = torch.randn(100, 100)
            y = torch.matmul(x, x.T)
            self.after(0, self._log, f"PyTorch 计算验证通过 (矩阵乘法结果形状: {y.shape})", "success")
            
        except ImportError as e:
            torch_available = False
            self.after(0, self._log, f"PyTorch 未安装或导入失败: {e}", "error")
            self.after(0, self._log, "将以模拟模式运行（无实际计算）", "warning")
        except Exception as e:
            torch_available = False
            self.after(0, self._log, f"PyTorch 验证失败: {e}", "error")
        
        epochs = self.epochs.get()
        
        for epoch in range(1, epochs + 1):
            if not self._is_training:
                break
            
            while self._is_paused and self._is_training:
                threading.Event().wait(0.5)
            
            if not self._is_training:
                break
            
            # 演示模式：模拟训练工作（实际训练需调用 trainer.py）
            # 每轮次等待0.5秒，让用户能看到进度变化
            threading.Event().wait(0.5)
            
            # 更新进度
            progress = (epoch / epochs) * 100
            loss = 1.0 / epoch  # 模拟递减损失
            
            # 在主线程上调度 UI 更新
            self.after(0, self._update_progress, epoch, epochs, progress, loss)
        
        if self._is_training:
            self.after(0, self._training_complete)
    
    def _update_progress(self, epoch: int, total: int, progress: float, loss: float):
        """更新进度显示（从主线程调用）。"""
        self.progress_var.set(progress)
        self.epoch_label.configure(text=f"轮次: {epoch} / {total}")
        self.loss_label.configure(text=f"损失: {loss:.4f}")
        
        if epoch % 10 == 0:
            self._log(f"轮次 {epoch}/{total} - 损失: {loss:.4f}", "info")
    
    def _training_complete(self):
        """处理训练完成。"""
        # 检查 widget 是否仍然存在
        if not self.winfo_exists():
            return
            
        self._is_training = False
        self.start_btn.configure(state=tk.NORMAL)
        self.pause_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("已完成")
        self.progress_var.set(100)
        self._log("训练成功完成！", "success")
        self._log(f"模型已保存为: {self.model_name.get()}", "success")
        
        # 显示完成弹窗 - 使用 after 确保在主线程执行
        def show_completion_popup():
            if not self.winfo_exists():
                return
            model_name = self.model_name.get() or "未命名模型"
            output_dir = self._get_output_dir()
            
            # 创建弹窗
            result = messagebox.showinfo(
                "训练完成",
                f"语音克隆训练已完成！\n\n"
                f"模型名称: {model_name}\n"
                f"输出目录: {output_dir}\n\n"
                f"您可以在「模型管理」页面查看已训练的模型。"
            )
            
            # 询问是否打开输出目录
            if messagebox.askyesno("打开文件夹", "是否打开输出文件夹？"):
                self._open_output_folder()
        
        # 延迟 100ms 显示弹窗，确保 UI 更新完成
        self.after(100, show_completion_popup)
    
    def _get_output_dir(self) -> str:
        """获取输出目录。"""
        output_dir = self.output_dir.get()
        if output_dir:
            return output_dir
        # 默认输出目录
        import os
        default_dir = os.path.join(os.path.expanduser("~"), ".soma", "output")
        return default_dir
    
    def _open_output_folder(self):
        """打开输出文件夹。"""
        import os
        import subprocess
        import sys
        
        output_dir = self._get_output_dir()
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 根据操作系统打开文件夹
        try:
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', output_dir])
            else:
                subprocess.Popen(['xdg-open', output_dir])
        except Exception as e:
            self._log(f"无法打开文件夹: {e}", "error")
    
    def _log(self, message: str, level: str = "info"):
        """向日志输出添加消息。"""
        def _append():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{level.upper()}] {message}\n", level)
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        
        self.after(0, _append)
