"""
Training page for SOMA GUI.

Provides interface for voice model training with real-time progress.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from typing import Optional, Dict, Any
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


class TrainingPage(BasePage):
    """
    Training page for voice model training.
    
    Features:
    - Dataset path selection
    - Model name configuration
    - Training parameters
    - Real-time progress and logging
    - Pause/Resume/Stop controls
    """
    
    PAGE_NAME = "Voice Clone"
    PAGE_ICON = "🎤"
    PAGE_DESCRIPTION = "Train voice models"
    
    # Default training parameters
    DEFAULT_PARAMS = {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.001,
        "save_every": 10,
    }
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the training page."""
        super().__init__(parent, app)
        
        # Training state
        self._is_training = False
        self._is_paused = False
        self._training_thread: Optional[threading.Thread] = None
        
        # Variables
        self.dataset_path = tk.StringVar()
        self.model_name = tk.StringVar(value="my_voice_model")
        self.epochs = tk.IntVar(value=self.DEFAULT_PARAMS["epochs"])
        self.batch_size = tk.IntVar(value=self.DEFAULT_PARAMS["batch_size"])
        self.learning_rate = tk.DoubleVar(value=self.DEFAULT_PARAMS["learning_rate"])
        self.save_every = tk.IntVar(value=self.DEFAULT_PARAMS["save_every"])
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
    
    def _create_widgets(self):
        """Create training page widgets."""
        # Title section
        self.create_title_section(
            self.content_frame,
            "Voice Clone Training",
            "Train a custom voice model from audio samples"
        )
        
        # Main content area with two columns
        main_frame = ttk.Frame(self.content_frame, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Configuration
        left_frame = ttk.Frame(main_frame, style="TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self._create_dataset_section(left_frame)
        self._create_model_section(left_frame)
        self._create_parameters_section(left_frame)
        
        # Right column - Progress and logs
        right_frame = ttk.Frame(main_frame, style="TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self._create_progress_section(right_frame)
        self._create_log_section(right_frame)
    
    def _create_dataset_section(self, parent: tk.Widget):
        """Create dataset selection section."""
        card = self.create_card(parent, "Training Dataset")
        
        # Path entry with browse button
        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)
        
        path_entry = ttk.Entry(path_frame, textvariable=self.dataset_path)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_btn = ttk.Button(path_frame, text="Browse...",
                               style="Secondary.TButton",
                               command=self._browse_dataset)
        browse_btn.pack(side=tk.RIGHT)
        
        # Help text
        help_label = ttk.Label(card, 
                              text="Select folder containing WAV files (16kHz+ recommended)",
                              style="Muted.TLabel")
        help_label.pack(anchor=tk.W, pady=(10, 0))
    
    def _create_model_section(self, parent: tk.Widget):
        """Create model configuration section."""
        card = self.create_card(parent, "Model Configuration")
        
        # Model name
        name_frame = ttk.Frame(card, style="Card.TFrame")
        name_frame.pack(fill=tk.X)
        
        ttk.Label(name_frame, text="Model Name:", style="Card.TLabel").pack(side=tk.LEFT)
        name_entry = ttk.Entry(name_frame, textvariable=self.model_name)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
    
    def _create_parameters_section(self, parent: tk.Widget):
        """Create training parameters section."""
        card = self.create_card(parent, "Training Parameters")
        
        # Parameters grid
        params = [
            ("Epochs:", self.epochs, "100", 1, 10000),
            ("Batch Size:", self.batch_size, "16", 1, 128),
            ("Learning Rate:", self.learning_rate, "0.001", 0.00001, 0.1),
            ("Save Every N:", self.save_every, "10", 1, 100),
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
        
        # Start/Stop buttons
        button_frame = ttk.Frame(card, style="Card.TFrame")
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.start_btn = ttk.Button(button_frame, text="▶ Start Training",
                                   style="Primary.TButton",
                                   command=self._start_training)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.pause_btn = ttk.Button(button_frame, text="⏸ Pause",
                                   style="Secondary.TButton",
                                   command=self._pause_training,
                                   state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop",
                                  style="Danger.TButton",
                                  command=self._stop_training,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
    
    def _create_progress_section(self, parent: tk.Widget):
        """Create progress display section."""
        card = self.create_card(parent, "Training Progress")
        
        # Progress bar
        progress_frame = ttk.Frame(card, style="Card.TFrame")
        progress_frame.pack(fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Status info
        status_frame = ttk.Frame(card, style="Card.TFrame")
        status_frame.pack(fill=tk.X)
        
        ttk.Label(status_frame, text="Status:", style="Card.TLabel").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                     style="Card.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Epoch info
        self.epoch_label = ttk.Label(card, text="Epoch: 0 / 0",
                                    style="Muted.TLabel")
        self.epoch_label.pack(anchor=tk.W, pady=(10, 0))
        
        # Loss info
        self.loss_label = ttk.Label(card, text="Loss: N/A",
                                   style="Muted.TLabel")
        self.loss_label.pack(anchor=tk.W)
    
    def _create_log_section(self, parent: tk.Widget):
        """Create log output section."""
        card = self.create_card(parent, "Training Log")
        
        # Log text widget with scrollbar
        log_frame = ttk.Frame(card, style="Card.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget
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
        
        # Configure text tags for coloring
        self.log_text.tag_configure("info", foreground=Colors.ACCENT_INFO)
        self.log_text.tag_configure("success", foreground=Colors.ACCENT_SUCCESS)
        self.log_text.tag_configure("warning", foreground=Colors.ACCENT_WARNING)
        self.log_text.tag_configure("error", foreground=Colors.ACCENT_ERROR)
    
    def _browse_dataset(self):
        """Open file dialog to select dataset folder."""
        folder = filedialog.askdirectory(
            title="Select Training Dataset Folder",
            initialdir="/"
        )
        if folder:
            self.dataset_path.set(folder)
            self._log(f"Dataset path set: {folder}", "info")
    
    def _start_training(self):
        """Start the training process."""
        # Validate inputs
        if not self.dataset_path.get():
            messagebox.showwarning("Warning", "Please select a dataset folder.")
            return
        
        if not self.model_name.get():
            messagebox.showwarning("Warning", "Please enter a model name.")
            return
        
        # Update UI state
        self._is_training = True
        self._is_paused = False
        self.start_btn.configure(state=tk.DISABLED)
        self.pause_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Training...")
        
        self._log("Starting training...", "info")
        self._log(f"Model: {self.model_name.get()}", "info")
        self._log(f"Dataset: {self.dataset_path.get()}", "info")
        self._log(f"Epochs: {self.epochs.get()}, Batch: {self.batch_size.get()}", "info")
        
        # Start training in background thread
        self._training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self._training_thread.start()
    
    def _pause_training(self):
        """Pause or resume training."""
        self._is_paused = not self._is_paused
        if self._is_paused:
            self.pause_btn.configure(text="▶ Resume")
            self.status_var.set("Paused")
            self._log("Training paused", "warning")
        else:
            self.pause_btn.configure(text="⏸ Pause")
            self.status_var.set("Training...")
            self._log("Training resumed", "info")
    
    def _stop_training(self):
        """Stop the training process."""
        self._is_training = False
        self._is_paused = False
        self.start_btn.configure(state=tk.NORMAL)
        self.pause_btn.configure(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Stopped")
        self._log("Training stopped by user", "warning")
    
    def _training_worker(self):
        """Background worker for training (simulated)."""
        epochs = self.epochs.get()
        
        for epoch in range(1, epochs + 1):
            if not self._is_training:
                break
            
            while self._is_paused and self._is_training:
                threading.Event().wait(0.5)
            
            if not self._is_training:
                break
            
            # Simulate training work
            threading.Event().wait(0.1)
            
            # Update progress
            progress = (epoch / epochs) * 100
            loss = 1.0 / epoch  # Simulated decreasing loss
            
            # Schedule UI updates on main thread
            self.after(0, self._update_progress, epoch, epochs, progress, loss)
        
        if self._is_training:
            self.after(0, self._training_complete)
    
    def _update_progress(self, epoch: int, total: int, progress: float, loss: float):
        """Update progress display (called from main thread)."""
        self.progress_var.set(progress)
        self.epoch_label.configure(text=f"Epoch: {epoch} / {total}")
        self.loss_label.configure(text=f"Loss: {loss:.4f}")
        
        if epoch % 10 == 0:
            self._log(f"Epoch {epoch}/{total} - Loss: {loss:.4f}", "info")
    
    def _training_complete(self):
        """Handle training completion."""
        self._is_training = False
        self.start_btn.configure(state=tk.NORMAL)
        self.pause_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Completed")
        self.progress_var.set(100)
        self._log("Training completed successfully!", "success")
        self._log(f"Model saved as: {self.model_name.get()}", "success")
    
    def _log(self, message: str, level: str = "info"):
        """Add a message to the log output."""
        def _append():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{level.upper()}] {message}\n", level)
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        
        self.after(0, _append)
