"""
Models page for SOMA GUI.

Provides interface for managing trained voice models.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List, Dict
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


class ModelsPage(BasePage):
    """
    Models page for managing trained voice models.
    
    Features:
    - Model list with details
    - Import/Export functionality
    - Delete models
    """
    
    PAGE_NAME = "Models"
    PAGE_ICON = "📁"
    PAGE_DESCRIPTION = "Manage trained models"
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the models page."""
        super().__init__(parent, app)
        
        # Model data (would be loaded from actual storage)
        self._models: List[Dict] = []
    
    def _create_widgets(self):
        """Create models page widgets."""
        # Title section
        self.create_title_section(
            self.content_frame,
            "Model Management",
            "View, import, export, and delete trained voice models"
        )
        
        # Toolbar
        toolbar = ttk.Frame(self.content_frame, style="TFrame")
        toolbar.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(toolbar, text="📥 Import Model", style="Secondary.TButton",
                  command=self._import_model).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar, text="📤 Export Model", style="Secondary.TButton",
                  command=self._export_model).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar, text="🗑 Delete", style="Danger.TButton",
                  command=self._delete_model).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar, text="🔄 Refresh", style="Secondary.TButton",
                  command=self._refresh_models).pack(side=tk.RIGHT)
        
        # Model list
        list_card = self.create_card(self.content_frame, "Trained Models")
        
        # Treeview for model list
        columns = ("name", "created", "size", "status")
        self.model_tree = ttk.Treeview(list_card, columns=columns, show="headings",
                                       height=10)
        
        self.model_tree.heading("name", text="Model Name")
        self.model_tree.heading("created", text="Created")
        self.model_tree.heading("size", text="Size")
        self.model_tree.heading("status", text="Status")
        
        self.model_tree.column("name", width=200)
        self.model_tree.column("created", width=150)
        self.model_tree.column("size", width=100)
        self.model_tree.column("status", width=100)
        
        self.model_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_card, orient=tk.VERTICAL,
                                 command=self.model_tree.yview)
        self.model_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.place(relx=1.0, rely=0, relheight=1.0, anchor=tk.NE)
        
        # Model details
        details_card = self.create_card(self.content_frame, "Model Details")
        
        self.details_text = tk.Text(details_card, height=6,
                                   bg=Colors.BG_INPUT,
                                   fg=Colors.TEXT_PRIMARY,
                                   font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
                                   state=tk.DISABLED)
        self.details_text.pack(fill=tk.X)
        
        # Load initial models
        self._refresh_models()
    
    def _refresh_models(self):
        """Refresh the model list."""
        # Clear existing items
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        
        # Add sample models (would load from actual storage)
        sample_models = [
            {"name": "No models found", "created": "-", "size": "-", "status": "-"},
        ]
        
        for model in sample_models:
            self.model_tree.insert("", tk.END, values=(
                model["name"],
                model["created"],
                model["size"],
                model["status"]
            ))
    
    def _import_model(self):
        """Import a model from file."""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Import Model",
            filetypes=[("Model files", "*.pth *.pt"), ("All files", "*.*")]
        )
        if filename:
            messagebox.showinfo("Info", "Model import coming soon!")
    
    def _export_model(self):
        """Export selected model to file."""
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to export.")
            return
        
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            title="Export Model",
            filetypes=[("Model files", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            messagebox.showinfo("Info", "Model export coming soon!")
    
    def _delete_model(self):
        """Delete selected model."""
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to delete.")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this model?"):
            messagebox.showinfo("Info", "Model deletion coming soon!")
