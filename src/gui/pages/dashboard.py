"""
Dashboard page for SOMA GUI.

Provides the home screen with quick actions and recent activity.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


class DashboardPage(BasePage):
    """
    Dashboard page showing overview and quick actions.
    
    Features:
    - Project info and version
    - Quick action buttons
    - Recent activity list
    - System status
    """
    
    PAGE_NAME = "Dashboard"
    PAGE_ICON = "🏠"
    PAGE_DESCRIPTION = "Home and quick actions"
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the dashboard page."""
        super().__init__(parent, app)
    
    def _create_widgets(self):
        """Create dashboard widgets."""
        # Title section
        self.create_title_section(
            self.content_frame,
            "Welcome to SOMA AI",
            "Your AI-powered cover workstation"
        )
        
        # Quick actions card
        self._create_quick_actions()
        
        # System status card
        self._create_system_status()
        
        # Recent activity card
        self._create_recent_activity()
    
    def _create_quick_actions(self):
        """Create quick action buttons section."""
        card = self.create_card(self.content_frame, "Quick Actions")
        
        # Create button grid
        button_frame = ttk.Frame(card, style="Card.TFrame")
        button_frame.pack(fill=tk.X)
        
        # Quick action buttons
        actions = [
            ("🎤", "Start Training", "Train a new voice model", self._on_start_training),
            ("🎵", "Create Cover", "Generate an AI cover", self._on_create_cover),
            ("🎼", "Separate Audio", "Split vocals and instruments", self._on_separate_audio),
            ("📁", "Manage Models", "View and manage models", self._on_manage_models),
        ]
        
        for i, (icon, title, desc, cmd) in enumerate(actions):
            btn_frame = ttk.Frame(button_frame, style="Card.TFrame")
            btn_frame.pack(side=tk.LEFT, padx=(0, 15), fill=tk.X, expand=True)
            
            # Action button
            btn = tk.Button(
                btn_frame,
                text=f"{icon}\n{title}",
                font=(Fonts.FAMILY, Fonts.SIZE_BODY, "bold"),
                bg=Colors.BG_TERTIARY,
                fg=Colors.TEXT_PRIMARY,
                activebackground=Colors.ACCENT_PRIMARY,
                activeforeground=Colors.BG_PRIMARY,
                bd=0,
                padx=20,
                pady=15,
                cursor="hand2",
                command=cmd
            )
            btn.pack(fill=tk.X)
            
            # Description
            desc_label = ttk.Label(btn_frame, text=desc, style="Muted.TLabel",
                                  wraplength=150)
            desc_label.pack(anchor=tk.W, pady=(5, 0))
    
    def _create_system_status(self):
        """Create system status section."""
        card = self.create_card(self.content_frame, "System Status")
        
        status_frame = ttk.Frame(card, style="Card.TFrame")
        status_frame.pack(fill=tk.X)
        
        # Status items
        statuses = [
            ("Device", self._get_device_info(), Colors.ACCENT_SUCCESS),
            ("Models", f"{self._get_model_count()} trained", Colors.ACCENT_INFO),
            ("Storage", self._get_storage_info(), Colors.ACCENT_WARNING),
        ]
        
        for label, value, color in statuses:
            row = ttk.Frame(status_frame, style="Card.TFrame")
            row.pack(fill=tk.X, pady=5)
            
            ttk.Label(row, text=label, style="Card.TLabel", width=15).pack(side=tk.LEFT)
            
            value_label = ttk.Label(row, text=value, style="Card.TLabel")
            value_label.configure(foreground=color)
            value_label.pack(side=tk.LEFT)
    
    def _create_recent_activity(self):
        """Create recent activity section."""
        card = self.create_card(self.content_frame, "Recent Activity")
        
        # Activity list
        activity_frame = ttk.Frame(card, style="Card.TFrame")
        activity_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sample recent items (would be populated from actual data)
        recent_items = [
            ("🎤", "Model 'Aria_v2' training completed", "2 hours ago"),
            ("🎵", "Cover 'Song_X' generated", "Yesterday"),
            ("🎼", "Audio separation completed", "3 days ago"),
        ]
        
        if not recent_items:
            ttk.Label(activity_frame, text="No recent activity",
                     style="Muted.TLabel").pack(anchor=tk.W, pady=10)
        else:
            for icon, text, time in recent_items:
                row = ttk.Frame(activity_frame, style="Card.TFrame")
                row.pack(fill=tk.X, pady=3)
                
                ttk.Label(row, text=icon, style="Card.TLabel").pack(side=tk.LEFT, padx=(0, 10))
                ttk.Label(row, text=text, style="Card.TLabel").pack(side=tk.LEFT)
                ttk.Label(row, text=time, style="Muted.TLabel").pack(side=tk.RIGHT)
    
    def _get_device_info(self) -> str:
        """Get current device information."""
        try:
            import torch
            if torch.cuda.is_available():
                return f"CUDA ({torch.cuda.get_device_name(0)})"
            return "CPU"
        except ImportError:
            return "CPU (torch not available)"
    
    def _get_model_count(self) -> int:
        """Get the number of trained models."""
        # Would query actual model storage
        return 0
    
    def _get_storage_info(self) -> str:
        """Get storage usage information."""
        # Would calculate actual storage usage
        return "0 MB used"
    
    def _on_start_training(self):
        """Navigate to training page."""
        if self.app:
            self.app.navigate_to("training")
    
    def _on_create_cover(self):
        """Navigate to inference page."""
        if self.app:
            self.app.navigate_to("inference")
    
    def _on_separate_audio(self):
        """Navigate to separation page."""
        if self.app:
            self.app.navigate_to("separation")
    
    def _on_manage_models(self):
        """Navigate to models page."""
        if self.app:
            self.app.navigate_to("models")
    
    def _on_visible(self):
        """Called when page becomes visible - refresh data."""
        # Refresh status information
        pass
