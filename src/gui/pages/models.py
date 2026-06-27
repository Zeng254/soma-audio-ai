"""
模型管理页面 - SOMA GUI

提供已训练声音模型的管理界面。
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List, Dict
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


class ModelsPage(BasePage):
    """
    模型管理页面 - 管理已训练的声音模型。
    
    功能：
    - 模型列表及详情
    - 导入/导出功能
    - 删除模型
    """
    
    PAGE_NAME = "模型管理"
    PAGE_ICON = "📁"
    PAGE_DESCRIPTION = "管理已训练的模型"
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """初始化模型管理页面。"""
        # 模型数据（实际会从存储加载）
        self._models: List[Dict] = []
        
        super().__init__(parent, app)
    
    def _create_widgets(self):
        """创建模型管理页面组件。"""
        # 标题区域
        self.create_title_section(
            self.content_frame,
            "模型管理",
            "查看、导入、导出和删除已训练的声音模型"
        )
        
        # 工具栏
        toolbar = ttk.Frame(self.content_frame, style="TFrame")
        toolbar.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(toolbar, text="📥 导入模型", style="Secondary.TButton",
                  command=self._import_model).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar, text="📤 导出模型", style="Secondary.TButton",
                  command=self._export_model).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar, text="🗑 删除", style="Danger.TButton",
                  command=self._delete_model).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar, text="🔄 刷新", style="Secondary.TButton",
                  command=self._refresh_models).pack(side=tk.RIGHT)
        
        # 模型列表
        list_card = self.create_card(self.content_frame, "已训练模型")
        
        # 模型列表 Treeview
        columns = ("name", "created", "size", "status")
        self.model_tree = ttk.Treeview(list_card, columns=columns, show="headings",
                                       height=10)
        
        self.model_tree.heading("name", text="模型名称")
        self.model_tree.heading("created", text="创建时间")
        self.model_tree.heading("size", text="大小")
        self.model_tree.heading("status", text="状态")
        
        self.model_tree.column("name", width=200)
        self.model_tree.column("created", width=150)
        self.model_tree.column("size", width=100)
        self.model_tree.column("status", width=100)
        
        self.model_tree.pack(fill=tk.BOTH, expand=True)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_card, orient=tk.VERTICAL,
                                 command=self.model_tree.yview)
        self.model_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.place(relx=1.0, rely=0, relheight=1.0, anchor=tk.NE)
        
        # 模型详情
        details_card = self.create_card(self.content_frame, "模型详情")
        
        self.details_text = tk.Text(details_card, height=6,
                                   bg=Colors.BG_INPUT,
                                   fg=Colors.TEXT_PRIMARY,
                                   font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
                                   state=tk.DISABLED)
        self.details_text.pack(fill=tk.X)
        
        # 加载初始模型
        self._refresh_models()
    
    def _refresh_models(self):
        """刷新模型列表。"""
        # 清除现有项目
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        
        # 添加示例模型（实际会从存储加载）
        sample_models = [
            {"name": "暂无模型", "created": "-", "size": "-", "status": "-"},
        ]
        
        for model in sample_models:
            self.model_tree.insert("", tk.END, values=(
                model["name"],
                model["created"],
                model["size"],
                model["status"]
            ))
    
    def _import_model(self):
        """从文件导入模型。"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="导入模型",
            filetypes=[("模型文件", "*.pth *.pt"), ("所有文件", "*.*")]
        )
        if filename:
            messagebox.showinfo("提示", "模型导入功能即将推出！")
    
    def _export_model(self):
        """导出选中的模型到文件。"""
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要导出的模型。")
            return
        
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            title="导出模型",
            filetypes=[("模型文件", "*.pth"), ("所有文件", "*.*")]
        )
        if filename:
            messagebox.showinfo("提示", "模型导出功能即将推出！")
    
    def _delete_model(self):
        """删除选中的模型。"""
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要删除的模型。")
            return
        
        if messagebox.askyesno("确认", "确定要删除此模型吗？"):
            messagebox.showinfo("提示", "模型删除功能即将推出！")
