from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow
from ui.pages.process_tab import ProcessTab
from ui.pages.project_tab import ProjectTab
from ui.pages.get_data_tab import GetDataTab
from core.path_manager import BASE_DIR
import os


class MenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(BASE_DIR, "ui", 'ui', "menu.ui")
        uic.loadUi(ui_path, self)

        self.tab_menu.setTabsClosable(True)
        self.tab_menu.tabCloseRequested.connect(self.close_tab)


        self.tab_menu.tabBar().setTabButton(
            0,
            self.tab_menu.tabBar().ButtonPosition.RightSide,
            None
        )

        self.btn_process_page.clicked.connect(lambda: self.open_tab("Process"))
        self.btn_project_page.clicked.connect(lambda: self.open_tab("Project"))
        self.btn_get_data.clicked.connect(lambda: self.open_tab("GetData"))

        self.btn_log_out.triggered.connect(self.back_login_tab)
        




    def open_tab(self, name):
        for i in range(self.tab_menu.count()):
            if self.tab_menu.tabText(i) == name:
                self.tab_menu.setCurrentIndex(i)
                return

        if name == "Process":
            tab = ProcessTab()
        elif name == "Project":
            tab = ProjectTab()
        elif name == "GetData":
            tab = GetDataTab()

        else:
            return
        
        
        self.tab_menu.addTab(tab, name)
        self.tab_menu.setCurrentWidget(tab)

    

    def close_tab(self, index):
        if index == 0:
            return

        widget = self.tab_menu.widget(index)
        print(f"[MenuWindow] Closing tab: {type(widget)}")               # ← thêm
        print(f"[MenuWindow] has stop_all_threads: {hasattr(widget, 'stop_all_threads')}")  # ← thêm

        if hasattr(widget, "stop_all_threads"):
            widget.stop_all_threads()

        self.tab_menu.removeTab(index)
        widget.deleteLater()





    def back_login_tab(self):
        from ui.pages.login import LoginPage



            
        # Đóng toàn bộ tab (trừ tab 0)
        while self.tab_menu.count() > 1:
            self.close_tab(1)

        self.login = LoginPage()
        self.login.show()

        self.close()
