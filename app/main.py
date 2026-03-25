import os
import sys



from PyQt6.QtWidgets import QApplication
from ui.pages.login import LoginPage
from core.path_manager import BASE_DIR


app = QApplication(sys.argv)
login = LoginPage()
login.show()
sys.exit(app.exec())