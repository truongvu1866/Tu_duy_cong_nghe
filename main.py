import sys
from PyQt6.QtWidgets import QApplication
from UI.Uicontroller import MainController

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainController()
    window.show()
    sys.exit(app.exec())
