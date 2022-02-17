from PyQt5.QtWidgets import QApplication, QLineEdit, QDialogButtonBox, QFormLayout, QDialog
from typing import List

class InputDialog(QDialog):
    def __init__(self, labels:List[str], defaults:List[str], parent=None):
        super().__init__(parent)
        
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)
        
        self.inputs = []
        for index, lab in enumerate(labels):
            line = QLineEdit(self)
            line.setText(defaults[index])
            self.inputs.append(line)
            layout.addRow(lab, self.inputs[-1])
        
        layout.addWidget(buttonBox)
        
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
    
    def getInputs(self):
        return tuple(input.text() for input in self.inputs)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    dialog = InputDialog(labels=["First","Second","Third","Fourth"])
    if dialog.exec():
        print(dialog.getInputs())
    exit(0)