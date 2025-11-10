from flet import *

def main(page:Page):
    def pick_folder_result(e:FilePickerResultEvent):
        before_path = folder_path_input.value
        folder_path_input.value = e.path if e.path else before_path
        folder_path_input.update()
    
    pick_folder_dialog = FilePicker(on_result=pick_folder_result)
    page.overlay.append(pick_folder_dialog)

    folder_path_input = TextField()
    folder_select_btn = TextButton(text="選択",icon=Icons.FOLDER,on_click=lambda _:pick_folder_dialog.get_directory_path(dialog_title="選択"))

    page.add(
        Row([folder_path_input,folder_select_btn])
    )

app(target=main)
