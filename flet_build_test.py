import flet as ft

def main(page: ft.Page):
    page.title = "Test App"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    page.add(
        ft.Text("Build Test Successful!", size=30)
    )

ft.app(main)