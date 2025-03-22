from rich.console import Console
from rich.theme import Theme
from datetime import datetime
from typing import Any

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green"
})

console = Console(theme=custom_theme)

class Logger:
    @staticmethod
    def info(message: Any, emoji: str = "ℹ️") -> None:
        console.print(f"{emoji} [{datetime.now().strftime('%H:%M:%S')}] [info]{message}[/]")
    
    @staticmethod
    def success(message: Any, emoji: str = "✅") -> None:
        console.print(f"{emoji} [{datetime.now().strftime('%H:%M:%S')}] [success]{message}[/]")
    
    @staticmethod
    def warning(message: Any, emoji: str = "⚠️") -> None:
        console.print(f"{emoji} [{datetime.now().strftime('%H:%M:%S')}] [warning]{message}[/]")
    
    @staticmethod
    def error(message: Any, emoji: str = "❌") -> None:
        console.print(f"{emoji} [{datetime.now().strftime('%H:%M:%S')}] [error]{message}[/]")
    
    @staticmethod
    def step(step_number: int, total_steps: int, message: Any, emoji: str = "🔄") -> None:
        console.print(f"\n{emoji} [{datetime.now().strftime('%H:%M:%S')}] [info]Step {step_number}/{total_steps}: {message}[/]")

    @staticmethod
    def api(method: str, url: str, emoji: str = "🌐") -> None:
        console.print(f"{emoji} [{datetime.now().strftime('%H:%M:%S')}] [info]{method} {url}[/]")

    @staticmethod
    def processing(message: Any, emoji: str = "⚙️") -> None:
        console.print(f"{emoji} [{datetime.now().strftime('%H:%M:%S')}] [info]{message}[/]")

    @staticmethod
    def data(message: Any, emoji: str = "📊") -> None:
        console.print(f"{emoji} [{datetime.now().strftime('%H:%M:%S')}] [info]{message}[/]") 