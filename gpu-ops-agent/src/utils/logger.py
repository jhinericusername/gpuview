from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
console = Console(highlight=False)
def info(msg: str): console.print(f"[bold cyan]INFO[/]: {msg}")
def warn(msg: str): console.print(f"[bold yellow]WARN[/]: {msg}")
def error(msg: str): console.print(f"[bold red]ERROR[/]: {msg}")
def panel(title: str, content: str): console.print(Panel.fit(content, title=title))
class Spinner:
    def __enter__(self):
        self.progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn())
        self.task = self.progress.add_task("Working…", total=None)
        self.progress.start(); return self
    def update(self, desc: str): self.progress.update(self.task, description=desc)
    def __exit__(self, exc_type, exc, tb): self.progress.stop()
