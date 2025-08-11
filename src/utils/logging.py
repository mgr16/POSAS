from rich.console import Console
from rich.table import Table

console = Console()

def log_metrics(epoch: int, fold: int, metrics: dict):
    table = Table(title=f"Fold {fold} - Epoch {epoch}")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)
