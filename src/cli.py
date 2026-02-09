import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import uvicorn
from src.data_pipeline import preprocessing
from src.training_pipeline import train
from src.api.api_app import app

def main()-> None:

    parser= argparse.ArgumentParser(description="Generous Tip Giver")
    
    parser.add_argument("command", choices=["process", "train", "route"],
                        help="Choose an action: process, train, or route")
    
    args= parser.parse_args()

    if args.command== "process":
        preprocessing.main()
    elif args.command == "train":
        train.tune_train()
    elif args.command == "route":
        uvicorn.run(app,host='0.0.0.0',port=8000)

if __name__=="__main__":
    main()

"""
import os
import sys
import logging

# --- STEP 1: Avoid Sys Path Hacks ---
# Instead of sys.path.insert, use a 'pip install -e .' (editable install) 
# or a proper PYTHONPATH. But for this file, we will keep it clean.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
except ImportError:
    print("Run: pip install typer rich")
    sys.exit(1)

# Initialize Typer and Rich Console
cli = typer.Typer(help="Generous Tip Giver: End-to-End ML Pipeline Control")
console = Console()

# --- STEP 2: Lazy Loading Strategy ---
# We define functions that import internal modules ONLY when called.
# This makes the CLI near-instant to respond.

@cli.command()
def process(
    input_path: str = typer.Option("data/raw", help="Path to raw data"),
    output_path: str = typer.Option("data/processed", help="Path to save processed data")
):
    """
#    Step 1: Run the Data Preprocessing Pipeline.
"""
    console.print(Panel(f"[bold blue]Processing Data[/]\nInput: {input_path}\nOutput: {output_path}"))
    from src.data_pipeline.preprocessing import main as run_prep
    run_prep()
    console.print("[bold green]✓ Preprocessing Complete.[/]")

@cli.command()
def train(
    epochs: int = typer.Option(10, help="Number of training epochs"),
    model_name: str = typer.Option("tipper_v1", help="Name for the exported model")
):
    """
#    Step 2: Run the Model Training Pipeline.
"""
    console.print(Panel(f"[bold magenta]Starting Training[/]\nModel: {model_name} | Epochs: {epochs}"))
    from src.training_pipeline.train import main as run_train
    run_train(epochs=epochs, name=model_name)
    console.print("[bold green]✓ Training Complete. Model saved.[/]")

@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Binding host"),
    port: int = typer.Option(8000, help="API Port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development")
):
    """
#    Step 3: Launch the FastAPI Production Server.
"""
    import uvicorn
    console.print(Panel(f"[bold green]Launching API Server[/]\nURL: http://{host}:{port}"))
    # We load the app string-based to allow for reload functionality
    uvicorn.run("src.api.api_app:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    cli()"""