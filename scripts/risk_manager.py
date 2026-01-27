import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

class RiskManager:
    def __init__(self):
        pass
        
    def optimize_stop_loss(self, trades_file: str, step: float = 0.01):
        """
        AI Optimization (Grid Search) to find the best Trailing Stop Loss.
        Replays the history: If Max Drawdown > Stop Loss, we exit early.
        """
        try:
            df = pd.read_csv(trades_file)
        except Exception as e:
            console.print(f"[red]Could not read {trades_file}: {e}[/red]")
            return
            
        console.print(f"[bold]Optimizing Risk Management on {len(df)} historical trades...[/bold]")
        
        # Grid Search Range: 10% to 40%
        stop_losses = np.arange(0.10, 0.41, step)
        
        best_sl = 0.0
        best_return = -np.inf
        results = []
        
        # Baseline (No Stop Loss)
        baseline_return = df['PnL'].sum()
        baseline_drawdowns = df['Max DD %'].min()
        
        for sl in stop_losses:
            # Simulation
            # If Max DD % (which is negative, e.g. -20%) is lower than -SL (e.g. -15%), trade stops out.
            # Condition: Max DD % <= -SL * 100
            
            # Since Max DD % in CSV is e.g. -32.5, and SL is 0.20 (20%)
            # Trigger = -32.5 <= -20.0
            
            stopped_out_mask = df['Max DD %'] <= -(sl * 100)
            
            # Outcome PnL
            # For stopped out trades: PnL becomes -SL (approx). 
            # (We assume we exit exactly at the stop. Slippage ignored for prototype)
            
            # PnL Vector
            simulated_pnl = df['PnL'].copy()
            simulated_pnl[stopped_out_mask] = -sl
            
            total_return = simulated_pnl.sum()
            
            results.append({
                "Stop Loss": sl,
                "Total Return": total_return,
                "Trades Stopped": stopped_out_mask.sum()
            })
            
            if total_return > best_return:
                best_return = total_return
                best_sl = sl
                
        # Display Results
        res_df = pd.DataFrame(results)
        
        console.print(f"\n[bold green]Optimization Complete![/bold green]")
        console.print(f"Baseline Return (No SL): [magenta]{baseline_return*100:.2f}%[/magenta]")
        console.print(f"Best Stop Loss found: [bold cyan]{best_sl*100:.1f}%[/bold cyan]")
        console.print(f"Optimized Return: [bold green]{best_return*100:.2f}%[/bold green]")
        
        improvement = best_return - baseline_return
        console.print(f"Improvement: [green]+{improvement*100:.2f}%[/green]")
        
        # Show Top 5 configs
        console.print("\n[bold]Top 5 Stop Loss Configurations:[/bold]")
        top_5 = res_df.nlargest(5, 'Total Return')
        
        table = Table()
        table.add_column("Stop Loss %")
        table.add_column("Total Return")
        table.add_column("Trades Stopped")
        
        for _, row in top_5.iterrows():
            table.add_row(
                f"{row['Stop Loss']*100:.1f}%",
                f"{row['Total Return']*100:.2f}%",
                f"{int(row['Trades Stopped'])}"
            )
            
        console.print(table)
        
        return best_sl

if __name__ == "__main__":
    rm = RiskManager()
    # Expects trades_log.csv in current directory or parent
    import os
    if os.path.exists("trades_log.csv"):
        rm.optimize_stop_loss("trades_log.csv")
    elif os.path.exists("../trades_log.csv"):
        rm.optimize_stop_loss("../trades_log.csv")
    elif os.path.exists("trades_log (1).csv"):
         rm.optimize_stop_loss("trades_log (1).csv")
    else:
        console.print("[red]trades_log.csv not found. Please place it in the folder.[/red]")
