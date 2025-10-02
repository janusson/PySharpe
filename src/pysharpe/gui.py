"""Simple Tkinter based GUI for running common PySharpe workflows."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable
from pathlib import Path

try:  # pragma: no cover - Tkinter availability differs per platform
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover - surfaced at call sites
    raise RuntimeError("Tkinter is required to use the PySharpe GUI") from exc

from . import data_collector, portfolio_optimization


class PySharpeGUI:
    """Tkinter front-end for downloading and optimising portfolios."""

    POLL_INTERVAL_MS = 200

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PySharpe Portfolio Toolkit")

        self.status_queue: queue.Queue[str] = queue.Queue()

        self._build_widgets()
        self._poll_queue()

    # ------------------------------------------------------------------
    # Widget construction helpers
    def _build_widgets(self) -> None:
        container = ttk.Notebook(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        download_frame = ttk.Frame(container, padding=10)
        optimise_frame = ttk.Frame(container, padding=10)
        container.add(download_frame, text="Download")
        container.add(optimise_frame, text="Optimise")

        self._build_download_tab(download_frame)
        self._build_optimise_tab(optimise_frame)

        self.log_text = tk.Text(self.root, height=10, wrap="word", state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

    def _build_download_tab(self, frame: ttk.Frame) -> None:
        ticker_box = ttk.LabelFrame(frame, text="Ticker symbols")
        ticker_box.pack(fill=tk.BOTH, expand=True)

        self.ticker_text = tk.Text(ticker_box, height=8, width=60)
        self.ticker_text.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=5
        )

        ticker_scroll = ttk.Scrollbar(ticker_box, command=self.ticker_text.yview)
        ticker_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        self.ticker_text.configure(yscrollcommand=ticker_scroll.set)

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)

        load_btn = ttk.Button(
            button_frame, text="Load from file", command=self._load_tickers_from_file
        )
        load_btn.pack(side=tk.LEFT)

        self.portfolio_name_var = tk.StringVar()
        ttk.Label(button_frame, text="Portfolio name:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(button_frame, textvariable=self.portfolio_name_var, width=20).pack(
            side=tk.LEFT
        )

        directory_frame = ttk.Frame(frame)
        directory_frame.pack(fill=tk.X, pady=5)

        self.price_dir_var = tk.StringVar(value=str(data_collector.PRICE_HISTORY_DIR))
        self.export_dir_var = tk.StringVar(value=str(data_collector.EXPORT_DIR))

        self._add_directory_selector(
            directory_frame, "Price history directory", self.price_dir_var
        )
        self._add_directory_selector(
            directory_frame, "Export directory", self.export_dir_var
        )

        options_frame = ttk.Frame(frame)
        options_frame.pack(fill=tk.X, pady=5)

        self.period_var = tk.StringVar(value="max")
        self.interval_var = tk.StringVar(value="1d")
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()

        self._add_labeled_entry(options_frame, "Period", self.period_var, 8)
        self._add_labeled_entry(options_frame, "Interval", self.interval_var, 8)
        self._add_labeled_entry(options_frame, "Start", self.start_var, 12)
        self._add_labeled_entry(options_frame, "End", self.end_var, 12)

        action_frame = ttk.Frame(frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))

        download_btn = ttk.Button(
            action_frame, text="Download prices", command=self._start_download
        )
        download_btn.pack(side=tk.LEFT)

        collate_btn = ttk.Button(
            action_frame, text="Collate prices", command=self._start_collate
        )
        collate_btn.pack(side=tk.LEFT, padx=(10, 0))

    def _build_optimise_tab(self, frame: ttk.Frame) -> None:
        intro = ttk.Label(
            frame,
            text=(
                "Optimise a portfolio using an existing collated price file."
                " Results are exported alongside any generated plots."
            ),
            wraplength=500,
            justify=tk.LEFT,
        )
        intro.pack(fill=tk.X, pady=(0, 10))

        form = ttk.Frame(frame)
        form.pack(fill=tk.X)

        self.optimise_name_var = tk.StringVar()
        self.collated_dir_var = tk.StringVar(value=str(data_collector.EXPORT_DIR))
        self.output_dir_var = tk.StringVar(value=str(Path("reports")))
        self.optimise_start_var = tk.StringVar()
        self.make_plot_var = tk.BooleanVar(value=True)

        self._add_labeled_entry(form, "Portfolio name", self.optimise_name_var, 20)
        self._add_directory_selector(form, "Collated directory", self.collated_dir_var)
        self._add_directory_selector(form, "Output directory", self.output_dir_var)
        self._add_labeled_entry(
            form, "Time constraint (optional start date)", self.optimise_start_var, 20
        )

        plot_toggle = ttk.Checkbutton(
            form, text="Create performance plot", variable=self.make_plot_var
        )
        plot_toggle.pack(anchor=tk.W, pady=(5, 0))

        action_frame = ttk.Frame(frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))

        optimise_btn = ttk.Button(
            action_frame, text="Optimise portfolio", command=self._start_optimise
        )
        optimise_btn.pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Utility helpers
    def _add_labeled_entry(
        self,
        frame: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        width: int,
    ) -> None:
        container = ttk.Frame(frame)
        container.pack(fill=tk.X, pady=2)
        ttk.Label(container, text=label, width=28, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Entry(container, textvariable=variable, width=width).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

    def _add_directory_selector(
        self, frame: ttk.Frame, label: str, variable: tk.StringVar
    ) -> None:
        container = ttk.Frame(frame)
        container.pack(fill=tk.X, pady=2)
        ttk.Label(container, text=label, width=28, anchor=tk.W).pack(side=tk.LEFT)
        entry = ttk.Entry(container, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            container,
            text="Browse",
            command=lambda var=variable: self._choose_directory(var),
        ).pack(side=tk.LEFT, padx=(5, 0))

    def _choose_directory(self, variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(initialdir=variable.get() or ".")
        if selected:
            variable.set(selected)

    def _queue_message(self, message: str) -> None:
        self.status_queue.put(message)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def _poll_queue(self) -> None:
        while True:
            try:
                message = self.status_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_log(message)
        self.root.after(self.POLL_INTERVAL_MS, self._poll_queue)

    # ------------------------------------------------------------------
    # Actions
    def _collect_tickers(self) -> list[str]:
        raw = self.ticker_text.get("1.0", tk.END)
        symbols: set[str] = set()
        for token in raw.replace(",", "\n").splitlines():
            cleaned = token.strip().upper()
            if cleaned:
                symbols.add(cleaned)
        return sorted(symbols)

    def _load_tickers_from_file(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select portfolio file",
        )
        if not file_path:
            return
        try:
            tickers = data_collector.read_tickers_from_file(Path(file_path))
        except Exception as exc:  # pragma: no cover - UI feedback
            messagebox.showerror("Unable to load tickers", str(exc))
            return

        self.ticker_text.delete("1.0", tk.END)
        self.ticker_text.insert(tk.END, "\n".join(sorted(tickers)))
        if not self.portfolio_name_var.get():
            self.portfolio_name_var.set(Path(file_path).stem)
        self._queue_message(f"Loaded {len(tickers)} tickers from {file_path}")

    def _start_download(self) -> None:
        tickers = self._collect_tickers()
        if not tickers:
            messagebox.showwarning(
                "No tickers", "Please enter at least one ticker symbol."
            )
            return

        kwargs = {
            "price_history_dir": Path(self.price_dir_var.get()).expanduser(),
            "period": self.period_var.get().strip() or "max",
            "interval": self.interval_var.get().strip() or "1d",
        }
        start = self.start_var.get().strip()
        end = self.end_var.get().strip()
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end

        self._queue_message(f"Starting download for {len(tickers)} tickers...")
        self._run_in_thread(self._download_task, tickers, kwargs)

    def _download_task(
        self, tickers: Iterable[str], kwargs: dict[str, str | Path]
    ) -> None:
        try:
            results = data_collector.download_portfolio_prices(tickers, **kwargs)
        except Exception as exc:  # pragma: no cover - API/network errors
            self._queue_message(f"Download failed: {exc}")
            return

        if results:
            self._queue_message(f"Downloaded data for {len(results)} tickers.")
        else:
            self._queue_message(
                "No data downloaded. Check ticker symbols or connection."
            )

    def _start_collate(self) -> None:
        tickers = self._collect_tickers()
        if not tickers:
            messagebox.showwarning(
                "No tickers", "Please enter at least one ticker symbol."
            )
            return

        portfolio_name = self.portfolio_name_var.get().strip() or "portfolio"
        kwargs = {
            "price_history_dir": Path(self.price_dir_var.get()).expanduser(),
            "tickers": tickers,
            "export_dir": Path(self.export_dir_var.get()).expanduser(),
        }

        self._queue_message(f"Collating prices for {portfolio_name}...")
        self._run_in_thread(self._collate_task, portfolio_name, kwargs)

    def _collate_task(self, portfolio_name: str, kwargs: dict[str, object]) -> None:
        try:
            frame = data_collector.collate_prices(portfolio_name, **kwargs)
        except Exception as exc:  # pragma: no cover - propagate via UI
            self._queue_message(f"Collation failed: {exc}")
            return

        if frame.empty:
            self._queue_message("Collation completed but no data was produced.")
        else:
            ticker_count = len(frame.columns)
            row_count = len(frame)
            self._queue_message(
                f"Collation complete for {portfolio_name}: {ticker_count} tickers,"
            )
            self._queue_message(f"{row_count} rows of price history written.")

    def _start_optimise(self) -> None:
        portfolio_name = self.optimise_name_var.get().strip()
        if not portfolio_name:
            messagebox.showwarning(
                "Missing portfolio name", "Enter a portfolio to optimise."
            )
            return

        kwargs = {
            "collated_dir": Path(self.collated_dir_var.get()).expanduser(),
            "output_dir": Path(self.output_dir_var.get()).expanduser(),
            "time_constraint": self.optimise_start_var.get().strip() or None,
            "make_plot": self.make_plot_var.get(),
        }

        self._queue_message(f"Optimising portfolio {portfolio_name}...")
        self._run_in_thread(self._optimise_task, portfolio_name, kwargs)

    def _optimise_task(self, portfolio_name: str, kwargs: dict[str, object]) -> None:
        try:
            weights, performance = portfolio_optimization.optimise_portfolio(
                portfolio_name, **kwargs
            )
        except FileNotFoundError as exc:  # pragma: no cover - UI feedback
            self._queue_message(f"Optimisation failed: {exc}")
            return
        except ValueError as exc:  # pragma: no cover - UI feedback
            self._queue_message(f"Optimisation failed: {exc}")
            return
        except Exception as exc:  # pragma: no cover - propagate via UI
            self._queue_message(f"Optimisation error: {exc}")
            return

        expected, volatility, sharpe = performance
        summary = (
            "Optimisation complete: "
            f"expected return {expected:.2%}, volatility {volatility:.2%}, "
            f"sharpe {sharpe:.2f}."
        )
        self._queue_message(summary)
        non_zero = {ticker: weight for ticker, weight in weights.items() if weight > 0}
        if non_zero:
            allocations = ", ".join(
                f"{ticker}={weight:.2%}" for ticker, weight in sorted(non_zero.items())
            )
            self._queue_message(f"Weights: {allocations}")

    def _run_in_thread(self, target, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        thread = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
        thread.start()


def main() -> None:
    root = tk.Tk()
    PySharpeGUI(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
