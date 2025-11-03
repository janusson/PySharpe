import datetime as dt
import io
from types import SimpleNamespace

import pandas as pd
import pytest

import app


class ColumnContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class ContainerContext:
    def __init__(self, owner: "StreamlitStub") -> None:
        self.owner = owner

    def __enter__(self) -> "StreamlitStub":
        return self.owner

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class Placeholder:
    def __init__(self, owner: "StreamlitStub") -> None:
        self.owner = owner
        self.emptied = False

    def container(self) -> ContainerContext:
        return ContainerContext(self.owner)

    def empty(self) -> None:
        self.emptied = True


class SidebarAPI:
    def __init__(self, owner: "StreamlitStub") -> None:
        self.owner = owner
        self.uploaded_file: io.StringIO | None = None
        self.text_value = "AAPL,MSFT"
        self.start_date = dt.date(2024, 1, 1)
        self.end_date = dt.date(2024, 6, 1)
        self.number_values = {
            "Initial Investment": 1000.0,
            "Monthly Contribution": 250.0,
        }
        self.slider_values = {
            "Months": 240,
            "Annual Return Rate": 0.08,
        }
        self.checkbox_values = {}

    def header(self, *_args, **_kwargs) -> None:
        return None

    def file_uploader(self, *_args, **_kwargs) -> io.StringIO | None:
        return self.uploaded_file

    def text_input(self, *_args, **_kwargs) -> str:
        return self.text_value

    def date_input(self, label: str, default: dt.date) -> dt.date:
        return self.start_date if label == "Start" else self.end_date

    def number_input(self, label: str, **kwargs) -> float:
        return self.number_values.get(label, kwargs.get("value", 0.0))

    def slider(self, label: str, **kwargs):
        return self.slider_values.get(label, kwargs.get("value"))

    def checkbox(self, label: str, value: bool = False, **_kwargs) -> bool:
        return self.checkbox_values.get(label, value)

    def info(self, message: str) -> None:
        self.owner.info(message)

    def warning(self, message: str) -> None:
        self.owner.warning(message)

    def error(self, message: str) -> None:
        self.owner.warning(message)


class StreamlitStub:
    def __init__(self) -> None:
        self.info_calls: list[str] = []
        self.warning_calls: list[str] = []
        self.success_calls: list[str] = []
        self.line_chart_calls: list[tuple[pd.DataFrame, dict]] = []
        self.bar_chart_calls: list[tuple[pd.Series, dict]] = []
        self.altair_chart_calls: list[tuple[object, dict]] = []
        self.metric_calls: list[tuple[str, str]] = []
        self.dataframe_calls: list[object] = []
        self.download_button_calls: list[tuple[str, dict]] = []
        self.subheader_calls: list[str] = []
        self.title_calls: list[str] = []
        self.write_calls: list[str] = []
        self.page_config_calls: list[tuple[tuple, dict]] = []
        self.placeholder_calls: list[Placeholder] = []
        self.button_states: dict[str, bool] = {}
        self.sidebar = SidebarAPI(self)
        self.session_state: dict[str, float | bool] = {
            "dca_rate_default": 0.08,
            "dca_rate_override": False,
            "dca_rate_value": 0.08,
            "dca_rate_pending_reset": False,
        }

    def cache_data(self, **_kwargs):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            wrapper.__wrapped__ = func  # type: ignore[attr-defined]
            return wrapper

        return decorator

    def info(self, message: str) -> None:
        self.info_calls.append(message)

    def warning(self, message: str) -> None:
        self.warning_calls.append(message)

    def success(self, message: str) -> None:
        self.success_calls.append(message)

    def set_page_config(self, *args, **kwargs) -> None:
        self.page_config_calls.append((args, kwargs))

    def title(self, message: str) -> None:
        self.title_calls.append(message)

    def write(self, message: str) -> None:
        self.write_calls.append(message)

    def caption(self, message: str) -> None:
        self.write_calls.append(message)

    def subheader(self, message: str) -> None:
        self.subheader_calls.append(message)

    def line_chart(self, data: pd.DataFrame, **kwargs) -> None:
        self.line_chart_calls.append((data, kwargs))

    def bar_chart(self, data: pd.Series, **kwargs) -> None:
        self.bar_chart_calls.append((data, kwargs))

    def altair_chart(self, chart: object, **kwargs) -> None:
        self.altair_chart_calls.append((chart, kwargs))

    def metric(self, label: str, value: str) -> None:
        self.metric_calls.append((label, value))

    def dataframe(self, value: object) -> None:
        self.dataframe_calls.append(value)

    def download_button(self, label: str, **kwargs) -> None:
        self.download_button_calls.append((label, kwargs))

    def columns(self, spec: int) -> list[ColumnContext]:
        return [ColumnContext() for _ in range(spec)]

    def empty(self) -> Placeholder:
        placeholder = Placeholder(self)
        self.placeholder_calls.append(placeholder)
        return placeholder

    def button(self, label: str, **kwargs) -> bool:  # noqa: ARG002 - kwargs unused
        return self.button_states.get(label, False)

    def set_button_state(self, label: str, value: bool) -> None:
        self.button_states[label] = value


class DummyChart:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.inner_radius: int | None = None
        self.encoding: dict | None = None
        self.properties_args: dict | None = None

    def mark_arc(self, innerRadius: int) -> "DummyChart":
        self.inner_radius = innerRadius
        return self

    def encode(self, **kwargs) -> "DummyChart":
        self.encoding = kwargs
        return self

    def properties(self, **kwargs) -> "DummyChart":
        self.properties_args = kwargs
        return self


class DummyAlt:
    def Chart(self, data: pd.DataFrame) -> DummyChart:  # noqa: N802
        return DummyChart(data)

    def Theta(self, field: str, type: str) -> tuple[str, str, str]:  # noqa: N802
        return ("theta", field, type)

    def Color(self, field: str, type: str) -> tuple[str, str, str]:  # noqa: N802
        return ("color", field, type)

    def Tooltip(self, field: str, title: str, format: str | None = None) -> tuple[str, str, str, str | None]:  # noqa: N802
        return ("tooltip", field, title, format)


@pytest.fixture
def streamlit_stub(monkeypatch: pytest.MonkeyPatch) -> StreamlitStub:
    stub = StreamlitStub()
    monkeypatch.setattr(app, "st", stub)
    return stub


@pytest.fixture
def alt_stub(monkeypatch: pytest.MonkeyPatch) -> DummyAlt:
    stub = DummyAlt()
    monkeypatch.setattr(app, "alt", stub)
    return stub


def test_plot_cumulative_returns_empty_data(streamlit_stub: StreamlitStub) -> None:
    app.plot_cumulative_returns(pd.DataFrame())

    assert streamlit_stub.info_calls == ["No price data available to plot cumulative returns."]
    assert not streamlit_stub.line_chart_calls


def test_plot_cumulative_returns_insufficient_history(streamlit_stub: StreamlitStub) -> None:
    price_frame = pd.DataFrame({"AAPL": [100.0]}, index=pd.date_range("2023-01-01", periods=1))

    app.plot_cumulative_returns(price_frame)

    assert streamlit_stub.info_calls[-1] == "Not enough price history to compute cumulative returns."


def test_plot_cumulative_returns_renders_chart(streamlit_stub: StreamlitStub) -> None:
    price_frame = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 103.0],
            "MSFT": [200.0, 198.0, 202.0],
        },
        index=pd.date_range("2023-01-01", periods=3, freq="D"),
    )

    app.plot_cumulative_returns(price_frame)

    assert len(streamlit_stub.line_chart_calls) == 1
    chart_data, kwargs = streamlit_stub.line_chart_calls[0]
    assert not chart_data.empty
    assert "height" in kwargs and kwargs["height"] == 320


def test_plot_weights_with_positive_allocations(streamlit_stub: StreamlitStub, alt_stub: DummyAlt) -> None:
    weights = SimpleNamespace(allocations={"AAPL": 0.6, "MSFT": 0.4})

    app.plot_weights(weights)

    assert len(streamlit_stub.bar_chart_calls) == 1
    assert len(streamlit_stub.altair_chart_calls) == 1
    chart, kwargs = streamlit_stub.altair_chart_calls[0]
    assert isinstance(chart, DummyChart)
    assert chart.data["Ticker"].tolist() == ["AAPL", "MSFT"]
    assert kwargs.get("use_container_width")


def test_plot_weights_no_positive_allocations(streamlit_stub: StreamlitStub, alt_stub: DummyAlt) -> None:
    weights = SimpleNamespace(allocations={"AAPL": 0.0, "MSFT": -0.2})

    app.plot_weights(weights)

    assert streamlit_stub.info_calls[-1] == "No positive weights to display."
    assert not streamlit_stub.bar_chart_calls
    assert not streamlit_stub.altair_chart_calls


def test_plot_weights_altair_missing(streamlit_stub: StreamlitStub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "alt", None)
    weights = SimpleNamespace(allocations={"AAPL": 0.7, "MSFT": 0.3})

    app.plot_weights(weights)

    assert "Install the 'altair' package" in streamlit_stub.info_calls[-1]
    assert len(streamlit_stub.bar_chart_calls) == 1


def test_render_metrics_table_returns_dataframe(streamlit_stub: StreamlitStub) -> None:
    results = app.MetricResults(
        returns=pd.DataFrame(),
        expected=pd.Series([0.1, 0.2], index=["AAPL", "MSFT"]),
        volatility=pd.Series([0.15, 0.25], index=["AAPL", "MSFT"]),
        sharpe=pd.Series([0.8, 0.9], index=["AAPL", "MSFT"]),
    )

    summary = app.render_metrics_table(results)

    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == ["expected_return", "annual_volatility", "sharpe_ratio"]
    assert streamlit_stub.dataframe_calls  # Styler object recorded


def test_render_dca_projection_emits_metrics(monkeypatch: pytest.MonkeyPatch, streamlit_stub: StreamlitStub) -> None:
    class FakeProjection:
        months = [0, 1, 2]
        balances = [1000.0, 1100.0, 1200.0]
        contributions = [1000.0, 1200.0, 1400.0]

        @staticmethod
        def final_balance() -> float:
            return 1200.0

        @staticmethod
        def final_contribution() -> float:
            return 1400.0

    monkeypatch.setattr(app, "simulate_dca", lambda **kwargs: FakeProjection())

    projection_df = app.render_dca_projection(3, 1000.0, 100.0, 0.08)

    assert list(projection_df.columns) == ["Months", "Balance", "Contributions"]
    assert len(streamlit_stub.line_chart_calls) == 1
    assert streamlit_stub.metric_calls[0] == ("Final Balance", "$1,200.00")
    assert streamlit_stub.metric_calls[1] == ("Total Contributions", "$1,400.00")


def test_sidebar_controls_download_flow(monkeypatch: pytest.MonkeyPatch, streamlit_stub: StreamlitStub, alt_stub: DummyAlt) -> None:
    streamlit_stub.sidebar.text_value = "AAPL, MSFT"
    streamlit_stub.sidebar.slider_values["Annual Return Rate"] = 0.1

    price_frame = pd.DataFrame(
        {
            "AAPL Close": [100.0, 101.0],
            "AAPL Volume": [1_000, 1_050],
            "MSFT Close": [200.0, 203.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    preview_frame = price_frame.drop(columns=["AAPL Volume"])
    metadata = pd.DataFrame(
        {"name": ["Apple", "Microsoft"], "exchange": ["NASDAQ", "NASDAQ"], "currency": ["USD", "USD"]},
        index=["AAPL", "MSFT"],
    )

    portfolio_data = app.PortfolioData(
        tickers=("AAPL", "MSFT"),
        prices=price_frame.drop(columns=["AAPL Volume"]).rename(columns={"AAPL Close": "AAPL", "MSFT Close": "MSFT"}),
        collated=price_frame.drop(columns=["AAPL Volume"]).rename(columns={"AAPL Close": "AAPL", "MSFT Close": "MSFT"}),
        price_history_dir=app.SETTINGS.price_history_dir,
        collated_path=app.SETTINGS.export_dir / "streamlit_test_collated.csv",
        start=price_frame.index.min(),
        end=price_frame.index.max(),
        warnings=(),
        used_cache=False,
    )

    monkeypatch.setattr(app, "load_prices", lambda tickers, start, end: portfolio_data)
    monkeypatch.setattr(app, "load_preview_data", lambda tickers, end_date: preview_frame.copy())
    monkeypatch.setattr(app, "gather_metadata", lambda tickers: metadata.copy())

    controls = app.sidebar_controls()

    assert controls["source"] == "download"
    assert controls["tickers"] == ("AAPL", "MSFT")
    assert list(controls["price_data"].columns) == ["AAPL", "MSFT"]
    assert controls["metadata"].index.name == "Ticker"
    assert streamlit_stub.session_state["dca_rate_override"] is True
    assert controls["download_summary"]["tickers"] == ("AAPL", "MSFT")
    assert controls["download_summary"]["warnings"] == ()
    assert controls["download_summary"]["used_cache"] is False
    assert controls["portfolio_data"].collated_path == portfolio_data.collated_path
    assert controls["portfolio_data"].used_cache is False
    assert controls["portfolio_data"].warnings == ()


def test_sidebar_controls_upload_flow(monkeypatch: pytest.MonkeyPatch, streamlit_stub: StreamlitStub) -> None:
    csv_data = io.StringIO("Date,AAPL\n2024-01-01,100\n2024-01-02,101\n")
    streamlit_stub.sidebar.uploaded_file = csv_data
    streamlit_stub.sidebar.slider_values["Annual Return Rate"] = 0.05

    monkeypatch.setattr(app, "load_prices", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_prices not used")))
    monkeypatch.setattr(app, "load_preview_data", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_preview_data not used")))
    monkeypatch.setattr(app, "gather_metadata", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("gather_metadata not used")))

    controls = app.sidebar_controls()

    assert controls["source"] == "upload"
    assert controls["tickers"] == ("AAPL",)
    assert controls["metadata"].loc["AAPL", "name"] == "Provided via CSV"
    assert streamlit_stub.session_state["dca_rate_override"] is True
    assert controls["download_summary"] is None
    assert isinstance(controls["portfolio_data"], app.PortfolioData)
    assert controls["portfolio_data"].used_cache is False


def test_main_renders_dashboard(monkeypatch: pytest.MonkeyPatch, streamlit_stub: StreamlitStub, alt_stub: DummyAlt) -> None:
    price_frame = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 102.0],
            "MSFT": [200.0, 202.0, 201.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    metadata = pd.DataFrame(
        {"name": ["Apple", "Microsoft"], "exchange": ["NASDAQ", "NASDAQ"], "currency": ["USD", "USD"]},
        index=["AAPL", "MSFT"],
    )
    metric_result = app.MetricResults(
        returns=price_frame.pct_change().dropna(),
        expected=pd.Series({"AAPL": 0.1, "MSFT": 0.2}),
        volatility=pd.Series({"AAPL": 0.15, "MSFT": 0.25}),
        sharpe=pd.Series({"AAPL": 0.65, "MSFT": 0.80}),
    )

    streamlit_stub.set_button_state("Compute Metrics", True)
    streamlit_stub.set_button_state("Optimise Portfolio", True)

    portfolio_data = app.PortfolioData(
        tickers=("AAPL", "MSFT"),
        prices=price_frame,
        collated=price_frame,
        price_history_dir=app.SETTINGS.price_history_dir,
        collated_path=app.SETTINGS.export_dir / "streamlit_demo_collated.csv",
        start=price_frame.index.min(),
        end=price_frame.index.max(),
        warnings=(),
        used_cache=False,
    )

    def fake_sidebar_controls():
        return {
            "data": price_frame,
            "price_data": price_frame,
            "portfolio_data": portfolio_data,
            "tickers": ("AAPL", "MSFT"),
            "preview": price_frame,
            "metadata": metadata,
            "start": dt.date(2024, 1, 1),
            "end": dt.date(2024, 1, 3),
            "source": "download",
            "dca_initial": 1000.0,
            "dca_monthly": 250.0,
            "dca_months": 24,
            "dca_rate": 0.08,
            "download_summary": {
                "tickers": ("AAPL", "MSFT"),
                "price_history_dir": str(app.SETTINGS.price_history_dir),
                "collated_path": str(portfolio_data.collated_path),
                "start": price_frame.index.min().isoformat(),
                "end": price_frame.index.max().isoformat(),
                "rows": price_frame.shape[0],
                "columns": price_frame.shape[1],
                "warnings": (),
                "used_cache": False,
            },
        }

    plot_calls: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(app, "sidebar_controls", fake_sidebar_controls)
    monkeypatch.setattr(app, "compute_metrics", lambda _: metric_result)
    monkeypatch.setattr(app, "plot_cumulative_returns", lambda df: plot_calls.update({"returns": df.copy()}))
    monkeypatch.setattr(app, "optimise_weights", lambda result: SimpleNamespace(allocations={"AAPL": 0.6, "MSFT": 0.4}))
    monkeypatch.setattr(app, "plot_weights", lambda weights: plot_calls.update({"weights": pd.Series(weights.allocations)}))
    monkeypatch.setattr(
        app,
        "render_dca_projection",
        lambda *args, **kwargs: pd.DataFrame({"Months": [0, 1], "Balance": [1000.0, 1100.0], "Contributions": [1000.0, 1250.0]}),
    )

    app.main()

    assert streamlit_stub.title_calls[0] == "PySharpe Interactive Dashboard"
    assert any("Downloaded" in msg for msg in streamlit_stub.success_calls)
    assert any("Price Preview" in call for call in streamlit_stub.subheader_calls)
    assert "returns" in plot_calls and "weights" in plot_calls
    assert len(streamlit_stub.download_button_calls) == 3
    assert streamlit_stub.placeholder_calls[2].emptied is True
