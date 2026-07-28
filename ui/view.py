import csv
import re
import threading
from datetime import datetime

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the '3d' projection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Frame, Label, Button, Canvas, StringVar, Entry, RIGHT, LEFT, Y, BOTH, ttk, filedialog

# ordinal so sorting "Shortable" is meaningful (scarcer borrow = higher)
_SHORTABLE_ORDER = {"None!": 4, "Hard": 3, "Medium": 2, "Easy": 1}


def _column_sort_key(col, value, ascending):
    """Key for sorting one Treeview column; missing/unparseable values always sink
    to the bottom regardless of sort direction (handled via the leading tuple flag)."""
    if col == "Ticker":
        return (0, value if ascending else tuple(-ord(c) for c in str(value)))

    if col == "Shortable":
        num = _SHORTABLE_ORDER.get(value)
        if num is None:
            return (1, 0)
        return (0, num if ascending else -num)

    if col == "TTM":
        # momentum % is always the trailing signed-percentage token, e.g. "FIRE 2d -6.6%"
        m = re.findall(r"[+-]?\d+\.?\d*%", str(value))
        if not m:
            return (1, 0)
        num = float(m[-1].replace("%", ""))
        return (0, num if ascending else -num)

    try:
        num = float(str(value).replace("%", "").replace("*", "").replace(",", ""))
        return (0, num if ascending else -num)
    except (TypeError, ValueError):
        return (1, 0)

# ---------- palette ----------
BG           = "#eef1f6"   # app background
CARD_BG      = "#ffffff"
CARD_BORDER  = "#d9dee8"
HEADER_BG    = "#181824"   # dark header bars
PANEL_BG     = "#e4e9f4"   # settings panel background
ACCENT       = "#4f6df5"
ACCENT_DARK  = "#3a56d6"
PRIME_A      = "#e3f4e8"   # prime row stripes
PRIME_B      = "#d2ecd9"
PRIME_ACCENT = "#2fa360"
SUB_A        = "#fdf3d7"   # subprime row stripes
SUB_B        = "#f8ecc2"
SUB_ACCENT   = "#d99a1f"
TEXT_DARK    = "#1a1a2e"
TEXT_MUTED   = "#6b7280"


class View:
    REFRESH_MS = 60000  # 1-minute refresh interval

    # Initializes the GUI layout and all tabs
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.root.title("Short Squeeze Screener")
        self.root.geometry("1460x830")
        self.root.minsize(1100, 600)
        self.root.configure(bg=BG)

        self._setup_style()

        self.tab_control = ttk.Notebook(self.root)
        self.screener_tab = Frame(self.tab_control, bg=BG)
        self.chart_tab = Frame(self.tab_control, bg=BG)
        self.breaking_tab = Frame(self.tab_control, bg=BG)
        self.correlation_tab = Frame(self.tab_control, bg=BG)

        self.tab_control.add(self.screener_tab, text="  📈  Screener  ")
        self.tab_control.add(self.chart_tab, text="  📊  Stock Chart  ")
        self.tab_control.add(self.breaking_tab, text="  📢  Breaking News  ")
        self.tab_control.add(self.correlation_tab, text="  📉  Correlation  ")
        self.tab_control.pack(expand=1, fill="both")

        # background-refresh guards (never fetch on the UI thread)
        self._screener_busy = False
        self._news_busy = False
        self._corr_fig = None
        self._corr_ax = None
        self._corr_canvas = None
        self._chart_fig = None
        self._chart_ax = None
        self._chart_canvas = None
        self._closing = False  # set on window close so background threads stop touching Tk

        # screener table sort state (one per system, since each has its own columns)
        # + last-fetched results (so re-sorting / re-thresholding never re-fetches)
        self._sort_state = {
            "my":     {"column": None, "ascending": True},
            "legacy": {"column": None, "ascending": True},
        }
        self._last_results = {"my_prime": [], "my_subprime": [], "legacy_prime": [], "legacy_subprime": []}
        self._threshold_busy = False

        self.build_screener_panel(self.screener_tab)
        self.build_chart_panel(self.chart_tab)
        self.build_breaking_news_tab(self.breaking_tab)
        self.build_correlation_panel(self.correlation_tab)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10), padding=(14, 9),
                        background="#dde3ee", foreground=TEXT_DARK, borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", CARD_BG)],
                  foreground=[("selected", ACCENT_DARK)])
        style.configure("Treeview", font=("Segoe UI", 10), rowheight=27,
                        background=CARD_BG, fieldbackground=CARD_BG, borderwidth=0)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"),
                        background="#2b2d42", foreground="white",
                        relief="flat", padding=(4, 7))
        style.map("Treeview.Heading", background=[("active", "#3a3d5c")])
        style.map("Treeview",
                  background=[("selected", "#dbe4ff")],
                  foreground=[("selected", TEXT_DARK)])
        style.configure("Vertical.TScrollbar", background="#c7cede", troughcolor=BG,
                        borderwidth=0, arrowsize=13)

    # ---------------- shared: scrollable container ----------------

    def _make_scrollable(self, parent, bg=BG):
        """Returns (container_to_pack, inner_frame_to_fill). Adds a vertical
        scrollbar and lets the mouse wheel scroll while hovering over it."""
        container = Frame(parent, bg=bg)
        canvas = Canvas(container, bg=bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner = Frame(canvas, bg=bg)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))

        def _wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        return container, inner

    # ---------------- Screener tab ----------------

    MY_COLS = ["Ticker", "Price", "Float (M)", "Rel Vol", "Change %", "SI %",
               "SI % (Live)", "DTC", "Short Vol %", "CTB", "Shortable", "TTM", "Squeeze Score"]
    MY_COL_WIDTHS = {
        "Ticker": 70, "Price": 65, "Float (M)": 75, "Rel Vol": 70,
        "Change %": 75, "SI %": 60, "SI % (Live)": 90, "DTC": 55,
        "Short Vol %": 90, "CTB": 65, "Shortable": 80, "TTM": 140,
        "Squeeze Score": 105,
    }
    def build_screener_panel(self, parent):
        header = Frame(parent, bg=HEADER_BG, pady=9)
        header.pack(fill="x")
        Label(header, text="  Short Squeeze Screener", font=("Segoe UI", 13, "bold"),
              bg=HEADER_BG, fg="white").pack(side=LEFT)
        self.screener_status = StringVar(value="Loading…")
        Label(header, textvariable=self.screener_status, font=("Segoe UI", 9),
              bg=HEADER_BG, fg="#9aa0b5").pack(side=RIGHT, padx=10)
        Button(header, text="⬇ Download CSV", command=self._download_csv,
               bg="#2b2d42", fg="white", activebackground="#3a3d5c", activeforeground="white",
               font=("Segoe UI", 9, "bold"), relief="flat", padx=10, pady=3,
               cursor="hand2", bd=0).pack(side=RIGHT, padx=(10, 0))

        self._build_threshold_panel(parent)

        scroll_container, self.screener_body = self._make_scrollable(parent, bg=BG)
        scroll_container.pack(expand=True, fill="both")

        self.refresh_screener_panel()

    def _build_threshold_panel(self, parent):
        """Lets a user retune the numeric cutoffs of BOTH classification
        systems without changing what factors each one uses."""
        t = self.controller.get_thresholds()
        outer = Frame(parent, bg=BG)
        outer.pack(fill="x")
        panel = Frame(outer, bg=PANEL_BG, highlightthickness=1, highlightbackground=CARD_BORDER)
        panel.pack(fill="x", padx=10, pady=8, ipady=6)

        def field(row, label, value, width=6):
            Label(row, text=label, font=("Segoe UI", 9), bg=PANEL_BG, fg=TEXT_DARK).pack(side=LEFT, padx=(10, 3))
            var = StringVar(value=str(value))
            Entry(row, textvariable=var, width=width, font=("Segoe UI", 9),
                  relief="flat", highlightthickness=1, highlightbackground=CARD_BORDER,
                  highlightcolor=ACCENT).pack(side=LEFT, ipady=2)
            return var

        row1 = Frame(panel, bg=PANEL_BG)
        row1.pack(fill="x", pady=(4, 4))
        Label(row1, text="⚙  My Prime Setup thresholds:", font=("Segoe UI", 9, "bold"),
              bg=PANEL_BG, fg=TEXT_DARK).pack(side=LEFT, padx=(10, 0))
        self.var_my_pressure = field(row1, "Pressure ≥", t["my"]["pressure"])
        self.var_my_ignition = field(row1, "Ignition ≥", t["my"]["ignition"])

        row2 = Frame(panel, bg=PANEL_BG)
        row2.pack(fill="x")
        Label(row2, text="⚙  Previous Prime Setup thresholds:", font=("Segoe UI", 9, "bold"),
              bg=PANEL_BG, fg=TEXT_DARK).pack(side=LEFT, padx=(10, 0))
        self.var_legacy_price_min  = field(row2, "Price $", t["legacy"]["price_min"])
        self.var_legacy_price_max  = field(row2, "to", t["legacy"]["price_max"])
        self.var_legacy_change_min = field(row2, "Change % ≥", t["legacy"]["change_min"])
        self.var_legacy_relvol_min = field(row2, "Rel Vol ≥", t["legacy"]["relvol_min"])
        self.var_legacy_si_min     = field(row2, "SI % ≥", t["legacy"]["si_min"])

        Button(row2, text="Apply", command=self._on_apply_thresholds,
               bg=ACCENT, fg="white", font=("Segoe UI", 9, "bold"), activebackground=ACCENT_DARK,
               relief="flat", padx=12, pady=2, cursor="hand2", bd=0).pack(side=LEFT, padx=(14, 6))
        self.threshold_status = StringVar(value="")
        Label(row2, textvariable=self.threshold_status, font=("Segoe UI", 8, "italic"),
              bg=PANEL_BG, fg=TEXT_MUTED).pack(side=LEFT, padx=6)

    def _on_apply_thresholds(self):
        if self._threshold_busy:
            return
        try:
            my_pressure = float(self.var_my_pressure.get())
            my_ignition = float(self.var_my_ignition.get())
            price_min  = float(self.var_legacy_price_min.get())
            price_max  = float(self.var_legacy_price_max.get())
            change_min = float(self.var_legacy_change_min.get())
            relvol_min = float(self.var_legacy_relvol_min.get())
            si_min     = float(self.var_legacy_si_min.get())
        except ValueError:
            self.threshold_status.set("Enter numbers only")
            return
        if price_min > price_max:
            self.threshold_status.set("Price min must be ≤ max")
            return

        self.controller.set_my_thresholds(my_pressure, my_ignition)
        self.controller.set_legacy_thresholds(price_min, price_max, change_min, relvol_min, si_min)

        self._threshold_busy = True
        self.threshold_status.set("Applying…")
        threading.Thread(target=self._reclassify_worker, daemon=True).start()

    def _reclassify_worker(self):
        try:
            result = self.controller.reclassify()
            if not self._closing:
                self.root.after(0, lambda: self._render_screener(result, reclassified=True))
        except Exception as e:
            print(f"[Screener] Reclassify error: {e}")
            if not self._closing:
                self.root.after(0, lambda: self.threshold_status.set("Error — see terminal"))
        finally:
            self._threshold_busy = False

    def refresh_screener_panel(self):
        """Kick off a background fetch; never blocks the window."""
        if self._closing:
            return
        if not self._screener_busy:
            self._screener_busy = True
            self.screener_status.set("Refreshing…")
            threading.Thread(target=self._screener_worker, daemon=True).start()
        self.root.after(self.REFRESH_MS, self.refresh_screener_panel)

    def _screener_worker(self):
        try:
            result = self.controller.get_screener_results()
            if not self._closing:
                self.root.after(0, lambda: self._render_screener(result))
        except Exception as e:
            print(f"[Screener] Refresh error: {e}")
            if not self._closing:
                self.root.after(0, lambda: self.screener_status.set("Error — see terminal"))
        finally:
            self._screener_busy = False

    def _render_screener(self, result, reclassified=False):
        self._last_results = result
        self._draw_screener_tables()
        if reclassified:
            self.threshold_status.set(f"Applied {datetime.now().strftime('%H:%M:%S')}")
        else:
            self.screener_status.set(f"Last updated {datetime.now().strftime('%H:%M:%S')}  •  auto-refreshes every 60s")

    def _on_sort_click(self, sort_key, col):
        """Column header clicked — toggle direction if same column, else start ascending."""
        state = self._sort_state[sort_key]
        if state["column"] == col:
            state["ascending"] = not state["ascending"]
        else:
            state["column"] = col
            state["ascending"] = True
        self._draw_screener_tables()  # re-sort only — no re-fetch

    def _sorted_rows(self, data, cols, sort_key):
        state = self._sort_state[sort_key]
        if state["column"] not in cols:
            return data
        idx = cols.index(state["column"])
        return sorted(data, key=lambda r: _column_sort_key(state["column"], r[idx], state["ascending"]))

    def _download_csv(self):
        """Export all 4 sections into one CSV, tagged with a 'Setup' column
        so they can still be told apart / filtered after combining — all four
        share the exact same 13 data columns, so one file is enough."""
        path = filedialog.asksaveasfilename(
            title="Save screener results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"screener_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        if not path:
            return  # user cancelled

        sections = [
            ("My Prime", self._last_results.get("my_prime", [])),
            ("My Subprime", self._last_results.get("my_subprime", [])),
            ("Previous Prime", self._last_results.get("legacy_prime", [])),
            ("Previous Subprime", self._last_results.get("legacy_subprime", [])),
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Setup"] + self.MY_COLS)
                for label, rows in sections:
                    for row in rows:
                        writer.writerow([label] + list(row))
            total = sum(len(rows) for _, rows in sections)
            self.screener_status.set(f"Saved {total} rows to {path}")
        except Exception as e:
            print(f"[Screener] CSV export failed: {e}")
            self.screener_status.set("Export failed — see terminal")

    def _draw_screener_tables(self):
        try:
            for widget in self.screener_body.winfo_children():
                widget.destroy()

            def add_section(title, data, cols, col_widths, sort_key, accent, stripe_a, stripe_b, badge):
                bar = Frame(self.screener_body, bg=BG)
                bar.pack(fill="x", padx=12, pady=(14, 4))
                Frame(bar, bg=accent, width=4, height=20).pack(side=LEFT, padx=(0, 8))
                Label(bar, text=badge, font=("Segoe UI", 12), bg=BG).pack(side=LEFT)
                Label(bar, text=f" {title}", font=("Segoe UI", 12, "bold"),
                      bg=BG, fg=TEXT_DARK).pack(side=LEFT)
                Label(bar, text=f"  {len(data)} stock{'s' if len(data) != 1 else ''}",
                      font=("Segoe UI", 9), bg=BG, fg=TEXT_MUTED).pack(side=LEFT, pady=2)

                if not data:
                    Label(self.screener_body, text="No stocks currently match this criteria",
                          font=("Segoe UI", 10, "italic"), bg=BG, fg=TEXT_MUTED).pack(pady=(0, 8), padx=14, anchor="w")
                    return

                card = Frame(self.screener_body, bg=CARD_BG, highlightthickness=1, highlightbackground=CARD_BORDER)
                card.pack(fill="x", padx=12, pady=(0, 6))

                sorted_data = self._sorted_rows(data, cols, sort_key)
                tree = ttk.Treeview(card, columns=cols, show="headings",
                                    height=max(len(sorted_data), 3))
                state = self._sort_state[sort_key]
                for col in cols:
                    label = col
                    if col == state["column"]:
                        label = f"{col} {'▲' if state['ascending'] else '▼'}"
                    tree.heading(col, text=label, command=lambda c=col: self._on_sort_click(sort_key, c))
                    tree.column(col, anchor="center", width=col_widths.get(col, 90))
                tree.tag_configure("a", background=stripe_a)
                tree.tag_configure("b", background=stripe_b)
                for i, row in enumerate(sorted_data):
                    tree.insert("", "end", values=row, tags=("a" if i % 2 == 0 else "b",))
                tree.pack(fill="x", padx=1, pady=1)

            add_section("My Prime Setup", self._last_results.get("my_prime", []),
                        self.MY_COLS, self.MY_COL_WIDTHS, "my",
                        PRIME_ACCENT, PRIME_A, PRIME_B, "⭐")
            add_section("My Subprime Setup", self._last_results.get("my_subprime", []),
                        self.MY_COLS, self.MY_COL_WIDTHS, "my",
                        SUB_ACCENT, SUB_A, SUB_B, "⚠️")
            add_section("Previous Prime Setup", self._last_results.get("legacy_prime", []),
                        self.MY_COLS, self.MY_COL_WIDTHS, "legacy",
                        PRIME_ACCENT, PRIME_A, PRIME_B, "⭐")
            add_section("Previous Subprime Setup", self._last_results.get("legacy_subprime", []),
                        self.MY_COLS, self.MY_COL_WIDTHS, "legacy",
                        SUB_ACCENT, SUB_A, SUB_B, "⚠️")

            Frame(self.screener_body, bg=BG, height=10).pack()  # bottom breathing room
        except Exception as e:
            print(f"[Screener] Render error: {e}")

    # ---------------- Breaking news tab ----------------

    def build_breaking_news_tab(self, parent):
        header = Frame(parent, bg=HEADER_BG, pady=9)
        header.pack(fill="x")
        Label(header, text="  Breaking News", font=("Segoe UI", 13, "bold"),
              bg=HEADER_BG, fg="white").pack(side=LEFT)
        self.news_status = StringVar(value="Loading…")
        Label(header, textvariable=self.news_status, font=("Segoe UI", 9),
              bg=HEADER_BG, fg="#9aa0b5").pack(side=RIGHT, padx=10)

        scroll_container, self.news_body = self._make_scrollable(parent, bg=BG)
        scroll_container.pack(expand=True, fill="both")

        self.refresh_breaking_news_tab()

    def refresh_breaking_news_tab(self):
        if self._closing:
            return
        if not self._news_busy:
            self._news_busy = True
            threading.Thread(target=self._news_worker, daemon=True).start()
        self.root.after(self.REFRESH_MS, self.refresh_breaking_news_tab)

    def _news_worker(self):
        try:
            headlines = self.controller.get_positive_news()
            if not self._closing:
                self.root.after(0, lambda: self._render_news(headlines))
        except Exception as e:
            print(f"[News] Refresh error: {e}")
        finally:
            self._news_busy = False

    def _confidence_color(self, score):
        if score >= 80: return "#1a7a3c", "#e6f4ec"
        if score >= 65: return "#b35c00", "#fff4e5"
        return "#555555", "#f5f5f5"

    def _render_news(self, headlines):
        try:
            for widget in self.news_body.winfo_children():
                widget.destroy()

            if not headlines:
                Label(self.news_body, text="No positive news at this time.",
                      font=("Segoe UI", 11), fg=TEXT_MUTED, bg=BG).pack(pady=40)
            else:
                for item in headlines:
                    headline = item["headline"]
                    tickers = ", ".join(item["tickers"]) if item["tickers"] else "—"
                    url = item["url"]
                    confidence = int(item["confidence_score"] * 100)
                    text_color, bg_color = self._confidence_color(confidence)

                    card = Frame(self.news_body, bg=CARD_BG, relief="flat",
                                 highlightthickness=1, highlightbackground=CARD_BORDER)
                    card.pack(fill="x", padx=12, pady=5)

                    badge_color = "#27ae60" if confidence >= 80 else "#e67e22" if confidence >= 65 else "#95a5a6"
                    Frame(card, bg=badge_color, width=5).pack(side=LEFT, fill="y")

                    content = Frame(card, bg=CARD_BG, padx=12, pady=10)
                    content.pack(side=LEFT, fill="both", expand=True)

                    lbl = Label(content, text=headline, font=("Segoe UI", 10, "bold"),
                                fg=TEXT_DARK, bg=CARD_BG, justify="left",
                                wraplength=900, cursor="hand2")
                    lbl.pack(anchor="w")
                    lbl.bind("<Button-1>", lambda e, u=url: self.controller.open_url(u))

                    bottom = Frame(content, bg=CARD_BG)
                    bottom.pack(anchor="w", pady=(5, 0))
                    Label(bottom, text=f"  {tickers}  ", font=("Segoe UI", 9, "bold"),
                          fg="#2c3e7a", bg="#eef0fb", padx=6, pady=2).pack(side=LEFT, padx=(0, 8))
                    Label(bottom, text=f"Confidence: {confidence}%", font=("Segoe UI", 9),
                          fg=text_color, bg=bg_color, padx=6, pady=2).pack(side=LEFT)

            self.news_status.set(f"Updated {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"[News] Render error: {e}")

    # ---------------- Correlation tab ----------------

    def build_correlation_panel(self, parent):
        header = Frame(parent, bg=HEADER_BG, pady=9)
        header.pack(fill="x")
        Label(header, text="  SI% vs Price Change vs Rel Vol", font=("Segoe UI", 13, "bold"),
              bg=HEADER_BG, fg="white").pack(side=LEFT)
        Label(header, text="drag to rotate", font=("Segoe UI", 9, "italic"),
              bg=HEADER_BG, fg="#9aa0b5").pack(side=LEFT, padx=10)
        Button(header, text="⟲  Reset View", command=self._reset_correlation_view,
               bg="#2b2d42", fg="white", activebackground="#3a3d5c", activeforeground="white",
               font=("Segoe UI", 9, "bold"), relief="flat", padx=10, pady=3,
               cursor="hand2", bd=0).pack(side=RIGHT, padx=10)

        self.correlation_body = Frame(parent, bg=BG)
        self.correlation_body.pack(expand=True, fill="both")

        self.refresh_correlation_panel()

    def _reset_correlation_view(self):
        if self._corr_ax is not None and self._corr_canvas is not None:
            self._corr_ax.view_init(elev=20, azim=-60)
            self._corr_canvas.draw()

    @staticmethod
    def _pearson(pairs):
        if len(pairs) < 2:
            return None
        xs, ys = zip(*pairs)
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            return None
        return float(np.corrcoef(xs, ys)[0, 1])

    def refresh_correlation_panel(self):
        """Reads cached data from the last screener refresh — no network calls."""
        if self._closing:
            return
        try:
            data = self.controller.get_correlation_data()
            # rows: (ticker, si_official, si_live, change_pct, rel_vol)
            rows = [(t, o, l, c, rv) for t, o, l, c, rv in data
                    if c is not None and (o is not None or l is not None)]

            # Build the figure/canvas ONCE and redraw into it. Rebuilding it
            # every refresh orphaned a Tk PhotoImage each cycle; those got
            # garbage-collected on whatever thread happened to trigger GC
            # (usually a background worker), and PhotoImage.__del__ calls into
            # Tcl — which is fatal off the main thread ("main thread is not in
            # main loop", then "Tcl_AsyncDelete: async handler deleted by the
            # wrong thread", which aborts the process). Reusing the canvas also
            # keeps the user's rotation instead of resetting it every minute.
            if self._corr_canvas is None:
                fig = plt.figure(figsize=(9, 6.5), facecolor=BG)
                ax = fig.add_subplot(111, projection="3d")
                self._corr_fig, self._corr_ax = fig, ax
                self._corr_canvas = FigureCanvasTkAgg(fig, master=self.correlation_body)
                self._corr_canvas.get_tk_widget().pack(fill="both", expand=True,
                                                       padx=10, pady=8)
                elev, azim = 20, -60
            else:
                ax = self._corr_ax
                elev, azim = ax.elev, ax.azim   # preserve however the user rotated it
                ax.clear()
            ax.set_facecolor("white")

            if not rows:
                ax.text2D(0.5, 0.5, "Waiting for screener data…", ha="center", va="center",
                           transform=ax.transAxes, fontsize=13, color=TEXT_MUTED)
            else:
                official_pts, live_pts = [], []  # (si, change) pairs — RelVol is a 3rd, shared axis
                for t, o, l, c, rv in rows:
                    z = np.log1p(rv) if rv is not None and rv > 0 else 0.0
                    # connector shows how far the live estimate moved from the official number
                    if o is not None and l is not None and abs(l - o) > 0.05:
                        ax.plot([o, l], [c, c], [z, z], color="#c9ced9", linewidth=1, zorder=1)
                    if o is not None:
                        official_pts.append((o, c))
                        ax.scatter(o, c, z, color="#9aa0b5", alpha=0.75, s=42, zorder=2)
                    if l is not None:
                        live_pts.append((l, c))
                        ax.scatter(l, c, z, color="#e74c3c", alpha=0.85, s=48, zorder=3)
                        ax.text(l, c, z, t, fontsize=7.5, color="#333", zorder=4)
                    elif o is not None:
                        ax.text(o, c, z, t, fontsize=7.5, color="#333", zorder=4)

                # trend line through the live estimates, projected onto the RelVol=0 floor
                if len(live_pts) >= 3:
                    xs, ys = zip(*live_pts)
                    if len(set(xs)) >= 2:
                        m, b = np.polyfit(xs, ys, 1)
                        xr = np.linspace(min(xs), max(xs), 50)
                        ax.plot(xr, m * xr + b, [0] * len(xr), color="#e74c3c", linewidth=1.2,
                                linestyle="--", alpha=0.6, zorder=2)

                ax.scatter([], [], [], color="#9aa0b5", label=f"Official SI% (n={len(official_pts)})")
                ax.scatter([], [], [], color="#e74c3c", label=f"SI% (Live) (n={len(live_pts)})")
                ax.legend(loc="upper left", fontsize=9)

                r_off = self._pearson(official_pts)
                r_live = self._pearson(live_pts)
                r_off_str = f"{r_off:.2f}" if r_off is not None else "n/a"
                r_live_str = f"{r_live:.2f}" if r_live is not None else "n/a"
                ax.set_title(f"r(official) = {r_off_str}    r(live) = {r_live_str}   "
                             f"(SI% vs Change%; Z = Rel Vol)", fontsize=11)

            ax.set_xlabel("Short Interest %")
            ax.set_ylabel("Price Change % (today)")
            ax.set_zlabel("Rel Vol (log scale)")
            ax.view_init(elev=elev, azim=azim)
            self._corr_fig.tight_layout()
            self._corr_canvas.draw_idle()
        except Exception as e:
            print(f"[Correlation] Refresh error: {e}")
        finally:
            if not self._closing:
                self.root.after(self.REFRESH_MS, self.refresh_correlation_panel)

    # ---------------- Stock chart tab ----------------

    def build_chart_panel(self, parent):
        bar = Frame(parent, bg=HEADER_BG, pady=9)
        bar.pack(fill="x")
        Label(bar, text="  Stock Chart", font=("Segoe UI", 13, "bold"),
              bg=HEADER_BG, fg="white").pack(side=LEFT)

        frame = Frame(parent, bg=BG)
        frame.pack(pady=18)
        Label(frame, text="Enter a stock ticker:", font=("Segoe UI", 10),
              bg=BG, fg=TEXT_DARK).pack()
        self.ticker_var = StringVar()
        Entry(frame, textvariable=self.ticker_var, width=20, font=("Segoe UI", 10),
              relief="flat", highlightthickness=1, highlightbackground=CARD_BORDER,
              highlightcolor=ACCENT).pack(pady=6, ipady=3)
        Button(frame, text="📈 Load Chart", command=self.plot_chart,
               bg=ACCENT, fg="white", font=("Segoe UI", 10, "bold"), activebackground=ACCENT_DARK,
               relief="flat", padx=14, pady=5, cursor="hand2", bd=0).pack()

        self.chart_frame = Frame(parent, bg=BG)
        self.chart_frame.pack(expand=True, fill="both")

    def plot_chart(self):
        ticker = self.ticker_var.get().upper().strip()
        if not ticker:
            return
        try:
            df = yf.download(ticker, period="5d", interval="30m")
            if df.empty:
                raise ValueError("No data returned")

            # Same build-once rule as the correlation panel: replacing the
            # canvas orphans a Tk PhotoImage, whose __del__ can fire on a
            # background thread and take the whole process down with it.
            if self._chart_canvas is None:
                self._chart_fig, self._chart_ax = plt.subplots(figsize=(6, 4))
                self._chart_canvas = FigureCanvasTkAgg(self._chart_fig,
                                                       master=self.chart_frame)
                self._chart_canvas.get_tk_widget().pack(fill="both", expand=True)
            ax = self._chart_ax
            ax.clear()

            df["Close"].plot(ax=ax)
            ax.set_title(f"{ticker} - 5 Day Price Chart")
            ax.set_ylabel("Price")
            self._chart_canvas.draw_idle()
        except Exception as e:
            print(f"[Chart] Failed to load chart: {e}")

    def on_close(self):
        """Stop background threads from touching Tk, close figures, then destroy the window."""
        self._closing = True
        plt.close("all")
        self.root.destroy()
