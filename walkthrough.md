# AMIE-APO: Quant Intelligence & Portfolio Optimization Platform

AMIE-APO is a production-grade quant intelligence platform designed for regime-aware portfolio optimization, real-time execution, and institutional-grade observability.

The system integrates multi-factor market intelligence (regime, volatility, liquidity), a high-performance optimization engine (C++/Numba), and a fully audited execution pipeline with real-time monitoring and reporting.

---

## 📦 System Capabilities

- **Architecture Documentation**  
  Includes a full system design with layered separation (Intelligence, Optimization, Execution, Observability).

- **Execution & Audit Logging**  
  Every optimization cycle is persisted using a structured JSON schema for traceability and debugging.

- **Validation Framework**  
  End-to-end automated testing covering numerical consistency, execution correctness, and failure handling.

- **Reporting Layer**  
  Generates analytical reports including Sharpe Ratio, PnL, Drawdown, and execution metrics from real system outputs.

---

## 🔁 End-to-End Flow

**Market Data** → **AMIS** (Regime + Volatility + Liquidity)  
→ **Portfolio Optimization** (C++/Numba)  
→ **Pre-Trade Risk Enforcement** → **Order Manager**  
→ **Alpaca Execution** → **Snapshot Logging**  
→ **Dashboard & Report Generation**

---

## 📊 System Intelligence Preview

````carousel
![Actual Dashboard Overview](/amie_apo_dashboard_1774911196244.png)
<!-- slide -->
![Actual Quant Intelligence Report](/amie_apo_performance_report_1774911214237.png)
````

> ⚡ **System Proof**: Captured directly from the live AMIE-APO system running in Alpaca paper trading mode.

---

## 🧪 Validation & Testing

- **Unit, Integration, & System Tests**: Comprehensive suite using `pytest`.
- **Numerical Consistency**: Automated validation between C++ and Numba solver outputs.
- **Failure Simulation**: Resilience testing for Redis and API connectivity fallbacks.
- **CI/CD Pipeline**: Infrastructure in place for enforcing coverage and regression safety.

---

## 🏁 System Maturity

**Production-grade (Paper Trading Validated)**

- Fully integrated execution pipeline
- Real-time observability and alerting
- Automated validation and regression protection
- Audit-ready logging and reporting

---

## ⚙️ Execution Mode

- **DRY Mode**: Safe simulation (no trades sent).
- **LIVE Mode (Paper)**: Executes trades via Alpaca paper API for validation.
- **LIVE Mode (Real)**: Disabled by default for risk management.
