"""Part B diagnostic: show exact yfinance field + EDGAR XBRL tag for each reconciled metric."""
import json
from datetime import datetime

import yfinance as yf

ticker = "AAPL"
print(f"\n{'='*70}")
print(f"RECONCILIATION DIAGNOSTIC: {ticker}")
print(f"{'='*70}\n")

# ── yfinance side ─────────────────────────────────────────────────────────────
yf_ticker = yf.Ticker(ticker)
info = yf_ticker.info

print("── yfinance info fields ──")
for field in ["totalRevenue", "revenuePerShare", "netIncomeToCommon",
              "sharesOutstanding", "impliedSharesOutstanding",
              "floatShares", "totalDebt", "totalCash",
              "trailingEps", "earningsTimestamp"]:
    print(f"  {field}: {info.get(field)}")

print()

# yfinance financials (annual)
print("── yfinance annual income statement (first 2 cols) ──")
try:
    fin = yf_ticker.financials  # annual
    print(f"  Columns (periods): {list(fin.columns[:4])}")
    for row in ["Total Revenue", "Net Income", "Basic EPS", "Diluted EPS"]:
        if row in fin.index:
            vals = fin.loc[row].iloc[:2].to_dict()
            print(f"  {row}: {vals}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# yfinance quarterly financials
print("── yfinance quarterly income statement (first 2 cols) ──")
try:
    qfin = yf_ticker.quarterly_financials
    print(f"  Columns (periods): {list(qfin.columns[:4])}")
    for row in ["Total Revenue", "Net Income"]:
        if row in qfin.index:
            vals = qfin.loc[row].iloc[:2].to_dict()
            print(f"  {row}: {vals}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# yfinance balance sheet
print("── yfinance annual balance sheet ──")
try:
    bs = yf_ticker.balance_sheet
    print(f"  Columns (periods): {list(bs.columns[:2])}")
    for row in ["Ordinary Shares Number", "Share Issued", "Common Stock"]:
        if row in bs.index:
            vals = bs.loc[row].iloc[:2].to_dict()
            print(f"  {row}: {vals}")
    # Check all rows containing "share" or "stock"
    share_rows = [r for r in bs.index if "share" in r.lower() or "stock" in r.lower()]
    print(f"  Share-related rows: {share_rows}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# ── EDGAR XBRL side ───────────────────────────────────────────────────────────
print("── EDGAR XBRL facts for AAPL ──")
import httpx

# Get CIK for AAPL
try:
    r = httpx.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "DeepDive Research vandit@deductive.ai"},
        timeout=15
    )
    tickers_map = r.json()
    cik = None
    for entry in tickers_map.values():
        if entry["ticker"].upper() == ticker:
            cik = str(entry["cik_str"]).zfill(10)
            break
    print(f"  CIK: {cik}")
except Exception as e:
    print(f"  CIK lookup error: {e}")
    cik = None

if cik:
    try:
        r2 = httpx.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers={"User-Agent": "DeepDive Research vandit@deductive.ai"},
            timeout=30
        )
        facts = r2.json()
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        dei = facts.get("facts", {}).get("dei", {})

        print(f"\n  us-gaap concepts available (total): {len(us_gaap)}")

        # Revenue concepts
        print("\n── Revenue XBRL candidates ──")
        for concept in ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                        "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"]:
            if concept in us_gaap:
                units = us_gaap[concept].get("units", {})
                usd = units.get("USD", [])
                # Filter to 10-K/10-Q only, last 3 entries
                filings_10k = [f for f in usd if f.get("form") in ("10-K", "10-Q")]
                recent = sorted(filings_10k, key=lambda x: x.get("end", ""), reverse=True)[:4]
                print(f"\n  {concept}:")
                for rec in recent:
                    print(f"    form={rec.get('form')}, start={rec.get('start')}, end={rec.get('end')}, "
                          f"val={rec.get('val'):,.0f}, filed={rec.get('filed')}, accn={rec.get('accn','')[:8]}")

        # Net income
        print("\n── Net Income XBRL candidates ──")
        for concept in ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersDiluted"]:
            if concept in us_gaap:
                units = us_gaap[concept].get("units", {})
                usd = units.get("USD", [])
                filings_10k = [f for f in usd if f.get("form") in ("10-K", "10-Q")]
                recent = sorted(filings_10k, key=lambda x: x.get("end", ""), reverse=True)[:4]
                print(f"\n  {concept}:")
                for rec in recent:
                    print(f"    form={rec.get('form')}, start={rec.get('start')}, end={rec.get('end')}, "
                          f"val={rec.get('val'):,.0f}, filed={rec.get('filed')}")

        # Shares
        print("\n── Shares XBRL candidates ──")
        for concept in ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]:
            src = us_gaap if concept in us_gaap else (dei if concept in dei else None)
            ns = "us-gaap" if concept in us_gaap else "dei"
            if src and concept in src:
                units = src[concept].get("units", {})
                shares_u = units.get("shares", [])
                filings_10k = [f for f in shares_u if f.get("form") in ("10-K", "10-Q", "10-K/A")]
                recent = sorted(filings_10k, key=lambda x: x.get("end", ""), reverse=True)[:4]
                print(f"\n  [{ns}] {concept}:")
                for rec in recent:
                    print(f"    form={rec.get('form')}, end={rec.get('end')}, "
                          f"val={rec.get('val'):,.0f}, filed={rec.get('filed')}")

        # What our reconcile.py ACTUALLY fetches — call extract_xbrl_metric directly
        print("\n── What reconcile.py currently fetches via extract_xbrl_metric ──")
        import sys
        sys.path.insert(0, ".")
        from data.edgar import extract_xbrl_metric
        for label, concept, ns in [
            ("revenue", "RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap"),
            ("net_income", "NetIncomeLoss", "us-gaap"),
            ("shares", "CommonStockSharesOutstanding", "us-gaap"),
        ]:
            recs = extract_xbrl_metric(facts, concept, ns, n_periods=2)
            print(f"\n  {label} ({concept}):")
            for rec in recs:
                print(f"    {rec}")

    except Exception as e:
        import traceback
        print(f"  XBRL fetch error: {e}")
        traceback.print_exc()

print()

# ── Side-by-side comparison ───────────────────────────────────────────────────
print("\n── CURRENT reconcile.py output for AAPL ──")
try:
    from analysis.reconcile import get_reconciliation
    recs = get_reconciliation(ticker)
    for r in recs:
        print(f"  {r.metric}: yf={r.yfinance_value:,.0f} edgar={r.edgar_value:,.0f} "
              f"diff={r.diff_pct:.1%} note={r.note[:80]}")
except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()
